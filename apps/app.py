"""
Streamlit UI for Neural Ellipsometry Reflectance.

Three tabs:
  1. Forward Simulator — interactive TMM demonstration
  2. Inverse Predictor — run SpectraNet V4 to invert spectra
  3. Model Performance — parity plots, loss curves, comparison table
"""

import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

from src.paths import artifact_path
from src.tmm_simulator import simulate_reflectance
from src.spectranet import SpectraNet
from src.refiner import refine_prediction
from src.robust_refiner import refine_prediction_guarded_multistart
from src.denoiser import DenoisingAutoencoder
from src.reliability import evaluate_identifiability
from src.thinfilm_visualizer import (
    make_visual_state,
    render_prediction_cards,
    render_thinfilm_visualizer,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

WAVELENGTHS = np.linspace(400, 800, 200)
PARAM_BOUNDS = np.array([[10.0, 300.0], [1.3, 2.5], [0.0, 0.5]],
                        dtype=np.float32)
PARAM_MIN = PARAM_BOUNDS[:, 0]
PARAM_RANGE = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

# ──────────────────────────────────────────────
# Cached model and normalization loading
# ──────────────────────────────────────────────


@st.cache_resource
def load_model():
    model = SpectraNet()
    model.load_state_dict(torch.load(artifact_path("models", "spectranet_v4.pt"),
                                     map_location="cpu",
                                     weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_norm_stats():
    data = np.load(artifact_path("data", "spectra_norm_v4.npz"))
    return data["mean"], data["std"]


@st.cache_resource
def load_denoiser():
    dae = DenoisingAutoencoder()
    dae.load_state_dict(torch.load(artifact_path("models", "denoiser_joint.pt"),
                                    map_location="cpu",
                                    weights_only=True))
    dae.eval()
    return dae


def denoise_spectrum(spectrum, dae):
    """Run denoiser on a raw reflectance spectrum. Returns denoised array."""
    x = torch.from_numpy(spectrum.astype(np.float32)[np.newaxis, :])
    with torch.no_grad():
        out = dae(x).numpy()[0]
    return out


def predict_params(spectrum, model, X_mean, X_std):
    """Normalize spectrum, run model, denormalize output."""
    x = (spectrum.astype(np.float32) - X_mean) / X_std
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x[np.newaxis, :])).numpy()[0]
    return pred_norm * PARAM_RANGE + PARAM_MIN


def predict_with_uncertainty(spectrum, model, X_mean, X_std, n_samples=100):
    """
    Run MC Dropout inference: keep dropout active during inference
    by calling model.train() instead of model.eval(), run n_samples
    forward passes, return mean and std of predictions.

    Returns:
        means: np.array shape (3,) — mean predicted [thickness, n, k]
        stds: np.array shape (3,) — std of predictions [thickness, n, k]
        all_samples: np.array shape (n_samples, 3) — all individual draws
    All in physical units (denormalized).
    """
    x = (spectrum.astype(np.float32) - X_mean) / X_std
    x_tensor = torch.from_numpy(x[np.newaxis, :])  # (1, 200)

    # Enable dropout by switching to train mode
    model.train()
    samples = []
    torch.manual_seed(42)  # deterministic dropout masks for reproducibility
    with torch.no_grad():
        for _ in range(n_samples):
            pred_norm = model(x_tensor).numpy()[0]  # (3,)
            pred_phys = pred_norm * PARAM_RANGE + PARAM_MIN
            samples.append(pred_phys)

    # Restore eval mode for normal inference
    model.eval()

    all_samples = np.array(samples)  # (n_samples, 3)
    means = all_samples.mean(axis=0)
    stds = all_samples.std(axis=0)
    return means, stds, all_samples


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(page_title="Neural Ellipsometry", layout="wide")
st.title("Neural Ellipsometry Reflectance")

# Sidebar controls
with st.sidebar:
    st.subheader("MC Dropout Settings")
    mc_n_samples = st.slider("MC samples", 20, 200, 100, step=10,
                             key="mc_samples",
                             help="Number of stochastic forward passes. "
                                  "Higher = slower but more stable "
                                  "uncertainty estimate.")

tab1, tab2, tab3 = st.tabs([
    "Forward Simulator",
    "Inverse Predictor",
    "Model Performance",
])

# ──────────────────────────────────────────────
# TAB 1: Forward Simulator
# ──────────────────────────────────────────────

with tab1:
    st.header("Transfer Matrix Method: Forward Simulator")

    col_sliders, col_plot = st.columns([1, 2])

    with col_sliders:
        st.subheader("Film Parameters")
        thickness = st.slider("Thickness (nm)", 10, 300, 100, step=1,
                              key="fwd_thick")
        n_val = st.slider("Refractive index n", 1.30, 2.50, 1.46,
                           step=0.01, key="fwd_n")
        k_val = st.slider("Extinction coefficient k", 0.000, 0.500, 0.000,
                           step=0.001, format="%.3f", key="fwd_k")

    spectrum = simulate_reflectance(thickness, n_val, k_val, WAVELENGTHS)

    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(WAVELENGTHS, spectrum, linewidth=2, color="#1f77b4")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title(f"Reflectance: d={thickness} nm, n={n_val:.2f}, "
                     f"k={k_val:.3f}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Min Reflectance", f"{spectrum.min():.4f}")
        st.metric("Max Reflectance", f"{spectrum.max():.4f}")

    with col_info2:
        # Context-aware explanation
        fringe_count = 0
        if len(spectrum) > 2:
            diffs = np.diff(np.sign(np.diff(spectrum)))
            fringe_count = int(np.sum(np.abs(diffs) > 0))

        explanation = []
        explanation.append(
            "**What you are seeing:** The reflectance spectrum of a "
            f"**{thickness} nm** thin film (n={n_val:.2f}, k={k_val:.3f}) "
            "on a silicon substrate, illuminated at normal incidence."
        )

        if k_val < 0.01:
            explanation.append(
                "The film is nearly transparent (k ~ 0), so light passes "
                "through the film and reflects off both the top surface and "
                "the film-substrate interface. These two reflected beams "
                "interfere, creating the oscillating pattern (fringes) you "
                "see in the spectrum."
            )
        else:
            explanation.append(
                f"With k = {k_val:.3f}, the film absorbs some light during "
                "each pass. This damps the interference fringes and "
                "generally reduces overall reflectance compared to a "
                "transparent film."
            )

        if thickness < 50:
            explanation.append(
                "The film is thin enough that less than one full fringe "
                "appears in the visible range. The spectrum shows a smooth "
                "curve rather than oscillations."
            )
        elif thickness > 200:
            explanation.append(
                "The thick film produces many closely-spaced fringes. "
                "In real measurements, spectral resolution limits how "
                "many fringes you can resolve."
            )

        if n_val > 2.0:
            explanation.append(
                f"A high refractive index (n={n_val:.2f}) means strong "
                "reflections at both interfaces, producing high-contrast "
                "fringes with large swings in reflectance."
            )
        elif n_val < 1.5:
            explanation.append(
                f"A low refractive index (n={n_val:.2f}, close to air) "
                "produces weak reflections at the air-film interface, "
                "so fringe contrast is lower."
            )

        st.markdown("\n\n".join(explanation))


# ──────────────────────────────────────────────
# TAB 2: Inverse Predictor
# ──────────────────────────────────────────────

with tab2:
    st.header("SpectraNet V4: Inverse Predictor")

    model = load_model()
    X_mean, X_std = load_norm_stats()

    input_mode = st.radio(
        "Input mode",
        ["Generate from known parameters", "Enter spectrum manually"],
        horizontal=True,
    )

    input_spectrum = None
    has_true_params = False
    true_thick, true_n_val, true_k_val = None, None, None

    if input_mode == "Enter spectrum manually":
        col_text, col_btn = st.columns([3, 1])
        with col_btn:
            example_spec = simulate_reflectance(100.0, 1.46, 0.0, WAVELENGTHS)
            if st.button("Use example spectrum (SiO2 100nm)"):
                st.session_state["spectrum_text"] = ", ".join(
                    f"{v:.6f}" for v in example_spec
                )
        with col_text:
            raw_text = st.text_area(
                "Paste 200 comma-separated reflectance values",
                height=120,
                key="spectrum_text",
            )
        if raw_text.strip():
            try:
                vals = [float(x.strip()) for x in raw_text.split(",")
                        if x.strip()]
                if len(vals) != 200:
                    st.error(f"Expected 200 values, got {len(vals)}. "
                             "Please provide exactly 200 comma-separated "
                             "reflectance values (400-800 nm).")
                else:
                    input_spectrum = np.array(vals, dtype=np.float32)
            except ValueError:
                st.error("Could not parse input. Make sure all values "
                         "are valid numbers separated by commas.")

    else:  # Generate from known parameters
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            gen_thick = st.slider("Thickness (nm)", 10, 300, 242, step=1,
                                  key="inv_thick")
            gen_n = st.slider("Refractive index n", 1.30, 2.50, 1.45,
                              step=0.01, key="inv_n")
            gen_k = st.slider("Extinction coefficient k", 0.000, 0.500,
                              0.200, step=0.001, format="%.3f", key="inv_k")
        with col_s2:
            noise_level = st.slider("Noise level (Gaussian std)",
                                    0.0, 0.02, 0.001, step=0.001,
                                    format="%.3f", key="noise")
            st.caption("Simulates measurement noise on top of the "
                       "ideal TMM spectrum.")
            if noise_level > 0.015:
                st.warning(
                    "High noise level: at std > 0.015, spectral "
                    "information is heavily corrupted. Predictions will "
                    "have large uncertainty and the refiner may not "
                    "converge reliably. Consider that real instruments "
                    "typically operate below std=0.005."
                )

        has_true_params = True
        true_thick, true_n_val, true_k_val = gen_thick, gen_n, gen_k

        clean = simulate_reflectance(gen_thick, gen_n, gen_k, WAVELENGTHS)
        rng = np.random.default_rng()
        noisy = clean + rng.normal(0, noise_level, size=clean.shape)
        noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
        input_spectrum = noisy

        fig_noisy, ax_noisy = plt.subplots(figsize=(8, 3))
        ax_noisy.plot(WAVELENGTHS, clean, label="Clean", alpha=0.5)
        ax_noisy.plot(WAVELENGTHS, noisy, label="Noisy (input)", linewidth=1)
        ax_noisy.set_xlabel("Wavelength (nm)")
        ax_noisy.set_ylabel("Reflectance")
        ax_noisy.set_title("Input spectrum (before prediction)")
        ax_noisy.legend()
        ax_noisy.grid(True, alpha=0.3)
        fig_noisy.tight_layout()
        st.pyplot(fig_noisy)
        plt.close(fig_noisy)

    if input_spectrum is not None:
        dae = load_denoiser()

        default_denoiser = False
        if has_true_params:
            default_denoiser = float(noise_level) >= 0.003

        col_opts, col_dt1, col_dt2, col_dt3 = st.columns([1.2, 1, 1, 1])
        with col_opts:
            use_denoiser = st.checkbox(
                "Denoise spectrum first",
                value=default_denoiser,
                key="use_denoiser",
            )
            use_refiner = st.checkbox("Refine with spectral optimizer", value=True, key="use_refiner")
            use_guarded_refiner = st.checkbox(
                "Use guarded refiner",
                value=True,
                key="use_guarded_refiner",
                disabled=not use_refiner,
                help=(
                    "Guarded mode adds NN-prior regularization and trust-region bounds. "
                    "It is safer in ambiguous inverse regimes."
                ),
            )
            lambda_prior = st.slider(
                "Refiner prior weight",
                min_value=0.00,
                max_value=0.40,
                value=0.10,
                step=0.01,
                key="lambda_prior",
                disabled=not (use_refiner and use_guarded_refiner),
            )
        with col_dt1:
            viz_layout = st.radio("Comparison view", ["Side-by-side", "Overlay"], horizontal=True, key="viz_layout", help="Side-by-side renders one panel per result. Overlay stacks them.")
            viz_show_waves = st.checkbox("Animate sweep", value=True, key="viz_show_waves")
        with col_dt2:
            viz_thickness_scale = st.slider("Thickness exaggeration", 0.8, 3.0, 1.6, 0.1, key="viz_thickness_scale")
            viz_speed = st.slider("Animation speed", 0.4, 2.5, 1.0, 0.1, key="viz_speed")
        with col_dt3:
            viz_show_uncertainty = st.checkbox("Show uncertainty shell", value=True, key="viz_show_uncertainty")
            viz_show_rays = st.checkbox("Show incident/reflected rays", value=True, key="viz_show_rays")

        # Apply denoiser if requested
        if use_denoiser:
            spectrum_for_model = denoise_spectrum(input_spectrum, dae)
        else:
            spectrum_for_model = input_spectrum

        if st.button("Predict Parameters", type="primary"):
            # Show denoising preview
            if use_denoiser:
                with st.expander("Denoising preview", expanded=False):
                    fig_dn, ax_dn = plt.subplots(figsize=(8, 3))
                    ax_dn.plot(WAVELENGTHS, input_spectrum, alpha=0.5,
                               label="Raw input", linewidth=0.8)
                    ax_dn.plot(WAVELENGTHS, spectrum_for_model, color="#2ca02c",
                               label="Denoised", linewidth=1.5)
                    ax_dn.set_xlabel("Wavelength (nm)")
                    ax_dn.set_ylabel("Reflectance")
                    ax_dn.set_title("Denoiser: Raw vs Cleaned Spectrum")
                    ax_dn.legend()
                    ax_dn.grid(True, alpha=0.3)
                    fig_dn.tight_layout()
                    st.pyplot(fig_dn)
                    plt.close(fig_dn)
                    st.info(
                        "The denoiser can improve the neural network initial "
                        "guess when noise is meaningful. On very clean synthetic "
                        "inputs it can sometimes shift the distribution, so it is "
                        "now defaulted off for low-noise generated spectra."
                    )
                    st.caption(
                        "Note: the joint denoiser is optimized for parameter "
                        "recovery accuracy, not visual smoothness. It reshapes "
                        "spectral features to improve inversion rather than "
                        "simply smoothing noise. At high noise levels the "
                        "output may appear noisier than the input while still "
                        "improving downstream predictions."
                    )

            means, stds, all_samples = predict_with_uncertainty(
                spectrum_for_model, model, X_mean, X_std,
                n_samples=mc_n_samples
            )
            pred_thick, pred_n, pred_k = means
            std_thick, std_n, std_k = stds
            ci_thick = 1.96 * std_thick
            ci_n = 1.96 * std_n
            ci_k = 1.96 * std_k

            ident = evaluate_identifiability(
                (float(pred_thick), float(pred_n), float(pred_k)),
                (float(ci_thick), float(ci_n), float(ci_k)),
                WAVELENGTHS,
            )
            ambiguous_inverse = bool(
                ident.level == "Weak"
                or float(pred_thick) < 50.0
                or ident.fringe_estimate < 0.5
                or ci_thick > 20.0
                or ci_n > 0.15
                or ci_k > 0.05
            )

            # Optionally run refiner (always against original raw spectrum)
            ref_result = None
            ref_alternatives = []
            if use_refiner:
                if use_guarded_refiner:
                    guarded_bundle = refine_prediction_guarded_multistart(
                        input_spectrum,
                        float(pred_thick),
                        float(pred_n),
                        float(pred_k),
                        ci95=np.array([ci_thick, ci_n, ci_k], dtype=np.float64),
                        wavelengths=WAVELENGTHS,
                        include_coarse_grid=ambiguous_inverse,
                        trust_region=(40.0, 0.25, 0.10),
                        lambda_prior=float(lambda_prior),
                        max_alternatives=3,
                    )
                    ref_result = guarded_bundle.primary.to_dict()
                    ref_alternatives = [alt.to_dict() for alt in guarded_bundle.alternatives]
                else:
                    ref_result = refine_prediction(
                        input_spectrum,
                        float(pred_thick), float(pred_n), float(pred_k),
                        WAVELENGTHS
                    )

            # --- Pre-compute metrics for Digital Twin ---
            if ref_result is not None:
                display_thick = ref_result["thickness"]
                display_n = ref_result["n"]
                display_k = ref_result["k"]
            else:
                display_thick = float(pred_thick)
                display_n = float(pred_n)
                display_k = float(pred_k)

            re_sim_nn = simulate_reflectance(float(pred_thick),
                                             float(pred_n),
                                             float(pred_k), WAVELENGTHS)
            re_sim_ref = (simulate_reflectance(display_thick, display_n,
                                               display_k, WAVELENGTHS)
                          if ref_result is not None else None)

            nn_mae = float(np.mean(np.abs(input_spectrum - re_sim_nn)))
            ref_mae = None
            if ref_result is not None:
                ref_mae = float(np.mean(np.abs(input_spectrum - re_sim_ref)))

            # --- Thin-Film Digital Twin ---
            viz_states = []
            if has_true_params:
                viz_states.append(
                    make_visual_state(
                        "Your Parameters",
                        float(true_thick),
                        float(true_n_val),
                        float(true_k_val),
                        role="truth",
                    )
                )

            viz_states.append(
                make_visual_state(
                    "Neural Network",
                    float(pred_thick),
                    float(pred_n),
                    float(pred_k),
                    ci_thickness=float(ci_thick),
                    ci_n=float(ci_n),
                    ci_k=float(ci_k),
                    spectral_mae=nn_mae,
                    role="nn",
                    emphasis=(ref_result is None),
                )
            )

            if ref_result is not None:
                viz_states.append(
                    make_visual_state(
                        "NN + Refiner",
                        float(ref_result["thickness"]),
                        float(ref_result["n"]),
                        float(ref_result["k"]),
                        spectral_mae=ref_mae,
                        role="refined",
                        emphasis=True,
                    )
                )

            primary_label = (
                "NN + Refiner" if ref_result is not None else "Neural Network"
            )

            st.subheader("Predicted Parameters: Thin-Film Digital Twin")
            twin_col, cards_col = st.columns([1.8, 1.2])
            with twin_col:
                render_thinfilm_visualizer(
                    viz_states,
                    layout=viz_layout,
                    animation_speed=viz_speed,
                    thickness_exaggeration=viz_thickness_scale,
                    show_uncertainty=viz_show_uncertainty,
                    show_wavelength_sweep=viz_show_waves,
                    show_rays=viz_show_rays,
                    height=560,
                    key="inverse_digital_twin",
                )
                st.markdown(
                    """
                    <div style="font-size:0.9rem; line-height:1.55; color:#475569; padding:0.35rem 0.2rem 0 0.2rem;">
                        <strong>Legend.</strong> Height encodes thickness, hue shifts with refractive index,
                        darker films indicate larger extinction coefficient, and the translucent shell around
                        the neural estimate visualizes the 95% thickness confidence interval.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cards_col:
                st.markdown(
                    render_prediction_cards(
                        viz_states,
                        primary_label=primary_label,
                    ),
                    unsafe_allow_html=True,
                )
                

            st.info(
                "The digital twin is intentionally qualitative: it provides a fast visual readout of the "
                "inferred film state while staying consistent with the underlying TMM model and spectral fit."
            )

            # Refiner convergence info
            if ref_result is not None:
                if ref_result["success"]:
                    st.success(
                        f"Converged in {ref_result['n_iterations']} "
                        f"iterations, residual improved "
                        f"{ref_result['improvement']:.1f}%"
                    )
                else:
                    st.warning("Optimizer did not fully converge")

            if ref_result is not None:
                st.info(
                    "**What the refiner does:** The neural network provides "
                    "a fast initial guess. The spectral optimizer then "
                    "fine-tunes the parameters by minimizing the MSE "
                    "between the input spectrum and the spectrum "
                    "re-simulated from the predicted parameters using "
                    "L-BFGS-B with physical box constraints. This "
                    "deterministic post-processing step typically reduces "
                    "the spectral residual by 90%+ and corrects small "
                    "systematic biases in the NN prediction. A better "
                    "spectral fit is not always a unique physical answer "
                    "in low-identifiability regimes."
                )

            st.markdown("### Reliability Assessment")
            level_color = {
                "Strong": "🟢",
                "Moderate": "🟠",
                "Weak": "🔴",
            }.get(ident.level, "🟠")
            st.write(
                f"{level_color} **Identifiability:** {ident.level} "
                f"(score={ident.score:.2f}, cond={ident.condition_number:.0f})"
            )
            st.caption(
                f"Fringe estimate: {ident.fringe_estimate:.2f} · "
                f"smallest singular value: {ident.smallest_singular_value:.4e}"
            )
            for reason in ident.reasons:
                st.write(f"- {reason}")

            if ambiguous_inverse:
                st.warning(
                    "This spectrum is in an ambiguous inverse regime. "
                    "A low spectral residual does not guarantee unique physical parameters."
                )

            if ref_alternatives:
                st.markdown("**Alternative plausible fits**")
                for idx_alt, alt in enumerate(ref_alternatives, 1):
                    alt_sim = simulate_reflectance(
                        float(alt["thickness"]),
                        float(alt["n"]),
                        float(alt["k"]),
                        WAVELENGTHS,
                    )
                    alt_mae = float(np.mean(np.abs(input_spectrum - alt_sim)))
                    st.write(
                        f"{idx_alt}. d={alt['thickness']:.2f} nm, "
                        f"n={alt['n']:.4f}, k={alt['k']:.4f}, "
                        f"spectral MAE={alt_mae:.6f}"
                    )

            # --- Spectral plots ---
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_ov, ax_ov = plt.subplots(figsize=(7, 4))
                ax_ov.plot(WAVELENGTHS, input_spectrum, "k-",
                           label="Original input", linewidth=1.5)
                if use_denoiser:
                    ax_ov.plot(WAVELENGTHS, spectrum_for_model,
                               color="#9467bd", alpha=0.6,
                               label="Denoised", linewidth=1)
                ax_ov.plot(WAVELENGTHS, re_sim_nn, "--", color="#1f77b4",
                           label="NN re-simulated", linewidth=1.5)
                if re_sim_ref is not None:
                    ax_ov.plot(WAVELENGTHS, re_sim_ref, "--",
                               color="#2ca02c",
                               label="Refined re-simulated", linewidth=1.5)
                ax_ov.set_xlabel("Wavelength (nm)")
                ax_ov.set_ylabel("Reflectance")
                ax_ov.set_title("Input vs Re-simulated Spectrum")
                ax_ov.legend(fontsize=8)
                ax_ov.grid(True, alpha=0.3)
                fig_ov.tight_layout()
                st.pyplot(fig_ov)
                plt.close(fig_ov)

            with col_p2:
                residual_nn = input_spectrum - re_sim_nn
                fig_res, ax_res = plt.subplots(figsize=(7, 4))
                ax_res.plot(WAVELENGTHS, residual_nn, color="#1f77b4",
                            linewidth=1, label="NN residual")
                if re_sim_ref is not None:
                    residual_ref = input_spectrum - re_sim_ref
                    ax_res.plot(WAVELENGTHS, residual_ref, color="#2ca02c",
                                linewidth=1, label="Refined residual")
                ax_res.axhline(0, color="gray", linestyle="--",
                               linewidth=0.5)
                ax_res.set_xlabel("Wavelength (nm)")
                ax_res.set_ylabel("Residual (input - re-sim)")
                ax_res.set_title("Spectral Residual")
                ax_res.legend()
                ax_res.grid(True, alpha=0.3)
                fig_res.tight_layout()
                st.pyplot(fig_res)
                plt.close(fig_res)

            st.metric("NN Spectral MAE", f"{nn_mae:.6f}")
            if ref_result is not None:
                st.metric("Refined Spectral MAE", f"{ref_mae:.6f}")

            # Uncertainty analysis expander
            with st.expander("Uncertainty Analysis"):
                param_names = ["Thickness (nm)", "n", "k"]
                fig_hist, axes_hist = plt.subplots(1, 3, figsize=(14, 3.5))
                for i, (ax, name) in enumerate(zip(axes_hist, param_names)):
                    ax.hist(all_samples[:, i], bins=25, edgecolor="black",
                            alpha=0.7, color="#1f77b4")
                    ax.axvline(means[i], color="red", linestyle="--",
                               linewidth=1.5, label=f"Mean = {means[i]:.3f}")
                    if ref_result is not None:
                        ref_vals = [ref_result["thickness"],
                                    ref_result["n"], ref_result["k"]]
                        ax.axvline(ref_vals[i], color="#2ca02c",
                                   linestyle="-.",
                                   linewidth=1.5,
                                   label=f"Refined = {ref_vals[i]:.3f}")
                    ax.set_xlabel(name)
                    ax.set_ylabel("Count")
                    ax.set_title(f"{name} ({mc_n_samples} MC samples)")
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                fig_hist.tight_layout()
                st.pyplot(fig_hist)
                plt.close(fig_hist)

                labels_for_interp = ["thickness", "n", "k"]
                thresholds_low = [5.0, 0.05, 0.01]
                confident = []
                uncertain = []
                for i, (name, thr) in enumerate(
                    zip(labels_for_interp, thresholds_low)
                ):
                    if stds[i] < thr:
                        confident.append(name)
                    else:
                        uncertain.append(name)

                if confident and uncertain:
                    st.info(
                        f"The model is **confident** about "
                        f"**{', '.join(confident)}** but "
                        f"**uncertain** about "
                        f"**{', '.join(uncertain)}**."
                    )
                elif confident:
                    st.success(
                        "The model is **confident** about all three "
                        "parameters."
                    )
                else:
                    st.warning(
                        "The model shows **elevated uncertainty** across "
                        "all parameters. The input spectrum may be outside "
                        "the training distribution or contain high noise."
                    )

# ──────────────────────────────────────────────
# TAB 3: Model Performance
# ──────────────────────────────────────────────

with tab3:
    st.header("Neural Thin-Film Metrology")

    # ── SECTION 1: The Problem ──────────────────

    st.subheader("1. The Problem")
    st.markdown("""
**Thin-film metrology** is the science of measuring the thickness and
optical properties of nanometer-scale coatings deposited on semiconductor
wafers. Every chip fabrication step (oxide growth, anti-reflective
coatings, passivation layers) requires verifying that the deposited film
meets spec. Getting this wrong means defective chips and scrapped wafers.

The measurement technique is **spectral reflectometry**: shine broadband
light (400-800 nm) onto the wafer surface and record how much reflects
back at each wavelength. The resulting **reflectance spectrum** encodes
the film's thickness, refractive index (n), and extinction coefficient
(k) through thin-film interference patterns. Thicker films produce more
fringes; higher n increases fringe contrast; nonzero k damps the
oscillations through absorption.

The **inverse problem**, extracting (thickness, n, k) from a measured
spectrum, is fundamentally hard because different parameter combinations
can produce nearly identical spectra. A slightly thicker film with a
slightly lower n shifts fringes the same way, creating a
**thickness-n degeneracy** that worsens for thin films (<50 nm) where
fewer than one fringe is visible. Traditional fitting methods (Levenberg-
Marquardt, simplex) require good initial guesses and often get trapped
in local minima.

**Why speed matters:** KLA and other metrology companies measure every
wafer at multiple points across the surface. A single 300mm wafer may
require hundreds of measurements, each needing sub-second turnaround.
Neural networks provide the speed (microseconds per inference) while
physics-based refinement provides the accuracy.
""")

    st.divider()

    # ── SECTION 2: The Forward Simulator ────────

    st.subheader("2. The Forward Simulator")
    st.markdown("""
The **Transfer Matrix Method (TMM)** computes reflectance from first
principles. For a single film on a substrate at normal incidence, it
constructs a 2x2 characteristic matrix from the film's complex
refractive index and phase thickness, then extracts the reflection
coefficient by matching boundary conditions at both interfaces.
The math is exact: no approximations, no fitting parameters.
""")

    # Plot 1: varying thickness
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_thick, ax_thick = plt.subplots(figsize=(7, 4))
        for d in [50, 100, 150, 200]:
            R = simulate_reflectance(d, 1.46, 0.0, WAVELENGTHS)
            ax_thick.plot(WAVELENGTHS, R, label=f"{d} nm", linewidth=1.5)
        ax_thick.set_xlabel("Wavelength (nm)")
        ax_thick.set_ylabel("Reflectance")
        ax_thick.set_title("Effect of Thickness (n=1.46, k=0)")
        ax_thick.legend()
        ax_thick.grid(True, alpha=0.3)
        ax_thick.annotate("More fringes = thicker film",
                          xy=(0.5, 0.02), xycoords="axes fraction",
                          fontsize=9, fontstyle="italic", ha="center",
                          color="#555555")
        fig_thick.tight_layout()
        st.pyplot(fig_thick)
        plt.close(fig_thick)

    with col_t2:
        fig_abs, ax_abs = plt.subplots(figsize=(7, 4))
        for kv in [0.0, 0.1, 0.2, 0.3]:
            R = simulate_reflectance(100, 1.46, kv, WAVELENGTHS)
            ax_abs.plot(WAVELENGTHS, R, label=f"k={kv:.1f}", linewidth=1.5)
        ax_abs.set_xlabel("Wavelength (nm)")
        ax_abs.set_ylabel("Reflectance")
        ax_abs.set_title("Effect of Absorption (d=100nm, n=1.46)")
        ax_abs.legend()
        ax_abs.grid(True, alpha=0.3)
        ax_abs.annotate("Damped fringes = absorbing film",
                        xy=(0.5, 0.02), xycoords="axes fraction",
                        fontsize=9, fontstyle="italic", ha="center",
                        color="#555555")
        fig_abs.tight_layout()
        st.pyplot(fig_abs)
        plt.close(fig_abs)

    st.markdown("""
A key contribution of this project is implementing the TMM **entirely in
PyTorch** so that gradients flow through the physics. This means the
training loss can include a term that re-simulates the spectrum from
predicted parameters and penalizes deviations, forcing the network to
learn physically self-consistent mappings rather than just statistical
correlations.
""")

    st.divider()

    # ── SECTION 3: Model Evolution ──────────────

    st.subheader("3. Model Evolution")

    v1_col, v3_col, v4_col = st.columns(3)

    with v1_col:
        st.markdown("##### V1: Baseline")
        st.markdown("""
**Changed:** MLP [512, 256, 128, 64], 100k uniform random samples

**Why:** Establish a baseline. Prove that a neural network can learn
the inverse mapping at all.

**Result:**
- Overall R\u00b2 = 0.952
- Thickness MAE = 3.65 nm
- n R\u00b2 = 0.871

**Finding:** The concept works, but n is the hard parameter. V2
experiments with deeper architectures, BatchNorm, and residual
connections all performed *worse*. Architecture was not the
bottleneck.
""")

    with v3_col:
        st.markdown("##### V3: Better Data")
        st.markdown("""
**Changed:** 500k stratified dataset (thickness-binned +
n-binned + uniform)

**Why:** V1's n R\u00b2 = 0.871 ceiling and V2's failure to improve
via architecture both pointed to data as the bottleneck. Uniform
sampling underrepresents thin films and extreme n values.

**Result:**
- Overall R\u00b2 = 0.967
- Thickness MAE = 2.79 nm
- n R\u00b2 = 0.915

**Finding:** Data quality matters more than architecture. 5x more
data with stratified sampling lifted every metric.
""")

    with v4_col:
        st.markdown("##### V4: Physics Loss")
        st.markdown("""
**Changed:** TMM implemented in PyTorch. Training loss includes
MSE between re-simulated and input spectrum, with gradients
flowing through the simulator.

**Why:** Physics-informed approaches are an active research
direction in semiconductor metrology. The differentiable TMM
acts as a regularizer that prevents the network from learning
unphysical parameter combinations.

**Result:**
- Overall R\u00b2 = 0.975
- Thickness MAE = 2.62 nm
- n R\u00b2 = 0.930

**Finding:** Physics loss most impactful on k (R\u00b2 0.990
\u2192 0.996) and n (R\u00b2 0.915 \u2192 0.930), the parameters
that directly control spectral shape.
""")

    st.divider()

    # ── SECTION 4: Results ──────────────────────

    st.subheader("4. Results")

    import pandas as pd

    results_df = pd.DataFrame({
        "Version": ["V1 (100k, baseline)", "V3 (500k, stratified)",
                     "V4 (500k, physics loss)"],
        "Thickness MAE (nm)": [3.65, 2.79, 2.62],
        "n R\u00b2": [0.871, 0.915, 0.930],
        "k R\u00b2": [0.992, 0.990, 0.996],
        "Overall R\u00b2": [0.952, 0.967, 0.975],
    })

    st.dataframe(
        results_df.style
        .background_gradient(subset=["n R\u00b2", "k R\u00b2",
                                     "Overall R\u00b2"],
                             cmap="Greens", vmin=0.85, vmax=1.0)
        .background_gradient(subset=["Thickness MAE (nm)"],
                             cmap="Greens_r", vmin=2.0, vmax=4.0),
        width="stretch",
        hide_index=True,
    )

    # Training artifacts: 2x2 grid
    st.markdown("##### Training Curves and Parity Plots")
    row1_a, row1_b = st.columns(2)
    row2_a, row2_b = st.columns(2)

    with row1_a:
        try:
            st.image(str(artifact_path("figures", "loss_curve_v3.png")))
            st.caption("V3 loss curve: early stopping at ~80 epochs. "
                       "Validation loss closely tracks training loss, "
                       "confirming no overfitting with 500k samples.")
        except FileNotFoundError:
            st.warning(f"{artifact_path('figures', 'loss_curve_v3.png')} not found")
    with row1_b:
        try:
            st.image(str(artifact_path("figures", "loss_curve_v4.png")))
            st.caption("V4 loss curve: the physics loss term adds a "
                       "constant offset but does not destabilize training. "
                       "Convergence is smooth.")
        except FileNotFoundError:
            st.warning(f"{artifact_path('figures', 'loss_curve_v4.png')} not found")
    with row2_a:
        try:
            st.image(str(artifact_path("figures", "parity_plot_v3.png")))
            st.caption("V3 parity: thickness and k are tight around y=x. "
                       "The n panel shows visible spread, the hardest "
                       "parameter to predict.")
        except FileNotFoundError:
            st.warning(f"{artifact_path('figures', 'parity_plot_v3.png')} not found")
    with row2_b:
        try:
            st.image(str(artifact_path("figures", "parity_plot_v4.png")))
            st.caption("V4 parity: physics loss tightens the n scatter "
                       "and nearly eliminates k outliers. Thickness "
                       "improves marginally.")
        except FileNotFoundError:
            st.warning(f"{artifact_path('figures', 'parity_plot_v4.png')} not found")

    st.markdown("""
**Reading the parity plots:** Each dot is one test sample. The x-axis is
the true value, y-axis is the predicted value. Perfect predictions fall
exactly on the red y=x line. The **thickness** panel is tightest because
thickness has the strongest, most unambiguous effect on the spectrum
(fringe count and spacing). The **n** panel is widest because refractive
index changes produce subtler spectral shifts that overlap with thickness
effects. This is the fundamental degeneracy of the inverse problem. The
**k** panel is tight because absorption produces a distinctive damping
pattern that is hard to confuse with other effects.
""")

    st.divider()

    # ── SECTION 5: The Full Pipeline ────────────

    st.subheader("5. The Full Pipeline")

    # Visual pipeline using styled containers
    st.markdown(
        '<p style="text-align:center; font-size:0.95rem; '
        'color:#555; margin-bottom:0.2rem;">'
        'Measured Reflectance Spectrum</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; font-size:1.4rem; '
        'line-height:1; margin:0;">&#x25BC;</p>',
        unsafe_allow_html=True,
    )

    pipe_cols = st.columns([1, 3])
    with pipe_cols[0]:
        st.markdown(
            '<div style="background:#e8f4fd; border-left:4px solid #1f77b4; '
            'padding:12px 14px; border-radius:0 6px 6px 0; '
            'margin-bottom:4px;">'
            '<strong style="font-size:1.05rem;">Joint Denoiser</strong></div>',
            unsafe_allow_html=True,
        )
    with pipe_cols[1]:
        st.markdown(
            '<div style="padding:12px 0; color:#444; font-size:0.9rem;">'
            'Physics-aware autoencoder trained with a frozen SpectraNet in '
            'the loop. Preserves spectral features that matter for parameter '
            'recovery rather than simply smoothing noise.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="text-align:center; font-size:1.4rem; '
        'line-height:1; margin:0;">&#x25BC;</p>',
        unsafe_allow_html=True,
    )

    pipe_cols2 = st.columns([1, 3])
    with pipe_cols2[0]:
        st.markdown(
            '<div style="background:#e8f8e8; border-left:4px solid #2ca02c; '
            'padding:12px 14px; border-radius:0 6px 6px 0; '
            'margin-bottom:4px;">'
            '<strong style="font-size:1.05rem;">SpectraNet V4</strong></div>',
            unsafe_allow_html=True,
        )
    with pipe_cols2[1]:
        st.markdown(
            '<div style="padding:12px 0; color:#444; font-size:0.9rem;">'
            '275k-parameter MLP trained with a differentiable TMM physics '
            'loss. Produces initial parameter estimates in microseconds. '
            'MC Dropout (100 forward passes) yields 95% confidence '
            'intervals.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="text-align:center; font-size:1.4rem; '
        'line-height:1; margin:0;">&#x25BC;</p>',
        unsafe_allow_html=True,
    )

    pipe_cols3 = st.columns([1, 3])
    with pipe_cols3[0]:
        st.markdown(
            '<div style="background:#fff3e0; border-left:4px solid #ff7f0e; '
            'padding:12px 14px; border-radius:0 6px 6px 0; '
            'margin-bottom:4px;">'
            '<strong style="font-size:1.05rem;">L-BFGS-B Refiner</strong>'
            '</div>',
            unsafe_allow_html=True,
        )
    with pipe_cols3[1]:
        st.markdown(
            '<div style="padding:12px 0; color:#444; font-size:0.9rem;">'
            'Minimizes spectral residual against the <strong>original</strong>'
            ' measured spectrum (not denoised) using box-constrained '
            'quasi-Newton optimization. Typically converges in 20-30 '
            'iterations, improving residual by 80-90%.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="text-align:center; font-size:1.4rem; '
        'line-height:1; margin:0;">&#x25BC;</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; font-size:0.95rem; '
        'color:#555; margin-top:0.2rem;">'
        'Predicted Parameters &nbsp;+&nbsp; 95% Confidence Intervals</p>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacing

    st.markdown("""
**Why this hybrid approach works:** The neural network provides a fast,
globally-informed initial guess that avoids local minima, the primary
failure mode of optimization-only approaches. The L-BFGS-B refiner then
exploits the exact physics (TMM) to close the remaining residual, achieving
accuracy that the network alone cannot reach. Together they combine the
speed of inference (~1ms) with the precision of optimization (~50ms).

**Why the refiner uses the original spectrum:** The denoiser reshapes
spectral features to help the neural network, but in doing so it may
subtly distort the spectrum away from the true physics. The refiner's job
is to find parameters whose TMM simulation exactly matches the actual
measurement, so it must optimize against the raw, unaltered spectrum.
The denoiser's role is limited to providing a better starting point for
the network. After that, the original measurement is the ground truth.
""")

    st.divider()

    # ── SECTION 6: Limitations ──────────────────

    st.subheader("6. Limitations and Honest Assessment")

    st.markdown("""
**The n-thickness degeneracy at low thickness (<50 nm):**
When a film is thin enough that less than one fringe appears in the
visible range, the spectrum becomes a smooth curve that can be fit by
multiple (thickness, n) combinations. The model correctly flags this
with wide MC Dropout uncertainty intervals, but the point prediction
may be unreliable. This is a fundamental physical limitation, not a
model deficiency.

**k is never predicted as exactly zero:**
The training distribution samples k uniformly from [0, 0.5]. The network
learns a continuous mapping and cannot output a hard zero, so predictions
for truly transparent films (k=0) typically floor at ~0.003-0.005. A
classification head ("is k=0?") could fix this but was not implemented.

**Denoiser limitations at high noise (std > 0.015):**
At noise levels above ~0.015, the signal-to-noise ratio degrades to the
point where spectral features (fringe positions, amplitudes) are
irreversibly corrupted. The denoiser cannot recover information that was
destroyed. It can only interpolate from what remains. The model's
uncertainty estimates correctly widen in this regime.

**Single-layer assumption:**
This project models a single thin film on a silicon substrate. Real
semiconductor stacks have 5-50 layers. The TMM generalizes naturally to
multi-layer stacks, but the inverse problem becomes exponentially harder
as the number of unknown parameters grows.

**Normal incidence only:**
Real spectroscopic ellipsometers measure at multiple angles of incidence
and both s- and p-polarization components. Multi-angle measurements break
the n-thickness degeneracy that limits this system. Extending to
oblique incidence requires adding angle-dependent Fresnel coefficients
to the TMM, straightforward in principle but a meaningful engineering
effort.

**What would actually fix these:**
Spectroscopic ellipsometry (measuring polarization change, not just
intensity), multi-angle reflectometry, and multi-layer TMM extensions.
These are the standard approaches in production metrology tools. This
project demonstrates the neural inversion concept on the simplest
physically meaningful case.
""")
