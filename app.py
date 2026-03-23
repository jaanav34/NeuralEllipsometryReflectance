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

from tmm_simulator import simulate_reflectance
from train import SpectraNet
from refiner import refine_prediction
from denoiser import DenoisingAutoencoder

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
    model.load_state_dict(torch.load("spectranet_v4.pt",
                                     map_location="cpu",
                                     weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_norm_stats():
    data = np.load("spectra_norm_v4.npz")
    return data["mean"], data["std"]


@st.cache_resource
def load_denoiser():
    dae = DenoisingAutoencoder()
    dae.load_state_dict(torch.load("denoiser_joint.pt",
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
    st.header("Transfer Matrix Method — Forward Simulator")

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
    st.header("SpectraNet V4 — Inverse Predictor")

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
        use_denoiser = st.checkbox("Denoise spectrum first", value=True,
                                   key="use_denoiser")
        use_refiner = st.checkbox("Refine with spectral optimizer",
                                  value=True, key="use_refiner")

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
                        "The denoiser improves the neural network's initial "
                        "guess. The spectral refiner always optimizes against "
                        "the original measured spectrum for maximum physical "
                        "accuracy."
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

            # Optionally run refiner (always against original raw spectrum)
            ref_result = None
            if use_refiner:
                ref_result = refine_prediction(
                    input_spectrum,
                    float(pred_thick), float(pred_n), float(pred_k),
                    WAVELENGTHS
                )

            # --- Helper: colored CI badge ---
            def _ci_badge(ci_val, std_val, fmt, thresholds):
                """Return colored markdown for a CI badge."""
                lo, hi = thresholds
                text = f"\u00b1{ci_val:{fmt}}"
                if std_val < lo:
                    return f":green-background[{text}]"
                elif std_val < hi:
                    return f":orange-background[{text}]"
                else:
                    return f":red-background[{text}]"

            # --- Show predicted parameters ---
            st.subheader("Predicted Parameters")

            ci_thick_badge = _ci_badge(ci_thick, std_thick, ".1f",
                                       (5, 15))
            ci_n_badge = _ci_badge(ci_n, std_n, ".4f", (0.05, 0.15))
            ci_k_badge = _ci_badge(ci_k, std_k, ".4f", (0.01, 0.03))

            # Row layout: one row per parameter, columns align across rows
            # Determine how many columns per row
            n_cols = 1  # always have NN
            if has_true_params:
                n_cols += 1
            if ref_result is not None:
                n_cols += 1

            # Header row
            hdr_cols = st.columns(n_cols)
            idx = 0
            if has_true_params:
                hdr_cols[idx].markdown("**Your Parameters**")
                idx += 1
            hdr_cols[idx].markdown("**Neural Network**")
            idx += 1
            if ref_result is not None:
                hdr_cols[idx].markdown("**NN + Refiner**")

            # Parameter rows: [true | NN value + CI | refiner]
            param_rows = [
                ("Thickness",
                 f"{true_thick} nm" if has_true_params else None,
                 f"{pred_thick:.1f} nm", ci_thick_badge,
                 f"{ref_result['thickness']:.1f} nm" if ref_result else None),
                ("Refractive index n",
                 f"{true_n_val:.2f}" if has_true_params else None,
                 f"{pred_n:.4f}", ci_n_badge,
                 f"{ref_result['n']:.4f}" if ref_result else None),
                ("Extinction coeff. k",
                 f"{true_k_val:.3f}" if has_true_params else None,
                 f"{pred_k:.4f}", ci_k_badge,
                 f"{ref_result['k']:.4f}" if ref_result else None),
            ]

            for label, true_val, nn_val, ci_badge, ref_val in param_rows:
                row = st.columns(n_cols)
                idx = 0
                if has_true_params:
                    row[idx].metric(label, true_val)
                    idx += 1
                row[idx].metric(label, nn_val)
                row[idx].markdown(f"95% CI: {ci_badge}")
                idx += 1
                if ref_result is not None:
                    row[idx].metric(label, ref_val)

            # Refiner convergence info
            if ref_result is not None:
                if ref_result["success"]:
                    st.success(
                        f"Converged in {ref_result['n_iterations']} "
                        f"iterations — residual improved "
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
                    "systematic biases in the NN prediction."
                )

            # --- Spectral plots ---
            # Use refined params if available, otherwise NN params
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

            # Residuals and overlay are always against the original raw spectrum
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

            nn_mae = float(np.mean(np.abs(input_spectrum - re_sim_nn)))
            st.metric("NN Spectral MAE", f"{nn_mae:.6f}")
            if ref_result is not None:
                ref_mae = float(np.mean(np.abs(input_spectrum - re_sim_ref)))
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
    st.header("Model Performance Across Versions")

    # V4 plots
    st.subheader("V4 — Physics-Informed Loss (Best)")
    col_v4a, col_v4b = st.columns(2)
    with col_v4a:
        try:
            st.image("loss_curve_v4.png", caption="V4 Loss Curve")
        except FileNotFoundError:
            st.warning("loss_curve_v4.png not found")
    with col_v4b:
        try:
            st.image("parity_plot_v4.png", caption="V4 Parity Plot")
        except FileNotFoundError:
            st.warning("parity_plot_v4.png not found")

    # V3 plots
    st.subheader("V3 — Larger Dataset, No Physics Loss")
    col_v3a, col_v3b = st.columns(2)
    with col_v3a:
        try:
            st.image("loss_curve_v3.png", caption="V3 Loss Curve")
        except FileNotFoundError:
            st.warning("loss_curve_v3.png not found")
    with col_v3b:
        try:
            st.image("parity_plot_v3.png", caption="V3 Parity Plot")
        except FileNotFoundError:
            st.warning("parity_plot_v3.png not found")

    # Comparison table
    st.subheader("Version Comparison")
    st.table({
        "Version": ["V1 (100k, baseline)", "V3 (500k, stratified)",
                     "V4 (500k, physics loss)"],
        "Thickness MAE (nm)": [3.65, 2.79, 2.62],
        "n R\u00b2": [0.871, 0.915, 0.930],
        "k R\u00b2": [0.992, 0.990, 0.996],
        "Overall R\u00b2": [0.952, 0.967, 0.975],
    })

    # Explanations
    st.subheader("Understanding the Results")

    st.markdown("""
**What the parity plots show:**
Each parity plot is a scatter of predicted vs true values for one parameter
across the test set. Points falling on the red y=x line indicate perfect
predictions. Spread around the line reflects prediction error. The MAE
annotation quantifies the average absolute deviation from truth.

**Why n is harder to predict than thickness or k:**
Refractive index (n) affects reflectance primarily through fringe spacing
and amplitude, both of which are also influenced by thickness. The spectrum
is more sensitive to thickness changes (which shift fringe positions
strongly) and to k changes (which damp overall reflectance), making those
two parameters easier to disentangle. In contrast, changes in n produce
subtler spectral shifts that can be partially mimicked by thickness
adjustments, leading to a higher prediction error.

**What the physics-informed loss contributed (V3 to V4):**
The differentiable TMM loss in V4 forces the network's predictions to be
physically self-consistent: the predicted (thickness, n, k) must
re-produce the input spectrum when passed through the TMM simulator.
This acts as a strong inductive bias that prevents the network from
finding parameter combinations that minimize the training MSE but
violate thin-film physics. The result is improved accuracy across all
three parameters, with the biggest gains on n (R\u00b2 0.915 to 0.930)
where the extra constraint helps resolve the thickness-n ambiguity.
""")
