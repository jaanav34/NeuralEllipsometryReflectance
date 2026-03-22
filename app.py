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


def predict_params(spectrum, model, X_mean, X_std):
    """Normalize spectrum, run model, denormalize output."""
    x = (spectrum.astype(np.float32) - X_mean) / X_std
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x[np.newaxis, :])).numpy()[0]
    return pred_norm * PARAM_RANGE + PARAM_MIN


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(page_title="Neural Ellipsometry", layout="wide")
st.title("Neural Ellipsometry Reflectance")

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
        if st.button("Predict Parameters", type="primary"):
            pred = predict_params(input_spectrum, model, X_mean, X_std)
            pred_thick, pred_n, pred_k = pred

            # Show predicted parameters
            st.subheader("Predicted Parameters")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Thickness", f"{pred_thick:.1f} nm")
            mc2.metric("Refractive index n", f"{pred_n:.4f}")
            mc3.metric("Extinction coefficient k", f"{pred_k:.4f}")

            # Re-simulate from predictions
            re_sim = simulate_reflectance(float(pred_thick), float(pred_n),
                                          float(pred_k), WAVELENGTHS)
            residual = input_spectrum - re_sim
            spectral_mae = np.mean(np.abs(residual))

            # Overlay plot
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_ov, ax_ov = plt.subplots(figsize=(7, 4))
                ax_ov.plot(WAVELENGTHS, input_spectrum, label="Input",
                           linewidth=1.5)
                ax_ov.plot(WAVELENGTHS, re_sim, "--",
                           label="Re-simulated from prediction",
                           linewidth=1.5)
                ax_ov.set_xlabel("Wavelength (nm)")
                ax_ov.set_ylabel("Reflectance")
                ax_ov.set_title("Input vs Re-simulated Spectrum")
                ax_ov.legend()
                ax_ov.grid(True, alpha=0.3)
                fig_ov.tight_layout()
                st.pyplot(fig_ov)
                plt.close(fig_ov)

            with col_p2:
                fig_res, ax_res = plt.subplots(figsize=(7, 4))
                ax_res.plot(WAVELENGTHS, residual, color="red", linewidth=1)
                ax_res.axhline(0, color="gray", linestyle="--", linewidth=0.5)
                ax_res.set_xlabel("Wavelength (nm)")
                ax_res.set_ylabel("Residual (input - re-sim)")
                ax_res.set_title("Spectral Residual")
                ax_res.grid(True, alpha=0.3)
                fig_res.tight_layout()
                st.pyplot(fig_res)
                plt.close(fig_res)

            st.metric("Spectral MAE", f"{spectral_mae:.6f}",
                      help="Mean absolute error between the input spectrum "
                           "and the spectrum re-simulated from predicted "
                           "parameters. Lower is better.")

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
