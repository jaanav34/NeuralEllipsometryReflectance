"""
Reusable inference utilities for Neural Thin-Film Metrology benchmarks.

This module intentionally mirrors the Streamlit inverse-tab pipeline:

    raw/generated spectrum
      -> optional joint denoiser
      -> SpectraNet V4 MC Dropout mean prediction
      -> optional L-BFGS-B refiner against the original raw spectrum

The benchmark scripts use this instead of re-implementing app logic in many
places. Keep app.py and this module behavior aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch

from src.denoiser import DenoisingAutoencoder
from src.paths import artifact_path
from src.spectranet import SpectraNet
from src.tmm_simulator import simulate_reflectance_batch

WAVELENGTHS = np.linspace(400, 800, 200).astype(np.float32)
PARAM_BOUNDS = np.array([[10.0, 300.0], [1.3, 2.5], [0.0, 0.5]], dtype=np.float32)
PARAM_MIN = PARAM_BOUNDS[:, 0]
PARAM_RANGE = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
PARAM_NAMES = ("thickness_nm", "n", "k")


@dataclass(frozen=True)
class ModelBundle:
    model: SpectraNet
    x_mean: np.ndarray
    x_std: np.ndarray
    denoiser: DenoisingAutoencoder | None
    device: torch.device


@dataclass(frozen=True)
class PipelineConfig:
    use_denoiser: bool = True
    mc_samples: int = 100
    batch_size: int = 8192
    device: str = "auto"
    deterministic_dropout_seed: int = 42


@dataclass(frozen=True)
class BatchPrediction:
    nn_mean: np.ndarray
    nn_std: np.ndarray
    nn_ci95: np.ndarray
    spectrum_for_model: np.ndarray

    def as_npz_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {
            f"{prefix}nn_mean": self.nn_mean,
            f"{prefix}nn_std": self.nn_std,
            f"{prefix}nn_ci95": self.nn_ci95,
            f"{prefix}spectrum_for_model": self.spectrum_for_model,
        }


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_bundle(
    model_name: str = "spectranet_v4.pt",
    norm_name: str = "spectra_norm_v4.npz",
    denoiser_name: str = "denoiser_joint.pt",
    device: str = "auto",
    load_denoiser: bool = True,
) -> ModelBundle:
    dev = resolve_device(device)

    model = SpectraNet().to(dev)
    model.load_state_dict(
        torch.load(artifact_path("models", model_name), map_location=dev, weights_only=True)
    )
    model.eval()

    norm = np.load(artifact_path("data", norm_name))
    x_mean = norm["mean"].astype(np.float32)
    x_std = norm["std"].astype(np.float32)
    x_std[x_std < 1e-8] = 1.0

    dae = None
    if load_denoiser:
        dae = DenoisingAutoencoder().to(dev)
        dae.load_state_dict(
            torch.load(artifact_path("models", denoiser_name), map_location=dev, weights_only=True)
        )
        dae.eval()

    return ModelBundle(model=model, x_mean=x_mean, x_std=x_std, denoiser=dae, device=dev)


def iter_slices(n_items: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n_items, batch_size):
        yield slice(start, min(start + batch_size, n_items))


def simulate_params(params: np.ndarray, wavelengths: np.ndarray = WAVELENGTHS) -> np.ndarray:
    params = np.asarray(params, dtype=np.float32)
    return simulate_reflectance_batch(
        params[:, 0].astype(np.float64),
        params[:, 1].astype(np.float64),
        params[:, 2].astype(np.float64),
        wavelengths.astype(np.float64),
    ).astype(np.float32)


def add_noise(
    spectra: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_std <= 0:
        return spectra.astype(np.float32, copy=False)
    noisy = spectra + rng.normal(0.0, noise_std, size=spectra.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def denoise_batch(
    spectra: np.ndarray,
    bundle: ModelBundle,
    batch_size: int = 8192,
) -> np.ndarray:
    if bundle.denoiser is None:
        raise ValueError("Denoiser was not loaded. Set load_denoiser=True.")

    out = np.empty_like(spectra, dtype=np.float32)
    with torch.no_grad():
        for sl in iter_slices(len(spectra), batch_size):
            xb = torch.from_numpy(spectra[sl].astype(np.float32, copy=False)).to(bundle.device)
            out[sl] = bundle.denoiser(xb).detach().cpu().numpy().astype(np.float32)
    return out


def predict_eval_batch(
    spectra: np.ndarray,
    bundle: ModelBundle,
    batch_size: int = 8192,
) -> np.ndarray:
    """Fast deterministic inference using model.eval(). Not the exact app path."""
    pred = np.empty((len(spectra), 3), dtype=np.float32)
    bundle.model.eval()
    with torch.no_grad():
        for sl in iter_slices(len(spectra), batch_size):
            x = (spectra[sl].astype(np.float32, copy=False) - bundle.x_mean) / bundle.x_std
            xb = torch.from_numpy(x.astype(np.float32, copy=False)).to(bundle.device)
            y = bundle.model(xb).detach().cpu().numpy().astype(np.float32)
            pred[sl] = y * PARAM_RANGE + PARAM_MIN
    return pred


def predict_mc_dropout_batch(
    spectra: np.ndarray,
    bundle: ModelBundle,
    n_samples: int = 100,
    batch_size: int = 4096,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Exact app-style MC Dropout inference, but batched.

    Returns physical-unit mean, std, and optionally no full sample tensor. Keeping
    every MC sample for millions of spectra is usually too large, so this stores
    only streaming first and second moments.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    n = len(spectra)
    sum_pred = np.zeros((n, 3), dtype=np.float64)
    sumsq_pred = np.zeros((n, 3), dtype=np.float64)

    # Match app behavior: deterministic dropout masks from a fixed manual seed.
    torch.manual_seed(seed)
    bundle.model.train()
    with torch.no_grad():
        for sample_idx in range(n_samples):
            for sl in iter_slices(n, batch_size):
                x = (spectra[sl].astype(np.float32, copy=False) - bundle.x_mean) / bundle.x_std
                xb = torch.from_numpy(x.astype(np.float32, copy=False)).to(bundle.device)
                y_norm = bundle.model(xb).detach().cpu().numpy().astype(np.float32)
                y_phys = y_norm * PARAM_RANGE + PARAM_MIN
                sum_pred[sl] += y_phys
                sumsq_pred[sl] += y_phys.astype(np.float64) ** 2
    bundle.model.eval()

    mean = (sum_pred / n_samples).astype(np.float32)
    if n_samples == 1:
        std = np.zeros_like(mean, dtype=np.float32)
    else:
        var = (sumsq_pred / n_samples) - (sum_pred / n_samples) ** 2
        std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    return mean, std, None


def predict_app_style_batch(
    spectra: np.ndarray,
    bundle: ModelBundle,
    config: PipelineConfig,
) -> BatchPrediction:
    """Run the same denoiser + MC Dropout mean path used in app.py."""
    if config.use_denoiser:
        spectrum_for_model = denoise_batch(spectra, bundle, batch_size=config.batch_size)
    else:
        spectrum_for_model = spectra.astype(np.float32, copy=False)

    nn_mean, nn_std, _ = predict_mc_dropout_batch(
        spectrum_for_model,
        bundle,
        n_samples=config.mc_samples,
        batch_size=config.batch_size,
        seed=config.deterministic_dropout_seed,
    )
    return BatchPrediction(
        nn_mean=nn_mean,
        nn_std=nn_std,
        nn_ci95=1.96 * nn_std,
        spectrum_for_model=spectrum_for_model,
    )


def normalized_abs_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / PARAM_RANGE


def aggregate_param_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    abs_err = np.abs(err)
    norm_abs_err = normalized_abs_error(pred, true)
    out: dict[str, float] = {}
    for i, name in enumerate(PARAM_NAMES):
        out[f"{name}_mae"] = float(abs_err[:, i].mean())
        out[f"{name}_p50_abs"] = float(np.percentile(abs_err[:, i], 50))
        out[f"{name}_p90_abs"] = float(np.percentile(abs_err[:, i], 90))
        out[f"{name}_p99_abs"] = float(np.percentile(abs_err[:, i], 99))
        ss_res = float(np.sum((true[:, i] - pred[:, i]) ** 2))
        ss_tot = float(np.sum((true[:, i] - true[:, i].mean()) ** 2))
        out[f"{name}_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out["normalized_mae"] = float(norm_abs_err.mean())
    out["overall_r2"] = float(np.mean([out[f"{name}_r2"] for name in PARAM_NAMES]))
    return out


def spectral_mae_for_params(
    spectra: np.ndarray,
    params: np.ndarray,
    wavelengths: np.ndarray = WAVELENGTHS,
    batch_size: int = 8192,
) -> np.ndarray:
    out = np.empty(len(spectra), dtype=np.float32)
    for sl in iter_slices(len(spectra), batch_size):
        sim = simulate_params(params[sl], wavelengths=wavelengths)
        out[sl] = np.mean(np.abs(spectra[sl] - sim), axis=1).astype(np.float32)
    return out


def approx_visible_fringe_count(params: np.ndarray) -> np.ndarray:
    """
    Approximate number of interference cycles across 400-800 nm.

    This is a diagnostic, not exact TMM. Values below roughly 0.5 are where the
    inverse problem becomes visibly underconstrained.
    """
    params = np.asarray(params, dtype=np.float32)
    d = params[:, 0]
    n = params[:, 1]
    return 2.0 * n * d * ((1.0 / 400.0) - (1.0 / 800.0))


def failure_flags(
    pred: np.ndarray,
    true: np.ndarray,
    ci95: np.ndarray | None = None,
    spectral_mae: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    abs_err = np.abs(pred - true)
    norm_err = normalized_abs_error(pred, true).mean(axis=1)
    flags: dict[str, np.ndarray] = {
        "bad_thickness": abs_err[:, 0] > 10.0,
        "bad_n": abs_err[:, 1] > 0.10,
        "bad_k": abs_err[:, 2] > 0.05,
        "catastrophic": (abs_err[:, 0] > 25.0) | (abs_err[:, 1] > 0.25) | (abs_err[:, 2] > 0.10),
        "high_norm_error": norm_err > 0.10,
        "thin_film": true[:, 0] < 50.0,
        "very_thin_film": true[:, 0] < 35.0,
        "low_fringe": approx_visible_fringe_count(true) < 0.5,
    }
    if ci95 is not None:
        flags["ci_misses_thickness"] = abs_err[:, 0] > ci95[:, 0]
        flags["ci_misses_n"] = abs_err[:, 1] > ci95[:, 1]
        flags["ci_misses_k"] = abs_err[:, 2] > ci95[:, 2]
        flags["ci_misses_any"] = flags["ci_misses_thickness"] | flags["ci_misses_n"] | flags["ci_misses_k"]
    if spectral_mae is not None:
        flags["good_spectrum_bad_params"] = (spectral_mae < 0.002) & flags["high_norm_error"]
    return flags


def reliability_risk_proxy(
    pred_params: np.ndarray,
    ci95: np.ndarray,
) -> np.ndarray:
    """
    Fast reliability proxy in [0, 1], where higher means higher risk.

    This is batch-safe and cheap, so it can be used on million-sample scans.
    It complements the local Jacobian-based identifiability score used in the app.
    """
    pred = np.asarray(pred_params, dtype=np.float32)
    ci = np.asarray(ci95, dtype=np.float32)

    ci_t = np.clip(ci[:, 0] / 20.0, 0.0, 2.0)
    ci_n = np.clip(ci[:, 1] / 0.15, 0.0, 2.0)
    ci_k = np.clip(ci[:, 2] / 0.05, 0.0, 2.0)

    thickness_penalty = np.clip((50.0 - pred[:, 0]) / 50.0, 0.0, 1.0)
    fringe = approx_visible_fringe_count(pred)
    fringe_penalty = np.clip((0.5 - fringe) / 0.5, 0.0, 1.0)

    weighted = (
        0.30 * ci_t
        + 0.30 * ci_n
        + 0.15 * ci_k
        + 0.15 * thickness_penalty
        + 0.10 * fringe_penalty
    )
    return np.clip(weighted / 1.6, 0.0, 1.0).astype(np.float32)
