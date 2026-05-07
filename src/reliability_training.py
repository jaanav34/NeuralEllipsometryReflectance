"""
Utilities for reliability-focused training and calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.inference_pipeline import approx_visible_fringe_count
from src.paths import artifact_path

PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float32)
PARAM_MAX = np.array([300.0, 2.5, 0.5], dtype=np.float32)
PARAM_RANGE = PARAM_MAX - PARAM_MIN


def _sample_uniform(rng: np.random.Generator, n: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    u = rng.random((n, 3), dtype=np.float32)
    return lo + u * (hi - lo)


def sample_curriculum_params(
    n_samples: int,
    seed: int = 123,
    hard_cases_csv: Path | None = None,
) -> np.ndarray:
    """
    Sampling mix:
      30% normal stratified
      20% thin films (10..50 nm)
      20% low n (1.30..1.45)
      15% low n + low k
      15% benchmark hard cases (if available), else random fallback
    """
    rng = np.random.default_rng(seed)
    n_samples = int(n_samples)
    counts = {
        "normal": int(0.30 * n_samples),
        "thin": int(0.20 * n_samples),
        "low_n": int(0.20 * n_samples),
        "low_n_low_k": int(0.15 * n_samples),
    }
    used = sum(counts.values())
    counts["hard"] = n_samples - used

    chunks: list[np.ndarray] = []
    chunks.append(_sample_uniform(rng, counts["normal"], PARAM_MIN, PARAM_MAX))

    thin_lo = PARAM_MIN.copy()
    thin_hi = PARAM_MAX.copy()
    thin_hi[0] = 50.0
    chunks.append(_sample_uniform(rng, counts["thin"], thin_lo, thin_hi))

    low_n_lo = PARAM_MIN.copy()
    low_n_hi = PARAM_MAX.copy()
    low_n_hi[1] = 1.45
    chunks.append(_sample_uniform(rng, counts["low_n"], low_n_lo, low_n_hi))

    low_n_low_k_lo = PARAM_MIN.copy()
    low_n_low_k_hi = PARAM_MAX.copy()
    low_n_low_k_hi[1] = 1.45
    low_n_low_k_hi[2] = 0.08
    chunks.append(_sample_uniform(rng, counts["low_n_low_k"], low_n_low_k_lo, low_n_low_k_hi))

    hard_n = counts["hard"]
    hard_block = None
    if hard_cases_csv is not None and hard_cases_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(hard_cases_csv)
            cols = ["true_thickness_nm", "true_n", "true_k"]
            if all(col in df.columns for col in cols) and len(df) > 0:
                vals = df[cols].to_numpy(dtype=np.float32)
                idx = rng.choice(len(vals), size=hard_n, replace=len(vals) < hard_n)
                hard_block = vals[idx]
        except Exception:
            hard_block = None
    if hard_block is None:
        hard_block = _sample_uniform(rng, hard_n, PARAM_MIN, PARAM_MAX)
    chunks.append(hard_block)

    params = np.vstack(chunks).astype(np.float32)
    rng.shuffle(params, axis=0)
    return params


def hard_case_weight_map(params_phys: np.ndarray) -> np.ndarray:
    params = np.asarray(params_phys, dtype=np.float32)
    weights = np.ones(len(params), dtype=np.float32)
    thickness = params[:, 0]
    n_vals = params[:, 1]
    k_vals = params[:, 2]
    fringe = approx_visible_fringe_count(params)

    weights *= np.where(thickness < 50.0, 2.0, 1.0)
    weights *= np.where(n_vals < 1.45, 2.0, 1.0)
    weights *= np.where((n_vals < 1.45) & (k_vals < 0.08), 1.5, 1.0)
    weights *= np.where(fringe < 0.5, 2.0, 1.0)
    return weights.astype(np.float32)


def catastrophic_flags_from_error(abs_err: np.ndarray) -> np.ndarray:
    err = np.asarray(abs_err, dtype=np.float32)
    return (
        (err[:, 0] > 25.0)
        | (err[:, 1] > 0.25)
        | (err[:, 2] > 0.10)
    )


def calibrate_ci_scale(
    abs_err: np.ndarray,
    pred_std: np.ndarray,
    target_coverage: float = 0.95,
) -> np.ndarray:
    """
    Return multiplicative scale per parameter for predicted std.
    """
    err = np.asarray(abs_err, dtype=np.float64)
    std = np.asarray(pred_std, dtype=np.float64)
    std = np.maximum(std, 1e-9)
    z = err / std
    q = float(target_coverage)
    scales = np.quantile(z, q, axis=0)
    return np.asarray(scales, dtype=np.float32)


def save_calibration(scales: Iterable[float], name: str = "risk_ci_calibration_v5.npz") -> Path:
    out = artifact_path("data", name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, scales=np.asarray(tuple(scales), dtype=np.float32))
    return out
