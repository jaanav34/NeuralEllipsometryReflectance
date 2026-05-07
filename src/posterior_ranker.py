from __future__ import annotations

import numpy as np
import torch

from src.spectranet_mdn import sample_mdn_posterior
from src.tmm_simulator import simulate_reflectance_batch

PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float32)
PARAM_RANGE = np.array([290.0, 1.2, 0.5], dtype=np.float32)


@torch.no_grad()
def rank_mdn_posterior_candidates(
    spectrum: np.ndarray,
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    wavelengths: np.ndarray,
    n_samples: int = 96,
    top_k: int = 5,
) -> list[dict[str, float]]:
    """
    Sample posterior candidates and rank by TMM residual to input spectrum.
    """
    samples_norm = sample_mdn_posterior(logits, means, scales, n_samples=n_samples)[0].cpu().numpy()
    params_phys = samples_norm * PARAM_RANGE + PARAM_MIN
    sim = simulate_reflectance_batch(
        params_phys[:, 0].astype(np.float64),
        params_phys[:, 1].astype(np.float64),
        params_phys[:, 2].astype(np.float64),
        wavelengths.astype(np.float64),
    )
    mae = np.mean(np.abs(sim - spectrum[None, :].astype(np.float64)), axis=1)
    order = np.argsort(mae)[:top_k]
    out: list[dict[str, float]] = []
    for rank, idx in enumerate(order, 1):
        p = params_phys[idx]
        out.append(
            {
                "rank": float(rank),
                "thickness": float(p[0]),
                "n": float(p[1]),
                "k": float(p[2]),
                "spectral_mae": float(mae[idx]),
            }
        )
    return out
