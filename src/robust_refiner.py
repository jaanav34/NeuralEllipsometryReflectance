"""
Diagnostic and robust refiners for thin-film inverse benchmarking.

The existing src.refiner.refine_prediction is a single-start L-BFGS-B solve.
That is fine for a demo, but it hides whether failures come from:
  1. a bad NN initialization,
  2. local optimizer basin selection,
  3. non-identifiability where wrong parameters fit the same spectrum.

This module keeps the original function untouched and adds instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from src.tmm_simulator import simulate_reflectance, simulate_reflectance_batch

PARAM_BOUNDS_LIST = [(10.0, 300.0), (1.3, 2.5), (0.0, 0.5)]
PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float64)
PARAM_MAX = np.array([300.0, 2.5, 0.5], dtype=np.float64)
PARAM_RANGE = PARAM_MAX - PARAM_MIN
DEFAULT_WAVELENGTHS = np.linspace(400, 800, 200)


@dataclass(frozen=True)
class RefinerResult:
    thickness: float
    n: float
    k: float
    success: bool
    n_iterations: int
    init_residual: float
    final_residual: float
    improvement: float
    objective_calls: int
    start_index: int
    accepted: bool
    rejected_reason: str
    hit_lower_bound: bool
    hit_upper_bound: bool

    def params(self) -> np.ndarray:
        return np.array([self.thickness, self.n, self.k], dtype=np.float64)

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def _clip_params(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, dtype=np.float64), PARAM_MIN), PARAM_MAX)


def objective_for_spectrum(spectrum: np.ndarray, wavelengths: np.ndarray):
    spectrum64 = np.asarray(spectrum, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        t, n, k = params
        sim = simulate_reflectance(t, n, k, wavelengths)
        return float(np.mean((spectrum64 - sim) ** 2))

    return objective


def refine_from_start(
    spectrum: np.ndarray,
    start: np.ndarray,
    wavelengths: np.ndarray = DEFAULT_WAVELENGTHS,
    start_index: int = 0,
    maxiter: int = 200,
) -> RefinerResult:
    start = _clip_params(start)
    objective = objective_for_spectrum(spectrum, wavelengths)
    init_residual = objective(start)

    calls = 0

    def counted_objective(params: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return objective(params)

    res = minimize(
        counted_objective,
        start,
        method="L-BFGS-B",
        bounds=PARAM_BOUNDS_LIST,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )

    final_residual = float(res.fun)
    improvement = (1.0 - final_residual / init_residual) * 100.0 if init_residual > 0 else 0.0
    x = _clip_params(res.x)
    eps = np.array([1e-5, 1e-6, 1e-6], dtype=np.float64)
    hit_lower = bool(np.any(np.isclose(x, PARAM_MIN, atol=eps)))
    hit_upper = bool(np.any(np.isclose(x, PARAM_MAX, atol=eps)))

    return RefinerResult(
        thickness=float(x[0]),
        n=float(x[1]),
        k=float(x[2]),
        success=bool(res.success),
        n_iterations=int(res.nit),
        init_residual=float(init_residual),
        final_residual=final_residual,
        improvement=float(improvement),
        objective_calls=int(calls),
        start_index=int(start_index),
        accepted=True,
        rejected_reason="",
        hit_lower_bound=hit_lower,
        hit_upper_bound=hit_upper,
    )


def local_start_cloud(nn_params: np.ndarray) -> np.ndarray:
    """Small deterministic cloud around the NN prediction."""
    t, n, k = _clip_params(nn_params)
    starts = [np.array([t, n, k], dtype=np.float64)]
    for dt in (-30.0, -15.0, 15.0, 30.0):
        starts.append(np.array([t + dt, n, k], dtype=np.float64))
    for dn in (-0.25, -0.10, 0.10, 0.25):
        starts.append(np.array([t, n + dn, k], dtype=np.float64))
    for dk in (-0.10, -0.04, 0.04, 0.10):
        starts.append(np.array([t, n, k + dk], dtype=np.float64))
    # Coupled thickness-n degeneracy directions: higher d with lower n and vice versa.
    starts.extend([
        np.array([t + 25.0, n - 0.18, k], dtype=np.float64),
        np.array([t - 25.0, n + 0.18, k], dtype=np.float64),
        np.array([t + 15.0, n - 0.10, k + 0.03], dtype=np.float64),
        np.array([t - 15.0, n + 0.10, k - 0.03], dtype=np.float64),
    ])
    clipped = np.vstack([_clip_params(s) for s in starts])
    return np.unique(np.round(clipped, decimals=8), axis=0)


def coarse_grid_starts(
    spectrum: np.ndarray,
    wavelengths: np.ndarray = DEFAULT_WAVELENGTHS,
    thickness_grid: Iterable[float] | None = None,
    n_grid: Iterable[float] | None = None,
    k_grid: Iterable[float] | None = None,
    top_k: int = 5,
) -> np.ndarray:
    """
    Find top coarse grid starts by spectral MSE.

    This is expensive but very useful for diagnosing catastrophic refiner cases.
    Use it on failure subsets, not millions of samples.
    """
    if thickness_grid is None:
        thickness_grid = np.linspace(10, 300, 30)
    if n_grid is None:
        n_grid = np.linspace(1.3, 2.5, 13)
    if k_grid is None:
        k_grid = np.linspace(0.0, 0.5, 11)

    tt, nn, kk = np.meshgrid(thickness_grid, n_grid, k_grid, indexing="ij")
    candidates = np.column_stack([tt.ravel(), nn.ravel(), kk.ravel()]).astype(np.float64)
    sim = simulate_reflectance_batch(candidates[:, 0], candidates[:, 1], candidates[:, 2], wavelengths)
    mse = np.mean((sim - np.asarray(spectrum, dtype=np.float64)[None, :]) ** 2, axis=1)
    idx = np.argsort(mse)[:top_k]
    return candidates[idx]


def refine_prediction_diagnostic(
    spectrum: np.ndarray,
    init_thickness: float,
    init_n: float,
    init_k: float,
    wavelengths: np.ndarray = DEFAULT_WAVELENGTHS,
    maxiter: int = 200,
    accept_only_if_improves: bool = True,
) -> RefinerResult:
    start = np.array([init_thickness, init_n, init_k], dtype=np.float64)
    result = refine_from_start(spectrum, start, wavelengths=wavelengths, maxiter=maxiter)
    if accept_only_if_improves and result.final_residual > result.init_residual:
        return RefinerResult(
            thickness=float(start[0]),
            n=float(start[1]),
            k=float(start[2]),
            success=result.success,
            n_iterations=result.n_iterations,
            init_residual=result.init_residual,
            final_residual=result.init_residual,
            improvement=0.0,
            objective_calls=result.objective_calls,
            start_index=result.start_index,
            accepted=False,
            rejected_reason="final residual worse than initial residual",
            hit_lower_bound=result.hit_lower_bound,
            hit_upper_bound=result.hit_upper_bound,
        )
    return result


def refine_prediction_multistart(
    spectrum: np.ndarray,
    init_thickness: float,
    init_n: float,
    init_k: float,
    wavelengths: np.ndarray = DEFAULT_WAVELENGTHS,
    include_coarse_grid: bool = False,
    maxiter: int = 200,
) -> RefinerResult:
    nn = np.array([init_thickness, init_n, init_k], dtype=np.float64)
    starts = [s for s in local_start_cloud(nn)]
    if include_coarse_grid:
        starts.extend([s for s in coarse_grid_starts(spectrum, wavelengths=wavelengths)])

    results = [
        refine_from_start(spectrum, s, wavelengths=wavelengths, start_index=i, maxiter=maxiter)
        for i, s in enumerate(starts)
    ]
    best = min(results, key=lambda r: r.final_residual)
    return best
