"""
Reliability and identifiability diagnostics for thin-film inverse inference.

These helpers keep the legacy prediction path unchanged and add quantitative
signals that indicate when a single parameter tuple is not uniquely supported
by the spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from src.tmm_simulator import simulate_reflectance

DEFAULT_WAVELENGTHS = np.linspace(400, 800, 200).astype(np.float64)
PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float64)
PARAM_MAX = np.array([300.0, 2.5, 0.5], dtype=np.float64)


@dataclass(frozen=True)
class IdentifiabilityResult:
    level: str
    score: float
    condition_number: float
    smallest_singular_value: float
    largest_singular_value: float
    fringe_estimate: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, float | str | tuple[str, ...]]:
        return asdict(self)


def _clip_params(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    return np.minimum(np.maximum(p, PARAM_MIN), PARAM_MAX)


def finite_difference_jacobian(
    params: Iterable[float],
    wavelengths: np.ndarray = DEFAULT_WAVELENGTHS,
    steps: tuple[float, float, float] = (0.25, 0.0025, 0.0010),
) -> np.ndarray:
    """
    Compute dR/d(thickness,n,k) via central finite differences.

    Returns Jacobian with shape (num_wavelengths, 3).
    """
    p0 = _clip_params(np.asarray(tuple(params), dtype=np.float64))
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    jac = np.empty((len(wavelengths), 3), dtype=np.float64)

    for i, step in enumerate(steps):
        lo = p0.copy()
        hi = p0.copy()
        lo[i] = max(PARAM_MIN[i], p0[i] - step)
        hi[i] = min(PARAM_MAX[i], p0[i] + step)
        delta = hi[i] - lo[i]
        if delta <= 1e-12:
            jac[:, i] = 0.0
            continue
        r_lo = simulate_reflectance(lo[0], lo[1], lo[2], wavelengths)
        r_hi = simulate_reflectance(hi[0], hi[1], hi[2], wavelengths)
        jac[:, i] = (r_hi - r_lo) / delta
    return jac


def fisher_information_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    j = np.asarray(jacobian, dtype=np.float64)
    return j.T @ j


def singular_values_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    j = np.asarray(jacobian, dtype=np.float64)
    return np.linalg.svd(j, compute_uv=False)


def approx_visible_fringe_count(thickness_nm: float, n_val: float) -> float:
    thickness = float(thickness_nm)
    refractive_index = float(n_val)
    return float(2.0 * refractive_index * thickness * ((1.0 / 400.0) - (1.0 / 800.0)))


def classify_identifiability(
    *,
    thickness_nm: float,
    n_val: float,
    ci95: tuple[float, float, float],
    singular_values: np.ndarray,
) -> IdentifiabilityResult:
    sv = np.asarray(singular_values, dtype=np.float64)
    sv_sorted = np.sort(np.maximum(sv, 0.0))
    smallest = float(sv_sorted[0]) if len(sv_sorted) else 0.0
    largest = float(sv_sorted[-1]) if len(sv_sorted) else 0.0
    condition_number = float(largest / max(smallest, 1e-12)) if largest > 0 else float("inf")
    ci_t, ci_n, ci_k = [float(x) for x in ci95]
    fringe = approx_visible_fringe_count(thickness_nm, n_val)

    reasons: list[str] = []
    score = 1.0

    if float(thickness_nm) < 50.0:
        reasons.append("thin film under 50 nm")
        score -= 0.30
    if fringe < 0.5:
        reasons.append("low fringe structure")
        score -= 0.25
    if ci_t > 20.0:
        reasons.append("wide thickness confidence interval")
        score -= 0.15
    if ci_n > 0.15:
        reasons.append("wide n confidence interval")
        score -= 0.15
    if ci_k > 0.05:
        reasons.append("wide k confidence interval")
        score -= 0.10
    if condition_number > 3_000.0:
        reasons.append("high local condition number")
        score -= 0.20
    if smallest < 5e-4:
        reasons.append("weak local spectral sensitivity")
        score -= 0.20

    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.40:
        level = "Weak"
    elif score < 0.70:
        level = "Moderate"
    else:
        level = "Strong"

    if not reasons:
        reasons.append("well-conditioned local response and tight uncertainty")

    return IdentifiabilityResult(
        level=level,
        score=score,
        condition_number=condition_number,
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        fringe_estimate=fringe,
        reasons=tuple(reasons),
    )


def evaluate_identifiability(
    params: Iterable[float],
    ci95: tuple[float, float, float],
    wavelengths: np.ndarray = DEFAULT_WAVELENGTHS,
) -> IdentifiabilityResult:
    p = _clip_params(np.asarray(tuple(params), dtype=np.float64))
    jacobian = finite_difference_jacobian(p, wavelengths=wavelengths)
    sv = singular_values_from_jacobian(jacobian)
    return classify_identifiability(
        thickness_nm=float(p[0]),
        n_val=float(p[1]),
        ci95=ci95,
        singular_values=sv,
    )
