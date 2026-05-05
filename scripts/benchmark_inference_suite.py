from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference_pipeline import (
    PARAM_NAMES,
    WAVELENGTHS,
    PipelineConfig,
    add_noise,
    aggregate_param_metrics,
    approx_visible_fringe_count,
    failure_flags,
    iter_slices,
    load_bundle,
    normalized_abs_error,
    predict_app_style_batch,
    reliability_risk_proxy,
    simulate_params,
    spectral_mae_for_params,
)
from src.paths import artifact_path, ensure_parent_dir
from src.robust_refiner import (
    refine_prediction_guarded_multistart,
    refine_prediction_batch_gpu,
    refine_prediction_diagnostic,
    refine_prediction_multistart,
)
from src.tmm_simulator import simulate_reflectance


FAILURE_FIELDNAMES = [
    "rank",
    "source",
    "noise_std",
    "true_thickness_nm",
    "true_n",
    "true_k",
    "nn_thickness_nm",
    "nn_n",
    "nn_k",
    "refined_thickness_nm",
    "refined_n",
    "refined_k",
    "nn_abs_d",
    "nn_abs_n",
    "nn_abs_k",
    "ref_abs_d",
    "ref_abs_n",
    "ref_abs_k",
    "nn_norm_error",
    "ref_norm_error",
    "nn_spectral_mae",
    "ref_spectral_mae",
    "ci95_d",
    "ci95_n",
    "ci95_k",
    "fringe_count_est",
    "refiner_improvement_pct",
    "refiner_iterations",
    "refiner_success",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test the exact app inference path over synthetic thin-film parameter space."
    )
    p.add_argument("--n-random", type=int, default=100_000, help="Uniform random samples.")
    p.add_argument("--grid", action="store_true", help="Also generate Cartesian grid samples.")
    p.add_argument("--grid-thickness", type=int, default=120)
    p.add_argument("--grid-n", type=int, default=80)
    p.add_argument("--grid-k", type=int, default=80)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.001, 0.005])
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--mc-samples", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-denoiser", action="store_true")
    p.add_argument(
        "--refine-strategy",
        choices=["none", "all", "random", "top_nn_error", "catastrophic"],
        default="catastrophic",
        help="Which cases receive CPU L-BFGS-B refinement. Use all only for overnight runs.",
    )
    p.add_argument("--max-refine", type=int, default=5000)
    p.add_argument("--refiner-workers", type=int, default=0, help="0 means serial. Processes are CPU-only.")
    p.add_argument("--robust-refiner", action="store_true", help="Use deterministic multi-start refiner on selected cases.")
    p.add_argument("--coarse-grid-on-top", type=int, default=25, help="Run coarse-grid multi-start on top N failures only.")
    p.add_argument("--gpu-refiner", action="store_true", help="Use batched GPU refiner for selected cases.")
    p.add_argument("--gpu-refiner-steps", type=int, default=120, help="Optimization steps for GPU refiner.")
    p.add_argument("--gpu-refiner-lr", type=float, default=0.05, help="Learning rate for GPU refiner.")
    p.add_argument("--gpu-refiner-chunk-size", type=int, default=2048, help="Chunk size for GPU refiner.")
    p.add_argument("--guarded-refiner", action="store_true", help="Use prior-regularized multi-start refiner.")
    p.add_argument("--guarded-prior-lambda", type=float, default=0.10, help="Prior weight for guarded refiner.")
    p.add_argument(
        "--guarded-trust-region",
        type=float,
        nargs=3,
        metavar=("DT", "DN", "DK"),
        default=[40.0, 0.25, 0.10],
        help="Trust-region half-widths for thickness, n, and k in guarded refiner.",
    )
    p.add_argument("--pipeline-label", default="default", help="Label saved in summary outputs for matrix runs.")
    p.add_argument("--output-stem", default="inference_stress_v4")
    p.add_argument("--top-k-failures", type=int, default=250)
    p.add_argument("--include-probes", action="store_true", help="Include hard-coded examples from manual app testing.")
    return p.parse_args()


def make_random_params(n: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(10.0, 300.0, n)
    nval = rng.uniform(1.3, 2.5, n)
    kval = rng.uniform(0.0, 0.5, n)
    return np.column_stack([t, nval, kval]).astype(np.float32)


def make_grid_params(nt: int, nn: int, nk: int) -> np.ndarray:
    t = np.linspace(10.0, 300.0, nt, dtype=np.float32)
    n = np.linspace(1.3, 2.5, nn, dtype=np.float32)
    k = np.linspace(0.0, 0.5, nk, dtype=np.float32)
    tt, nnv, kk = np.meshgrid(t, n, k, indexing="ij")
    return np.column_stack([tt.ravel(), nnv.ravel(), kk.ravel()]).astype(np.float32)


def make_probe_params() -> np.ndarray:
    return np.array(
        [
            [63.0, 1.7700, 0.2000],
            [31.0, 1.8400, 0.3200],
            [66.0, 1.7000, 0.1540],
            [34.0, 1.4200, 0.1540],
            [109.0, 1.5500, 0.2180],
            [137.0, 1.7200, 0.2180],
        ],
        dtype=np.float32,
    )


def select_refine_indices(
    strategy: str,
    true_params: np.ndarray,
    nn_params: np.ndarray,
    ci95: np.ndarray,
    nn_spectral_mae: np.ndarray,
    max_refine: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(true_params)
    if strategy == "none" or max_refine <= 0:
        return np.array([], dtype=np.int64)
    if strategy == "all":
        return np.arange(n, dtype=np.int64)
    if strategy == "random":
        return rng.choice(n, size=min(max_refine, n), replace=False).astype(np.int64)

    flags = failure_flags(nn_params, true_params, ci95=ci95, spectral_mae=nn_spectral_mae)
    if strategy == "catastrophic":
        candidates = np.flatnonzero(flags["catastrophic"] | flags["bad_n"] | flags["bad_k"])
    elif strategy == "top_nn_error":
        candidates = np.arange(n)
    else:
        raise ValueError(strategy)

    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    score = normalized_abs_error(nn_params[candidates], true_params[candidates]).mean(axis=1)
    order = np.argsort(score)[::-1]
    selected = candidates[order[: min(max_refine, len(order))]]
    return selected.astype(np.int64)


def _refine_worker(
    payload: tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        bool,
        bool,
        float,
        tuple[float, float, float],
    ]
) -> tuple[int, dict[str, Any]]:
    idx, spectrum, init, ci95, robust, coarse, guarded, prior_lambda, trust_region = payload
    if guarded:
        bundle = refine_prediction_guarded_multistart(
            spectrum,
            float(init[0]),
            float(init[1]),
            float(init[2]),
            ci95=ci95,
            wavelengths=WAVELENGTHS,
            include_coarse_grid=coarse,
            trust_region=trust_region,
            lambda_prior=prior_lambda,
            max_alternatives=3,
        )
        res_dict = bundle.primary.to_dict()
        res_dict["n_alternatives"] = len(bundle.alternatives)
        return idx, res_dict
    if robust:
        res = refine_prediction_multistart(
            spectrum,
            float(init[0]),
            float(init[1]),
            float(init[2]),
            WAVELENGTHS,
            include_coarse_grid=coarse,
        )
    else:
        res = refine_prediction_diagnostic(
            spectrum,
            float(init[0]),
            float(init[1]),
            float(init[2]),
            WAVELENGTHS,
        )
    return idx, res.to_dict()


def run_refiner_subset(
    spectra: np.ndarray,
    nn_params: np.ndarray,
    selected: np.ndarray,
    ci95: np.ndarray,
    robust: bool,
    guarded: bool,
    guarded_prior_lambda: float,
    guarded_trust_region: tuple[float, float, float],
    coarse_grid_on_top: int,
    workers: int,
    use_gpu_refiner: bool,
    gpu_refiner_steps: int,
    gpu_refiner_lr: float,
    gpu_refiner_chunk_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    refined = np.full((len(spectra), 3), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    selected_set = list(map(int, selected))
    if not selected_set:
        return refined, records

    if use_gpu_refiner:
        t0 = time.perf_counter()
        subset_spectra = spectra[selected_set].astype(np.float32, copy=False)
        subset_init = nn_params[selected_set].astype(np.float32, copy=False)
        refined_subset, subset_records = refine_prediction_batch_gpu(
            subset_spectra,
            subset_init,
            WAVELENGTHS,
            steps=gpu_refiner_steps,
            lr=gpu_refiner_lr,
            chunk_size=gpu_refiner_chunk_size,
            device="cuda",
            accept_only_if_improves=True,
        )
        for local_i, idx in enumerate(selected_set):
            refined[idx] = refined_subset[local_i]
            subset_records[local_i]["sample_index"] = int(idx)
        records.extend(subset_records)
        elapsed = time.perf_counter() - t0
        rate = len(selected_set) / max(elapsed, 1e-9)
        print(f"  refined {len(selected_set):,}/{len(selected_set):,} cases ({rate:.1f}/s)", flush=True)
        return refined, records

    payloads = []
    for rank, idx in enumerate(selected_set):
        coarse = bool(robust and rank < coarse_grid_on_top)
        payloads.append(
            (
                idx,
                spectra[idx].astype(np.float64),
                nn_params[idx].astype(np.float64),
                ci95[idx].astype(np.float64),
                robust,
                coarse,
                guarded,
                float(guarded_prior_lambda),
                guarded_trust_region,
            )
        )

    t0 = time.perf_counter()
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_refine_worker, p) for p in payloads]
            for done, fut in enumerate(as_completed(futures), 1):
                idx, rec = fut.result()
                refined[idx] = [rec["thickness"], rec["n"], rec["k"]]
                rec["sample_index"] = idx
                records.append(rec)
                if done % 100 == 0 or done == len(futures):
                    rate = done / max(time.perf_counter() - t0, 1e-9)
                    print(f"  refined {done:,}/{len(futures):,} cases ({rate:.1f}/s)", flush=True)
    else:
        for done, p in enumerate(payloads, 1):
            idx, rec = _refine_worker(p)
            refined[idx] = [rec["thickness"], rec["n"], rec["k"]]
            rec["sample_index"] = idx
            records.append(rec)
            if done % 100 == 0 or done == len(payloads):
                rate = done / max(time.perf_counter() - t0, 1e-9)
                print(f"  refined {done:,}/{len(payloads):,} cases ({rate:.1f}/s)", flush=True)
    return refined, records


def slice_metrics(true_params: np.ndarray, pred: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bins = {
        "thickness": [10, 25, 35, 50, 75, 100, 150, 200, 300.001],
        "n": [1.3, 1.45, 1.6, 1.8, 2.0, 2.25, 2.5001],
        "k": [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.5001],
        "fringe_est": [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 99.0],
    }
    values = {
        "thickness": true_params[:, 0],
        "n": true_params[:, 1],
        "k": true_params[:, 2],
        "fringe_est": approx_visible_fringe_count(true_params),
    }
    for dim, edges in bins.items():
        vals = values[dim]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (vals >= lo) & (vals < hi)
            if not np.any(mask):
                continue
            metrics = aggregate_param_metrics(pred[mask], true_params[mask])
            row = {"model": prefix, "slice_dim": dim, "slice_lo": lo, "slice_hi": hi, "count": int(mask.sum())}
            row.update(metrics)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_failure_rows(
    true_params: np.ndarray,
    nn_params: np.ndarray,
    refined_params: np.ndarray,
    nn_spectral_mae: np.ndarray,
    ref_spectral_mae: np.ndarray,
    ci95: np.ndarray,
    ref_records: list[dict[str, Any]],
    noise_std: float,
    top_k: int,
) -> list[dict[str, Any]]:
    ref_by_idx = {int(r["sample_index"]): r for r in ref_records}
    ref_available = np.isfinite(refined_params[:, 0])
    effective_ref = np.where(ref_available[:, None], refined_params, nn_params)
    nn_norm = normalized_abs_error(nn_params, true_params).mean(axis=1)
    ref_norm = normalized_abs_error(effective_ref, true_params).mean(axis=1)
    # Prioritize cases where the pipeline is bad, or spectral fit improved while parameters got worse.
    score = np.maximum(nn_norm, ref_norm)
    score += ((ref_available) & (ref_spectral_mae < nn_spectral_mae) & (ref_norm > nn_norm)).astype(np.float32) * 0.25
    order = np.argsort(score)[::-1][:top_k]

    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, 1):
        ref_rec = ref_by_idx.get(int(idx), {})
        refp = effective_ref[idx]
        nn_abs = np.abs(nn_params[idx] - true_params[idx])
        ref_abs = np.abs(refp - true_params[idx])
        reasons = []
        if true_params[idx, 0] < 50:
            reasons.append("thin film")
        if approx_visible_fringe_count(true_params[idx : idx + 1])[0] < 0.5:
            reasons.append("low fringe count")
        if nn_abs[1] > 0.10:
            reasons.append("NN n error")
        if ref_available[idx] and ref_norm[idx] > nn_norm[idx] * 1.2:
            reasons.append("refiner worsened parameter error")
        if ref_available[idx] and ref_spectral_mae[idx] < 0.002 and ref_norm[idx] > 0.10:
            reasons.append("good spectrum, bad parameters")
        rows.append(
            {
                "rank": rank,
                "source": "synthetic",
                "noise_std": noise_std,
                "true_thickness_nm": true_params[idx, 0],
                "true_n": true_params[idx, 1],
                "true_k": true_params[idx, 2],
                "nn_thickness_nm": nn_params[idx, 0],
                "nn_n": nn_params[idx, 1],
                "nn_k": nn_params[idx, 2],
                "refined_thickness_nm": refp[0],
                "refined_n": refp[1],
                "refined_k": refp[2],
                "nn_abs_d": nn_abs[0],
                "nn_abs_n": nn_abs[1],
                "nn_abs_k": nn_abs[2],
                "ref_abs_d": ref_abs[0],
                "ref_abs_n": ref_abs[1],
                "ref_abs_k": ref_abs[2],
                "nn_norm_error": nn_norm[idx],
                "ref_norm_error": ref_norm[idx],
                "nn_spectral_mae": nn_spectral_mae[idx],
                "ref_spectral_mae": ref_spectral_mae[idx] if np.isfinite(ref_spectral_mae[idx]) else "",
                "ci95_d": ci95[idx, 0],
                "ci95_n": ci95[idx, 1],
                "ci95_k": ci95[idx, 2],
                "fringe_count_est": approx_visible_fringe_count(true_params[idx : idx + 1])[0],
                "refiner_improvement_pct": ref_rec.get("improvement", ""),
                "refiner_iterations": ref_rec.get("n_iterations", ""),
                "refiner_success": ref_rec.get("success", ""),
                "failure_reason": "; ".join(reasons),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.gpu_refiner and not torch.cuda.is_available():
        raise RuntimeError("GPU refiner requested but CUDA is not available on this machine.")

    out_dir = ensure_parent_dir(artifact_path("benchmarks", args.output_stem, "dummy.txt")).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    params_parts = []
    if args.n_random > 0:
        params_parts.append(make_random_params(args.n_random, rng))
    if args.grid:
        params_parts.append(make_grid_params(args.grid_thickness, args.grid_n, args.grid_k))
    if args.include_probes:
        params_parts.append(make_probe_params())
    true_params = np.vstack(params_parts).astype(np.float32)

    print("Inference stress suite")
    print("=" * 100)
    print(f"samples: {len(true_params):,}")
    print(f"noise levels: {args.noise_levels}")
    print(f"mc samples: {args.mc_samples}")
    print(f"denoiser: {not args.no_denoiser}")
    print(f"refine strategy: {args.refine_strategy}, max_refine={args.max_refine}, robust={args.robust_refiner}")
    print(f"guarded refiner: {args.guarded_refiner} (lambda={args.guarded_prior_lambda})")
    print(f"pipeline label: {args.pipeline_label}")
    if args.gpu_refiner:
        print(
            "gpu refiner: "
            f"enabled, steps={args.gpu_refiner_steps}, lr={args.gpu_refiner_lr}, "
            f"chunk={args.gpu_refiner_chunk_size}"
        )
    print("=" * 100, flush=True)

    bundle = load_bundle(device=args.device, load_denoiser=not args.no_denoiser)
    config = PipelineConfig(
        use_denoiser=not args.no_denoiser,
        mc_samples=args.mc_samples,
        batch_size=args.batch_size,
        device=args.device,
    )

    clean_spectra = simulate_params(true_params, WAVELENGTHS)

    summary: dict[str, Any] = {
        "args": vars(args),
        "n_samples": int(len(true_params)),
        "started_at_unix": time.time(),
        "noise_results": [],
    }
    all_failure_rows: list[dict[str, Any]] = []
    all_slice_rows: list[dict[str, Any]] = []

    for noise_std in args.noise_levels:
        print(f"\nNoise std={noise_std:.4f}", flush=True)
        t_noise = time.perf_counter()
        spectra = add_noise(clean_spectra, noise_std, rng)

        print("  running denoiser + MC Dropout NN...", flush=True)
        t0 = time.perf_counter()
        pred = predict_app_style_batch(spectra, bundle, config)
        nn_time = time.perf_counter() - t0
        print(f"  NN path done in {nn_time:.1f}s", flush=True)

        print("  computing NN spectral residuals...", flush=True)
        nn_spectral_mae = spectral_mae_for_params(spectra, pred.nn_mean, WAVELENGTHS, batch_size=args.batch_size)
        nn_metrics = aggregate_param_metrics(pred.nn_mean, true_params)
        nn_flags = failure_flags(pred.nn_mean, true_params, ci95=pred.nn_ci95, spectral_mae=nn_spectral_mae)
        nn_risk = reliability_risk_proxy(pred.nn_mean, pred.nn_ci95)

        selected = select_refine_indices(
            args.refine_strategy,
            true_params,
            pred.nn_mean,
            pred.nn_ci95,
            nn_spectral_mae,
            args.max_refine,
            rng,
        )
        print(f"  selected {len(selected):,} cases for refinement", flush=True)

        refined = np.full_like(pred.nn_mean, np.nan, dtype=np.float32)
        ref_records: list[dict[str, Any]] = []
        ref_spectral_mae = np.full(len(true_params), np.nan, dtype=np.float32)
        ref_metrics: dict[str, float] = {}
        if len(selected):
            refined, ref_records = run_refiner_subset(
                spectra,
                pred.nn_mean,
                selected,
                pred.nn_ci95,
                robust=args.robust_refiner,
                guarded=args.guarded_refiner,
                guarded_prior_lambda=float(args.guarded_prior_lambda),
                guarded_trust_region=tuple(float(v) for v in args.guarded_trust_region),
                coarse_grid_on_top=args.coarse_grid_on_top,
                workers=args.refiner_workers,
                use_gpu_refiner=args.gpu_refiner,
                gpu_refiner_steps=args.gpu_refiner_steps,
                gpu_refiner_lr=args.gpu_refiner_lr,
                gpu_refiner_chunk_size=args.gpu_refiner_chunk_size,
            )
            print("  computing refined spectral residuals...", flush=True)
            ref_spectral_mae[selected] = spectral_mae_for_params(
                spectra[selected], refined[selected], WAVELENGTHS, batch_size=args.batch_size
            )
            ref_metrics = aggregate_param_metrics(refined[selected], true_params[selected])
        ref_worse_rate = None
        ref_improved_spectral_rate = None
        if len(selected):
            nn_norm_sel = normalized_abs_error(pred.nn_mean[selected], true_params[selected]).mean(axis=1)
            ref_norm_sel = normalized_abs_error(refined[selected], true_params[selected]).mean(axis=1)
            ref_worse_rate = float(np.mean(ref_norm_sel > nn_norm_sel))
            ref_improved_spectral_rate = float(np.mean(ref_spectral_mae[selected] < nn_spectral_mae[selected]))

        failures = make_failure_rows(
            true_params,
            pred.nn_mean,
            refined,
            nn_spectral_mae,
            ref_spectral_mae,
            pred.nn_ci95,
            ref_records,
            noise_std,
            args.top_k_failures,
        )
        all_failure_rows.extend(failures)
        all_slice_rows.extend(slice_metrics(true_params, pred.nn_mean, prefix=f"nn_noise_{noise_std}"))
        if len(selected):
            all_slice_rows.extend(slice_metrics(true_params[selected], refined[selected], prefix=f"refined_noise_{noise_std}"))

        result = {
            "noise_std": float(noise_std),
            "pipeline_label": args.pipeline_label,
            "wall_time_s": float(time.perf_counter() - t_noise),
            "nn_time_s": float(nn_time),
            "nn_metrics": nn_metrics,
            "nn_flag_rates": {k: float(v.mean()) for k, v in nn_flags.items()},
            "nn_risk_summary": {
                "mean": float(np.mean(nn_risk)),
                "p90": float(np.percentile(nn_risk, 90)),
                "p99": float(np.percentile(nn_risk, 99)),
                "fraction_over_0p7": float(np.mean(nn_risk > 0.7)),
            },
            "n_refined": int(len(selected)),
            "refined_metrics_on_selected": ref_metrics,
            "refiner_records_summary": {
                "success_rate": float(np.mean([r["success"] for r in ref_records])) if ref_records else None,
                "mean_iterations": float(np.mean([r["n_iterations"] for r in ref_records])) if ref_records else None,
                "mean_improvement_pct": float(np.mean([r["improvement"] for r in ref_records])) if ref_records else None,
                "worsened_param_rate": ref_worse_rate,
                "improved_spectral_rate": ref_improved_spectral_rate,
            },
        }
        summary["noise_results"].append(result)

        print("  summary:")
        print(f"    NN thickness MAE: {nn_metrics['thickness_nm_mae']:.4f} nm")
        print(f"    NN n MAE:         {nn_metrics['n_mae']:.5f}")
        print(f"    NN k MAE:         {nn_metrics['k_mae']:.5f}")
        print(f"    NN catastrophic:  {result['nn_flag_rates']['catastrophic'] * 100:.2f}%")
        print(f"    CI miss any:      {result['nn_flag_rates'].get('ci_misses_any', 0.0) * 100:.2f}%")
        print(f"    Risk > 0.7:       {result['nn_risk_summary']['fraction_over_0p7'] * 100:.2f}%")
        if ref_metrics:
            print(f"    refined subset n MAE: {ref_metrics['n_mae']:.5f}")
            print(
                "    refiner worsened params: "
                f"{result['refiner_records_summary']['worsened_param_rate'] * 100:.2f}%"
            )

        # Save compact arrays per noise level. This is intentionally NPZ, not CSV, for scale.
        npz_path = out_dir / f"{args.output_stem}_noise_{str(noise_std).replace('.', 'p')}.npz"
        np.savez_compressed(
            npz_path,
            true_params=true_params,
            nn_mean=pred.nn_mean,
            nn_std=pred.nn_std,
            nn_ci95=pred.nn_ci95,
            nn_spectral_mae=nn_spectral_mae,
            selected_refine_indices=selected,
            refined_params=refined,
            ref_spectral_mae=ref_spectral_mae,
        )
        print(f"  saved arrays: {npz_path}", flush=True)

    summary["finished_at_unix"] = time.time()
    summary_path = out_dir / f"{args.output_stem}_summary.json"
    failure_path = out_dir / f"{args.output_stem}_top_failures.csv"
    slices_path = out_dir / f"{args.output_stem}_slice_metrics.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(failure_path, all_failure_rows, fieldnames=FAILURE_FIELDNAMES)
    write_csv(slices_path, all_slice_rows)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {failure_path}")
    print(f"  {slices_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
