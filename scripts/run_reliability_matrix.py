from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import artifact_path, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reliability benchmark matrix across key app pipeline modes."
    )
    parser.add_argument("--n-random", type=int, default=100_000)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.001, 0.005])
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-refine", type=int, default=5_000)
    parser.add_argument("--refiner-workers", type=int, default=0)
    parser.add_argument("--gpu-refiner", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-stem", default="reliability_matrix")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def run_mode(mode: str, args: argparse.Namespace) -> Path:
    stem = f"{args.output_stem}_{mode}"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "benchmark_inference_suite.py"),
        "--n-random",
        str(args.n_random),
        "--noise-levels",
        *[str(x) for x in args.noise_levels],
        "--mc-samples",
        str(args.mc_samples),
        "--batch-size",
        str(args.batch_size),
        "--max-refine",
        str(args.max_refine),
        "--refiner-workers",
        str(args.refiner_workers),
        "--seed",
        str(args.seed),
        "--output-stem",
        stem,
        "--pipeline-label",
        mode,
        "--device",
        args.device,
    ]

    if mode == "nn_only":
        cmd.extend(["--no-denoiser", "--refine-strategy", "none"])
    elif mode == "denoiser_nn":
        cmd.extend(["--refine-strategy", "none"])
    elif mode == "nn_guarded_refiner":
        cmd.extend(["--no-denoiser", "--refine-strategy", "catastrophic", "--guarded-refiner"])
    elif mode == "denoiser_nn_guarded_refiner":
        cmd.extend(["--refine-strategy", "catastrophic", "--guarded-refiner"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if args.gpu_refiner and "guarded_refiner" in mode:
        cmd.append("--gpu-refiner")

    print(f"\nRunning mode: {mode}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return artifact_path("benchmarks", stem, f"{stem}_summary.json")


def flatten_summary(mode: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in summary.get("noise_results", []):
        rows.append(
            {
                "mode": mode,
                "noise_std": result["noise_std"],
                "samples": summary.get("n_samples"),
                "nn_thickness_mae": result["nn_metrics"]["thickness_nm_mae"],
                "nn_n_mae": result["nn_metrics"]["n_mae"],
                "nn_k_mae": result["nn_metrics"]["k_mae"],
                "nn_catastrophic_rate": result["nn_flag_rates"]["catastrophic"],
                "nn_ci_miss_any": result["nn_flag_rates"].get("ci_misses_any"),
                "nn_good_spectrum_bad_params": result["nn_flag_rates"].get("good_spectrum_bad_params"),
                "risk_gt_0p7": result["nn_risk_summary"]["fraction_over_0p7"],
                "n_refined": result["n_refined"],
                "refiner_worsened_param_rate": result["refiner_records_summary"]["worsened_param_rate"],
                "refiner_improved_spectral_rate": result["refiner_records_summary"]["improved_spectral_rate"],
                "wall_time_s": result["wall_time_s"],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    modes = [
        "nn_only",
        "denoiser_nn",
        "nn_guarded_refiner",
        "denoiser_nn_guarded_refiner",
    ]

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for mode in modes:
        summary_path = run_mode(mode, args)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[mode] = summary
        rows.extend(flatten_summary(mode, summary))

    out_dir = ensure_parent_dir(artifact_path("benchmarks", args.output_stem, "dummy.txt")).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.output_stem}_matrix_summary.json"
    csv_path = out_dir / f"{args.output_stem}_matrix_summary.csv"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
