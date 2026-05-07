from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import artifact_path, ensure_parent_dir
from src.reliability_training import sample_curriculum_params
from src.tmm_simulator import simulate_reflectance_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate curriculum dataset for reliability training.")
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="dataset_v5_curriculum.npz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hard_cases = artifact_path(
        "benchmarks",
        "reliability_matrix_smoke_nn_only",
        "reliability_matrix_smoke_nn_only_top_failures.csv",
    )
    params = sample_curriculum_params(args.n_samples, seed=args.seed, hard_cases_csv=hard_cases)
    wavelengths = np.linspace(400, 800, 200, dtype=np.float64)
    spectra = simulate_reflectance_batch(
        params[:, 0].astype(np.float64),
        params[:, 1].astype(np.float64),
        params[:, 2].astype(np.float64),
        wavelengths,
    ).astype(np.float32)
    out = ensure_parent_dir(artifact_path("data", args.output))
    np.savez(out, X=spectra, y=params.astype(np.float32))
    print(f"Saved {out}")
    print(f"X shape: {spectra.shape}, y shape: {params.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
