from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import artifact_path, ensure_parent_dir
from src.reliability_training import sample_curriculum_params
from src.tmm_simulator import simulate_reflectance_realistic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V5 realistic synthetic dataset.")
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--native-oxide-nm", type=float, default=1.5)
    parser.add_argument("--film-n-slope", type=float, default=0.0)
    parser.add_argument("--film-k-slope", type=float, default=0.0)
    parser.add_argument("--output", default="dataset_v5_realistic.npz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = sample_curriculum_params(args.n_samples, seed=args.seed)
    wavelengths = np.linspace(400, 800, 200, dtype=np.float64)
    x = np.empty((args.n_samples, len(wavelengths)), dtype=np.float32)

    for i in tqdm(range(args.n_samples), desc="Generating realistic spectra"):
        t, n, k = params[i]
        x[i] = simulate_reflectance_realistic(
            thickness_nm=float(t),
            n=float(n),
            k=float(k),
            wavelengths_nm=wavelengths,
            native_oxide_nm=float(args.native_oxide_nm),
            oxide_n=1.46,
            oxide_k=0.0,
            use_dispersive_si=True,
            film_n_slope=float(args.film_n_slope),
            film_k_slope=float(args.film_k_slope),
        ).astype(np.float32)

    out = ensure_parent_dir(artifact_path("data", args.output))
    np.savez(
        out,
        X=x,
        y=params.astype(np.float32),
        wavelengths=wavelengths.astype(np.float32),
        native_oxide_nm=np.float32(args.native_oxide_nm),
        film_n_slope=np.float32(args.film_n_slope),
        film_k_slope=np.float32(args.film_k_slope),
        substrate_model=np.array("KLA_Si_dispersive", dtype=object),
    )
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
