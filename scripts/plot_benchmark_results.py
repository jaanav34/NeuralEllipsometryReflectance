from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paths import artifact_path, ensure_parent_dir


def parse_args():
    p = argparse.ArgumentParser(description="Plot heatmaps from benchmark NPZ output.")
    p.add_argument("npz", help="Path to benchmark .npz file")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--stem", default=None)
    return p.parse_args()


def heatmap_thickness_n(true_params, values, title, out_path, k_max=None):
    mask = np.isfinite(values)
    if k_max is not None:
        mask &= true_params[:, 2] <= k_max
    t = true_params[mask, 0]
    n = true_params[mask, 1]
    v = values[mask]
    t_edges = np.linspace(10, 300, 41)
    n_edges = np.linspace(1.3, 2.5, 41)
    grid = np.full((len(t_edges) - 1, len(n_edges) - 1), np.nan)
    for i in range(len(t_edges) - 1):
        for j in range(len(n_edges) - 1):
            m = (t >= t_edges[i]) & (t < t_edges[i + 1]) & (n >= n_edges[j]) & (n < n_edges[j + 1])
            if np.any(m):
                grid[i, j] = np.nanmedian(v[m])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=[t_edges[0], t_edges[-1], n_edges[0], n_edges[-1]],
    )
    ax.set_xlabel("True thickness (nm)")
    ax.set_ylabel("True n")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="median value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    data = np.load(args.npz)
    true_params = data["true_params"]
    nn_mean = data["nn_mean"]
    refined = data["refined_params"]
    stem = args.stem or Path(args.npz).stem
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.npz).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_abs = np.abs(nn_mean - true_params)
    norm_err = (nn_abs / np.array([290.0, 1.2, 0.5], dtype=np.float32)).mean(axis=1)
    heatmap_thickness_n(true_params, norm_err, "NN normalized parameter error", out_dir / f"{stem}_nn_norm_error_heatmap.png")
    heatmap_thickness_n(true_params, nn_abs[:, 1], "NN absolute n error", out_dir / f"{stem}_nn_n_error_heatmap.png")
    heatmap_thickness_n(true_params, nn_abs[:, 2], "NN absolute k error", out_dir / f"{stem}_nn_k_error_heatmap.png")

    if np.isfinite(refined[:, 0]).any():
        ref_abs = np.abs(refined - true_params)
        ref_norm = (ref_abs / np.array([290.0, 1.2, 0.5], dtype=np.float32)).mean(axis=1)
        heatmap_thickness_n(true_params, ref_norm, "Refined normalized parameter error", out_dir / f"{stem}_ref_norm_error_heatmap.png")

    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
