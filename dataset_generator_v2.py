"""
Stratified dataset generator for neural-network-based thin-film inversion.

Generates 500k samples using a mix of uniform and stratified sampling
to ensure physically interesting regions are well-represented.
"""

import numpy as np
import time

from tmm_simulator import simulate_reflectance_batch

# Fixed wavelength grid shared by all samples
WAVELENGTHS = np.linspace(400, 800, 200)

# Sampling ranges for film parameters
THICKNESS_RANGE = (10.0, 300.0)   # nm
N_RANGE = (1.3, 2.5)             # refractive index
K_RANGE = (0.0, 0.5)             # extinction coefficient


def generate_dataset(seed=42):
    """
    Generate a 500k stratified dataset of reflectance spectra.

    Sampling strategy:
      - 60% (300k): fully uniform random across all parameters
      - 20% (100k): thickness-stratified (10 bins, 10k each)
      - 20% (100k): n-stratified (10 bins, 10k each)

    Returns
    -------
    X : np.ndarray, shape (500000, 200)
    y : np.ndarray, shape (500000, 3)
    """
    rng = np.random.default_rng(seed)

    # --- Group 1: 300k fully uniform ---
    n1 = 300_000
    t1 = rng.uniform(*THICKNESS_RANGE, size=n1)
    n1_vals = rng.uniform(*N_RANGE, size=n1)
    k1 = rng.uniform(*K_RANGE, size=n1)

    # --- Group 2: 100k thickness-stratified ---
    n2_per_bin = 10_000
    n_bins = 10
    t_edges = np.linspace(*THICKNESS_RANGE, n_bins + 1)
    t2_parts, n2_parts, k2_parts = [], [], []
    for i in range(n_bins):
        t2_parts.append(rng.uniform(t_edges[i], t_edges[i + 1], size=n2_per_bin))
        n2_parts.append(rng.uniform(*N_RANGE, size=n2_per_bin))
        k2_parts.append(rng.uniform(*K_RANGE, size=n2_per_bin))
    t2 = np.concatenate(t2_parts)
    n2_vals = np.concatenate(n2_parts)
    k2 = np.concatenate(k2_parts)

    # --- Group 3: 100k n-stratified ---
    n3_per_bin = 10_000
    n_edges = np.linspace(*N_RANGE, n_bins + 1)
    t3_parts, n3_parts, k3_parts = [], [], []
    for i in range(n_bins):
        t3_parts.append(rng.uniform(*THICKNESS_RANGE, size=n3_per_bin))
        n3_parts.append(rng.uniform(n_edges[i], n_edges[i + 1], size=n3_per_bin))
        k3_parts.append(rng.uniform(*K_RANGE, size=n3_per_bin))
    t3 = np.concatenate(t3_parts)
    n3_vals = np.concatenate(n3_parts)
    k3 = np.concatenate(k3_parts)

    # --- Combine all groups ---
    thicknesses = np.concatenate([t1, t2, t3])
    n_values = np.concatenate([n1_vals, n2_vals, n3_vals])
    k_values = np.concatenate([k1, k2, k3])

    y = np.column_stack([thicknesses, n_values, k_values]).astype(np.float32)

    # Simulate all spectra in one vectorized call
    print(f"Simulating {len(y):,} spectra...")
    t0 = time.time()
    X = simulate_reflectance_batch(thicknesses, n_values, k_values,
                                   WAVELENGTHS).astype(np.float32)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Shuffle everything together
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y


if __name__ == "__main__":
    X, y = generate_dataset()

    # Save to disk
    np.savez("dataset_v2.npz", X=X, y=y)
    print(f"\nSaved dataset_v2.npz")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Print parameter ranges
    labels = ["thickness (nm)", "n", "k"]
    for col, name in enumerate(labels):
        print(f"  {name}: min={y[:, col].min():.4f}, max={y[:, col].max():.4f}")

    # Histogram summary to confirm stratification
    print("\nThickness bin counts:")
    t_edges = np.linspace(10, 300, 11)
    counts, _ = np.histogram(y[:, 0], bins=t_edges)
    for i in range(len(counts)):
        bar = "#" * (counts[i] // 1000)
        print(f"  [{t_edges[i]:6.1f}, {t_edges[i+1]:6.1f}): {counts[i]:6d}  {bar}")

    print("\nn bin counts:")
    n_edges = np.linspace(1.3, 2.5, 11)
    counts, _ = np.histogram(y[:, 1], bins=n_edges)
    for i in range(len(counts)):
        bar = "#" * (counts[i] // 1000)
        print(f"  [{n_edges[i]:.2f}, {n_edges[i+1]:.2f}): {counts[i]:6d}  {bar}")
