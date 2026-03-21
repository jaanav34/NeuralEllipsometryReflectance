"""
Synthetic dataset generator for neural-network-based thin-film inversion.

Generates random (thickness, n, k) film parameters, simulates their
reflectance spectra via the TMM, and saves the input-output pairs
as a .npz file suitable for supervised training.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tmm_simulator import simulate_reflectance

# Fixed wavelength grid shared by all samples
WAVELENGTHS = np.linspace(400, 800, 200)

# Sampling ranges for film parameters
THICKNESS_RANGE = (10.0, 300.0)   # nm
N_RANGE = (1.3, 2.5)             # refractive index
K_RANGE = (0.0, 0.5)             # extinction coefficient


def generate_dataset(num_samples=100_000, seed=42):
    """
    Generate a synthetic dataset of reflectance spectra and film parameters.

    Parameters
    ----------
    num_samples : int
        Number of (spectrum, parameter) pairs to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (num_samples, 200)
        Reflectance spectra (one per row).
    y : np.ndarray, shape (num_samples, 3)
        Film parameters [thickness_nm, n, k] for each spectrum.
    """
    rng = np.random.default_rng(seed)

    # Sample film parameters uniformly within specified ranges
    thicknesses = rng.uniform(*THICKNESS_RANGE, size=num_samples)
    n_values = rng.uniform(*N_RANGE, size=num_samples)
    k_values = rng.uniform(*K_RANGE, size=num_samples)

    # Stack parameters into label array: each row is [thickness, n, k]
    y = np.column_stack([thicknesses, n_values, k_values])

    # Compute reflectance spectrum for each sample
    X = np.empty((num_samples, len(WAVELENGTHS)))
    for i in tqdm(range(num_samples), desc="Generating spectra"):
        X[i] = simulate_reflectance(thicknesses[i], n_values[i], k_values[i],
                                    WAVELENGTHS)

    return X, y


if __name__ == "__main__":
    # Generate the full dataset
    X, y = generate_dataset(num_samples=100_000)

    # Save to disk
    np.savez("dataset.npz", X=X, y=y)
    print(f"\nSaved dataset.npz")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Print parameter ranges to confirm correct sampling
    labels = ["thickness (nm)", "n", "k"]
    for col, name in enumerate(labels):
        print(f"  {name}: min={y[:, col].min():.4f}, max={y[:, col].max():.4f}")

    # Plot 3 random spectra as a visual sanity check
    rng = np.random.default_rng(0)
    indices = rng.choice(len(X), size=3, replace=False)

    plt.figure(figsize=(8, 5))
    for idx in indices:
        t, n, k = y[idx]
        label = f"d={t:.1f}nm, n={n:.2f}, k={k:.3f}"
        plt.plot(WAVELENGTHS, X[idx], linewidth=1.5, label=label)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Sample Reflectance Spectra from Generated Dataset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
