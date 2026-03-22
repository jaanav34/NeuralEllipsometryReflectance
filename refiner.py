"""
Spectral residual closer: takes a neural network prediction as initial
guess and refines it by minimizing the MSE between the re-simulated
spectrum and the input spectrum using L-BFGS-B.
"""

import numpy as np
from scipy.optimize import minimize
import time

from tmm_simulator import simulate_reflectance, simulate_reflectance_batch

# Default wavelength grid (must match training)
DEFAULT_WAVELENGTHS = np.linspace(400, 800, 200)

# Physical bounds (same as training)
PARAM_BOUNDS_LIST = [(10.0, 300.0), (1.3, 2.5), (0.0, 0.5)]


def refine_prediction(spectrum, init_thickness, init_n, init_k,
                      wavelengths=None):
    """
    Refine a neural network prediction by minimizing spectral residual.

    Uses scipy.optimize.minimize with the L-BFGS-B method and
    box constraints matching the training parameter bounds:
        thickness: [10, 300] nm
        n: [1.3, 2.5]
        k: [0.0, 0.5]

    The objective function is MSE between simulate_reflectance()
    output and the input spectrum.

    Parameters
    ----------
    spectrum : np.ndarray shape (200,)
        The input reflectance spectrum to fit (raw, unnormalized).
    init_thickness, init_n, init_k : float
        Initial guess from the neural network.
    wavelengths : np.ndarray, optional
        Defaults to np.linspace(400, 800, 200).

    Returns
    -------
    result : dict with keys:
        thickness, n, k : float — refined parameters
        success : bool — whether optimizer converged
        n_iterations : int — number of optimizer iterations
        init_residual : float — spectral MSE before refinement
        final_residual : float — spectral MSE after refinement
        improvement : float — percentage improvement in residual
    """
    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS

    spectrum = np.asarray(spectrum, dtype=np.float64)

    # Compute initial residual
    init_sim = simulate_reflectance(init_thickness, init_n, init_k,
                                    wavelengths)
    init_residual = float(np.mean((spectrum - init_sim) ** 2))

    # Objective: MSE between simulated and input spectrum
    def objective(params):
        t, n, k = params
        sim = simulate_reflectance(t, n, k, wavelengths)
        return float(np.mean((spectrum - sim) ** 2))

    x0 = np.array([init_thickness, init_n, init_k], dtype=np.float64)

    res = minimize(objective, x0, method="L-BFGS-B",
                   bounds=PARAM_BOUNDS_LIST,
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})

    final_residual = float(res.fun)
    if init_residual > 0:
        improvement = (1.0 - final_residual / init_residual) * 100.0
    else:
        improvement = 0.0

    return {
        "thickness": float(res.x[0]),
        "n": float(res.x[1]),
        "k": float(res.x[2]),
        "success": bool(res.success),
        "n_iterations": int(res.nit),
        "init_residual": init_residual,
        "final_residual": final_residual,
        "improvement": improvement,
    }


def benchmark_refiner(n_samples=500, seed=42):
    """
    Compare NN-only vs NN+refiner on random test samples.

    Generates n_samples random (thickness, n, k) values,
    simulates their spectra, runs NN prediction, then runs
    refiner on each, and reports:
    - MAE for thickness, n, k: NN-only vs NN+refiner
    - Average spectral residual: before vs after
    - Average number of optimizer iterations
    - Average time per sample: NN-only vs NN+refiner

    Loads model and norm stats internally.
    Prints a clean comparison table.
    """
    import torch
    from train import SpectraNet

    # Load model
    model = SpectraNet()
    model.load_state_dict(torch.load("spectranet_v4.pt",
                                     map_location="cpu",
                                     weights_only=True))
    model.eval()

    # Load norm stats
    norm = np.load("spectra_norm_v4.npz")
    X_mean, X_std = norm["mean"], norm["std"]

    PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float32)
    PARAM_RANGE = np.array([290.0, 1.2, 0.5], dtype=np.float32)

    wavelengths = DEFAULT_WAVELENGTHS

    # Generate random test samples
    rng = np.random.default_rng(seed)
    true_thick = rng.uniform(10, 300, n_samples)
    true_n = rng.uniform(1.3, 2.5, n_samples)
    true_k = rng.uniform(0.0, 0.5, n_samples)

    # Simulate all spectra at once
    spectra = simulate_reflectance_batch(true_thick, true_n, true_k,
                                         wavelengths).astype(np.float32)

    # NN prediction (batched)
    X_norm = (spectra - X_mean) / X_std
    t0_nn = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_norm)).numpy()
    nn_time = time.perf_counter() - t0_nn
    nn_pred = pred_norm * PARAM_RANGE + PARAM_MIN

    # Refiner (sample by sample)
    refined_pred = np.empty((n_samples, 3))
    total_iters = 0
    total_residual_before = 0.0
    total_residual_after = 0.0

    t0_ref = time.perf_counter()
    for i in range(n_samples):
        res = refine_prediction(spectra[i],
                                float(nn_pred[i, 0]),
                                float(nn_pred[i, 1]),
                                float(nn_pred[i, 2]),
                                wavelengths)
        refined_pred[i] = [res["thickness"], res["n"], res["k"]]
        total_iters += res["n_iterations"]
        total_residual_before += res["init_residual"]
        total_residual_after += res["final_residual"]
    refiner_time = time.perf_counter() - t0_ref

    true_params = np.stack([true_thick, true_n, true_k], axis=1)

    # MAE
    nn_mae = np.mean(np.abs(nn_pred - true_params), axis=0)
    ref_mae = np.mean(np.abs(refined_pred - true_params), axis=0)

    labels = ["thickness (nm)", "n", "k"]

    print("\n" + "=" * 65)
    print(f"BENCHMARK: NN-only vs NN+Refiner ({n_samples} samples)")
    print("=" * 65)
    print(f"  {'Parameter':20s} {'NN MAE':>10s} {'NN+Ref MAE':>12s} "
          f"{'Improvement':>12s}")
    print("-" * 65)
    for i, name in enumerate(labels):
        imp = (1.0 - ref_mae[i] / nn_mae[i]) * 100.0 if nn_mae[i] > 0 else 0
        print(f"  {name:20s} {nn_mae[i]:10.4f} {ref_mae[i]:12.4f} "
              f"{imp:11.1f}%")
    print("-" * 65)
    print(f"  {'Avg spectral MSE':20s} "
          f"{total_residual_before / n_samples:10.2e} "
          f"{total_residual_after / n_samples:12.2e}")
    print(f"  {'Avg iterations':20s} {'':>10s} "
          f"{total_iters / n_samples:12.1f}")
    print(f"  {'Time per sample':20s} "
          f"{nn_time / n_samples * 1000:9.2f} ms "
          f"{refiner_time / n_samples * 1000:11.2f} ms")
    print(f"  {'Total time':20s} "
          f"{nn_time:9.3f} s  "
          f"{refiner_time:11.3f} s")
    print("=" * 65)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from train import SpectraNet

    # --- Benchmark ---
    benchmark_refiner(n_samples=500)

    # --- Detailed single example ---
    print("\n" + "=" * 65)
    print("DETAILED EXAMPLE")
    print("=" * 65)

    true_t, true_n_val, true_k_val = 147.3, 1.85, 0.12
    wavelengths = DEFAULT_WAVELENGTHS
    spectrum = simulate_reflectance(true_t, true_n_val, true_k_val,
                                    wavelengths).astype(np.float32)

    # NN prediction
    model = SpectraNet()
    model.load_state_dict(torch.load("spectranet_v4.pt",
                                     map_location="cpu",
                                     weights_only=True))
    model.eval()
    norm = np.load("spectra_norm_v4.npz")
    X_mean, X_std = norm["mean"], norm["std"]
    PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float32)
    PARAM_RANGE = np.array([290.0, 1.2, 0.5], dtype=np.float32)

    x_norm = (spectrum - X_mean) / X_std
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_norm[np.newaxis, :])).numpy()[0]
    nn_pred = pred_norm * PARAM_RANGE + PARAM_MIN

    # Refine
    res = refine_prediction(spectrum, float(nn_pred[0]), float(nn_pred[1]),
                            float(nn_pred[2]), wavelengths)

    print(f"  {'':15s} {'True':>10s} {'NN':>10s} {'NN+Refiner':>12s}")
    print("-" * 50)
    print(f"  {'thickness (nm)':15s} {true_t:10.2f} {nn_pred[0]:10.2f} "
          f"{res['thickness']:12.2f}")
    print(f"  {'n':15s} {true_n_val:10.4f} {nn_pred[1]:10.4f} "
          f"{res['n']:12.4f}")
    print(f"  {'k':15s} {true_k_val:10.4f} {nn_pred[2]:10.4f} "
          f"{res['k']:12.4f}")
    print(f"\n  Optimizer converged: {res['success']}")
    print(f"  Iterations: {res['n_iterations']}")
    print(f"  Spectral MSE: {res['init_residual']:.2e} → "
          f"{res['final_residual']:.2e} ({res['improvement']:.1f}% reduction)")

    # Plot
    nn_sim = simulate_reflectance(float(nn_pred[0]), float(nn_pred[1]),
                                  float(nn_pred[2]), wavelengths)
    ref_sim = simulate_reflectance(res["thickness"], res["n"], res["k"],
                                   wavelengths)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 1],
                                    sharex=True)
    ax1.plot(wavelengths, spectrum, "k-", linewidth=1.5, label="Input (truth)")
    ax1.plot(wavelengths, nn_sim, "--", color="#1f77b4", linewidth=1.5,
             label=f"NN (MSE={res['init_residual']:.2e})")
    ax1.plot(wavelengths, ref_sim, "--", color="#2ca02c", linewidth=1.5,
             label=f"NN+Refiner (MSE={res['final_residual']:.2e})")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(f"Refiner Example: d={true_t}nm, n={true_n_val}, "
                  f"k={true_k_val}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(wavelengths, spectrum - nn_sim, color="#1f77b4", linewidth=1,
             label="NN residual")
    ax2.plot(wavelengths, spectrum - ref_sim, color="#2ca02c", linewidth=1,
             label="Refined residual")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Residual")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("refiner_example.png", dpi=150)
    print("\nSaved refiner_example.png")
