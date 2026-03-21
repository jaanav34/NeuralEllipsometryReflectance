"""
Transfer Matrix Method (TMM) simulator for thin-film reflectance.

Computes normal-incidence reflectance spectra for a single thin film
on a silicon substrate using the characteristic matrix formulation.
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_reflectance(thickness_nm, n, k, wavelengths_nm):
    """
    Compute reflectance spectrum of a single thin film on a silicon substrate.

    Uses the Transfer Matrix Method at normal incidence. The stack is:
        air (semi-infinite) | thin film | silicon substrate (semi-infinite)

    Parameters
    ----------
    thickness_nm : float
        Film thickness in nanometers.
    n : float
        Refractive index of the film (real part).
    k : float
        Extinction coefficient of the film (imaginary part of refractive index).
    wavelengths_nm : array_like
        Wavelengths in nanometers at which to evaluate reflectance.

    Returns
    -------
    reflectance : np.ndarray
        Reflectance values (between 0 and 1) at each wavelength.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)

    # Complex refractive indices for each medium
    n_air = 1.0 + 0.0j          # incident medium: air
    n_film = n + 1j * k          # thin film layer
    n_si = 3.88 + 1j * 0.02     # substrate: silicon (fixed)

    # Phase thickness of the film at each wavelength.
    # delta = (2 * pi / lambda) * n_film * d
    # This is the optical phase accumulated by light traversing the film once.
    delta = (2.0 * np.pi / wavelengths) * n_film * thickness_nm

    # --- Transfer Matrix Method ---
    #
    # For a single layer at normal incidence, the characteristic (transfer)
    # matrix M relates the fields at the top interface to the bottom:
    #
    #   M = | cos(delta)          -i*sin(delta)/n_film |
    #       | -i*n_film*sin(delta)    cos(delta)       |
    #
    # where delta is the phase thickness and n_film is the complex
    # refractive index of the layer.

    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    # Elements of the 2x2 characteristic matrix
    m11 = cos_d
    m12 = -1j * sin_d / n_film
    m22 = cos_d
    m21 = -1j * n_film * sin_d

    # --- Reflection coefficient ---
    #
    # For a film on a substrate, the reflection coefficient is:
    #
    #   r = (m11*n_air + m12*n_air*n_si - m21 - m22*n_si)
    #     / (m11*n_air + m12*n_air*n_si + m21 + m22*n_si)
    #
    # Derivation: matching boundary conditions at both interfaces gives
    # the total reflection in terms of the transfer matrix elements and
    # the refractive indices of the surrounding semi-infinite media.

    # Numerator:   (m11 + m12*n_si)*n_air - (m21 + m22*n_si)
    # Denominator: (m11 + m12*n_si)*n_air + (m21 + m22*n_si)
    numerator = (m11 + m12 * n_si) * n_air - (m21 + m22 * n_si)
    denominator = (m11 + m12 * n_si) * n_air + (m21 + m22 * n_si)

    r = numerator / denominator

    # Reflectance is the squared modulus of the complex reflection coefficient
    reflectance = np.abs(r) ** 2

    return reflectance


def simulate_reflectance_batch(thicknesses, n_vals, k_vals, wavelengths_nm):
    """
    Vectorized TMM: compute reflectance spectra for a batch of films at once.

    All intermediate arrays have shape (B, W) via broadcasting — no Python
    loops over batch or wavelength dimensions.

    Parameters
    ----------
    thicknesses : np.ndarray, shape (B,)
        Film thicknesses in nanometers.
    n_vals : np.ndarray, shape (B,)
        Refractive indices (real part) for each sample.
    k_vals : np.ndarray, shape (B,)
        Extinction coefficients for each sample.
    wavelengths_nm : np.ndarray, shape (W,)
        Shared wavelength grid in nanometers.

    Returns
    -------
    reflectance : np.ndarray, shape (B, W)
        Reflectance spectra for all samples.
    """
    # Ensure inputs are arrays
    thicknesses = np.asarray(thicknesses, dtype=float)
    n_vals = np.asarray(n_vals, dtype=float)
    k_vals = np.asarray(k_vals, dtype=float)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)

    # Complex refractive indices
    n_air = 1.0 + 0.0j
    n_si = 3.88 + 0.02j

    # Film refractive index: shape (B, 1) for broadcasting with (W,)
    n_film = (n_vals + 1j * k_vals)[:, np.newaxis]   # (B, 1)
    d = thicknesses[:, np.newaxis]                     # (B, 1)

    # Phase thickness: (B, 1) * (W,) → (B, W)
    delta = (2.0 * np.pi / wavelengths) * n_film * d  # (B, W)

    # Characteristic matrix elements — all (B, W)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    m11 = cos_d
    m12 = -1j * sin_d / n_film
    m21 = -1j * n_film * sin_d
    m22 = cos_d

    # Reflection coefficient
    numerator = (m11 + m12 * n_si) * n_air - (m21 + m22 * n_si)
    denominator = (m11 + m12 * n_si) * n_air + (m21 + m22 * n_si)
    r = numerator / denominator

    return np.abs(r) ** 2  # (B, W)


def simulate_reflectance_torch(thicknesses, n_vals, k_vals, wavelengths):
    """
    Differentiable TMM in pure PyTorch — gradients flow through all inputs.

    Implements the same Transfer Matrix Method as the numpy versions but
    using PyTorch complex tensors so autograd can compute d(reflectance)/d(params).

    Parameters
    ----------
    thicknesses : torch.Tensor, shape (B,)
        Film thicknesses in nanometers. May require grad.
    n_vals : torch.Tensor, shape (B,)
        Refractive indices (real part). May require grad.
    k_vals : torch.Tensor, shape (B,)
        Extinction coefficients. May require grad.
    wavelengths : torch.Tensor, shape (W,)
        Wavelength grid in nanometers.

    Returns
    -------
    reflectance : torch.Tensor, shape (B, W)
        Reflectance values, differentiable w.r.t. all inputs.
    """
    import torch

    # Fixed complex refractive indices for air and silicon substrate
    n_air = torch.tensor(1.0 + 0.0j, dtype=torch.complex128, device=thicknesses.device)
    n_si = torch.tensor(3.88 + 0.02j, dtype=torch.complex128, device=thicknesses.device)

    # Build complex film refractive index: (B,) → (B, 1)
    n_film = torch.complex(n_vals.double(), k_vals.double()).unsqueeze(1)  # (B, 1)
    d = thicknesses.double().unsqueeze(1)                                  # (B, 1)
    wl = wavelengths.double().unsqueeze(0)                                 # (1, W)

    # Phase thickness: (B, 1) * (1, W) → (B, W)
    delta = (2.0 * torch.pi / wl) * n_film * d  # (B, W)

    # Characteristic matrix elements — all (B, W) complex
    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)
    m11 = cos_d
    m12 = -1j * sin_d / n_film
    m21 = -1j * n_film * sin_d
    m22 = cos_d

    # Reflection coefficient
    numerator = (m11 + m12 * n_si) * n_air - (m21 + m22 * n_si)
    denominator = (m11 + m12 * n_si) * n_air + (m21 + m22 * n_si)
    r = numerator / denominator

    # Reflectance = |r|^2, returned as real-valued float tensor
    reflectance = torch.abs(r) ** 2

    return reflectance.float()


if __name__ == "__main__":
    # --- Demo: 100 nm SiO2 film on silicon ---

    thickness = 100.0       # nm
    n_sio2 = 1.46           # refractive index of SiO2
    k_sio2 = 0.0            # SiO2 is transparent in visible range

    wavelengths = np.linspace(400, 800, 200)  # 400-800 nm, 200 points

    R = simulate_reflectance(thickness, n_sio2, k_sio2, wavelengths)

    print(f"Min reflectance: {R.min():.4f}")
    print(f"Max reflectance: {R.max():.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, R, linewidth=2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Reflectance Spectrum: 100 nm SiO2 on Silicon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Validate batch function against loop ---
    import time

    B = 512
    rng = np.random.default_rng(0)
    t_batch = rng.uniform(10, 300, B)
    n_batch = rng.uniform(1.3, 2.5, B)
    k_batch = rng.uniform(0.0, 0.5, B)

    # Looped reference
    t0 = time.perf_counter()
    looped = np.stack([simulate_reflectance(t_batch[i], n_batch[i], k_batch[i],
                                            wavelengths) for i in range(B)])
    t_loop = time.perf_counter() - t0

    # Vectorized batch
    t0 = time.perf_counter()
    batched = simulate_reflectance_batch(t_batch, n_batch, k_batch, wavelengths)
    t_batch_time = time.perf_counter() - t0

    max_err = np.max(np.abs(looped - batched))
    print(f"\nBatch validation (B={B}):")
    print(f"  Max error vs loop: {max_err:.2e}")
    print(f"  Loop time:  {t_loop*1000:.1f} ms")
    print(f"  Batch time: {t_batch_time*1000:.1f} ms")
    print(f"  Speedup:    {t_loop/t_batch_time:.1f}x")

    # --- Validate torch implementation ---
    import torch

    B_torch = 1000
    rng2 = np.random.default_rng(99)
    t_np = rng2.uniform(10, 300, B_torch).astype(np.float64)
    n_np = rng2.uniform(1.3, 2.5, B_torch).astype(np.float64)
    k_np = rng2.uniform(0.0, 0.5, B_torch).astype(np.float64)
    wl_np = np.linspace(400, 800, 200)

    # Numpy reference
    ref = simulate_reflectance_batch(t_np, n_np, k_np, wl_np)

    # Torch CPU
    t_t = torch.tensor(t_np, requires_grad=True)
    n_t = torch.tensor(n_np, requires_grad=False)
    k_t = torch.tensor(k_np, requires_grad=False)
    wl_t = torch.tensor(wl_np)

    out_cpu = simulate_reflectance_torch(t_t, n_t, k_t, wl_t)
    max_diff = (out_cpu.detach().numpy().astype(np.float64) - ref).max()
    print(f"\nTorch vs numpy batch (B={B_torch}):")
    print(f"  Max abs difference: {abs(max_diff):.2e}")

    # Gradient check
    out_cpu.sum().backward()
    grad_ok = t_t.grad is not None and not torch.all(t_t.grad == 0).item()
    print(f"  Gradient flows to thickness: {grad_ok}")
    if t_t.grad is not None:
        print(f"  Grad mean abs: {t_t.grad.abs().mean():.6e}")

    # Timing: numpy batch vs torch CPU vs torch CUDA
    n_runs = 20

    # Numpy batch
    t0 = time.perf_counter()
    for _ in range(n_runs):
        simulate_reflectance_batch(t_np, n_np, k_np, wl_np)
    t_numpy = (time.perf_counter() - t0) / n_runs * 1000

    # Torch CPU
    t_t_cpu = torch.tensor(t_np)
    n_t_cpu = torch.tensor(n_np)
    k_t_cpu = torch.tensor(k_np)
    wl_t_cpu = torch.tensor(wl_np)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        simulate_reflectance_torch(t_t_cpu, n_t_cpu, k_t_cpu, wl_t_cpu)
    t_torch_cpu = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\nTiming (B={B_torch}, avg of {n_runs} runs):")
    print(f"  Numpy batch: {t_numpy:.1f} ms")
    print(f"  Torch CPU:   {t_torch_cpu:.1f} ms")

    # Torch CUDA (if available)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        t_t_g = torch.tensor(t_np, device=dev)
        n_t_g = torch.tensor(n_np, device=dev)
        k_t_g = torch.tensor(k_np, device=dev)
        wl_t_g = torch.tensor(wl_np, device=dev)
        # Warm up
        for _ in range(5):
            simulate_reflectance_torch(t_t_g, n_t_g, k_t_g, wl_t_g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            simulate_reflectance_torch(t_t_g, n_t_g, k_t_g, wl_t_g)
        torch.cuda.synchronize()
        t_torch_cuda = (time.perf_counter() - t0) / n_runs * 1000
        print(f"  Torch CUDA:  {t_torch_cuda:.1f} ms")
    else:
        print("  CUDA not available, skipping GPU timing")
