"""
Transfer Matrix Method (TMM) simulator for thin-film reflectance.

Computes normal-incidence reflectance spectra for a single thin film
on a silicon substrate using the characteristic matrix formulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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

    All intermediate arrays have shape (B, W) via broadcasting, with no Python
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

    # Characteristic matrix elements, all (B, W)
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
    Differentiable TMM in pure PyTorch. Gradients flow through all inputs.

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

    # Characteristic matrix elements, all (B, W) complex
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


def simulate_reflectance_torch_fast(thicknesses, n_vals, k_vals, wavelengths):
    """
    Faster differentiable TMM for training.

    This version uses float32 and complex64 instead of float64 and complex128.
    It is usually much faster on consumer GPUs while keeping useful precision
    for neural network training losses.
    """
    import torch

    device = thicknesses.device

    n_air = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    n_si = torch.tensor(3.88 + 0.02j, dtype=torch.complex64, device=device)

    n_film = torch.complex(n_vals.float(), k_vals.float()).unsqueeze(1)
    d = thicknesses.float().unsqueeze(1)
    wl = wavelengths.float().unsqueeze(0)

    delta = (2.0 * torch.pi / wl) * n_film * d

    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)

    m11 = cos_d
    m12 = -1j * sin_d / n_film
    m21 = -1j * n_film * sin_d
    m22 = cos_d

    numerator = (m11 + m12 * n_si) * n_air - (m21 + m22 * n_si)
    denominator = (m11 + m12 * n_si) * n_air + (m21 + m22 * n_si)
    r = numerator / denominator

    return torch.abs(r) ** 2


_SI_TABLE_CACHE = None


def load_si_nk_table(table_path=None):
    """
    Load wavelength-dependent Si n and k values from TSV file.
    """
    global _SI_TABLE_CACHE
    if _SI_TABLE_CACHE is not None and table_path is None:
        return _SI_TABLE_CACHE

    if table_path is None:
        table_path = Path(__file__).resolve().parent / "material_data" / "si_nk_kla_public.tsv"

    data = np.genfromtxt(table_path, delimiter="\t", names=True, dtype=float)
    wl = np.asarray(data["wavelength_nm"], dtype=float)
    n_vals = np.asarray(data["n"], dtype=float)
    k_vals = np.asarray(data["k"], dtype=float)

    order = np.argsort(wl)
    wl = wl[order]
    n_vals = n_vals[order]
    k_vals = k_vals[order]

    if table_path is None or str(table_path).endswith("si_nk_kla_public.tsv"):
        _SI_TABLE_CACHE = (wl, n_vals, k_vals)
    return wl, n_vals, k_vals


def interpolate_si_nk(wavelengths_nm, table_path=None):
    """
    Interpolate Si substrate n and k at requested wavelengths.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    wl_table, n_table, k_table = load_si_nk_table(table_path=table_path)
    n_interp = np.interp(wavelengths, wl_table, n_table)
    k_interp = np.interp(wavelengths, wl_table, k_table)
    return n_interp, k_interp


def film_dispersion_simple(wavelengths_nm, n0, k0, n_slope=0.0, k_slope=0.0, pivot_nm=600.0):
    """
    Simple linear film dispersion model around pivot wavelength.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    delta = (wavelengths - float(pivot_nm)) / float(pivot_nm)
    n_vals = np.asarray(n0, dtype=float) + float(n_slope) * delta
    k_vals = np.asarray(k0, dtype=float) + float(k_slope) * delta
    return n_vals, np.clip(k_vals, 0.0, None)


def _reflectance_from_stack_normal_incidence(n_layers_complex, d_layers_nm, wavelengths_nm, n_incident=1.0 + 0.0j):
    """
    Generic normal-incidence TMM for stack:
    incident | layer1 ... layerN | substrate

    n_layers_complex is a list where each element is:
      - layer array shape (W,) for layers and final substrate, or
      - scalar complex value.
    d_layers_nm contains thicknesses only for physical layers (not substrate).
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    num_w = len(wavelengths)
    d_layers = np.asarray(d_layers_nm, dtype=float)
    reflectance = np.empty(num_w, dtype=float)

    layer_arrays = []
    for entry in n_layers_complex:
        arr = np.asarray(entry)
        if arr.ndim == 0:
            arr = np.full(num_w, complex(arr), dtype=np.complex128)
        else:
            arr = arr.astype(np.complex128)
            if len(arr) != num_w:
                raise ValueError("Layer array length must match wavelengths.")
        layer_arrays.append(arr)

    if len(layer_arrays) != len(d_layers) + 1:
        raise ValueError("Expected one substrate entry in n_layers_complex beyond physical layers.")

    n0 = complex(n_incident)
    for wi, wl in enumerate(wavelengths):
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j

        for li, d_nm in enumerate(d_layers):
            n_layer = layer_arrays[li][wi]
            delta = (2.0 * np.pi / wl) * n_layer * d_nm
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            a11 = cos_d
            a12 = -1j * sin_d / n_layer
            a21 = -1j * n_layer * sin_d
            a22 = cos_d

            b11 = m11 * a11 + m12 * a21
            b12 = m11 * a12 + m12 * a22
            b21 = m21 * a11 + m22 * a21
            b22 = m21 * a12 + m22 * a22
            m11, m12, m21, m22 = b11, b12, b21, b22

        n_sub = layer_arrays[-1][wi]
        numerator = (m11 + m12 * n_sub) * n0 - (m21 + m22 * n_sub)
        denominator = (m11 + m12 * n_sub) * n0 + (m21 + m22 * n_sub)
        r = numerator / denominator
        reflectance[wi] = np.abs(r) ** 2
    return reflectance


def simulate_reflectance_realistic(
    thickness_nm,
    n,
    k,
    wavelengths_nm,
    native_oxide_nm=0.0,
    oxide_n=1.46,
    oxide_k=0.0,
    use_dispersive_si=True,
    si_table_path=None,
    film_n_slope=0.0,
    film_k_slope=0.0,
):
    """
    Realistic forward model:
      air | optional native SiO2 | target film | Si substrate with n(λ),k(λ)

    Legacy compatibility:
      If use_dispersive_si=False and native_oxide_nm=0 with zero slopes,
      this reduces to the same assumptions as simulate_reflectance().
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if use_dispersive_si:
        n_si, k_si = interpolate_si_nk(wavelengths, table_path=si_table_path)
        n_sub = n_si + 1j * k_si
    else:
        n_sub = np.full(len(wavelengths), 3.88 + 1j * 0.02, dtype=np.complex128)

    n_film_vals, k_film_vals = film_dispersion_simple(
        wavelengths,
        n0=float(n),
        k0=float(k),
        n_slope=float(film_n_slope),
        k_slope=float(film_k_slope),
    )
    n_film = n_film_vals + 1j * k_film_vals

    layers_n = []
    layers_d = []
    if float(native_oxide_nm) > 0.0:
        oxide_arr = np.full(len(wavelengths), complex(float(oxide_n), float(oxide_k)), dtype=np.complex128)
        layers_n.append(oxide_arr)
        layers_d.append(float(native_oxide_nm))

    layers_n.append(n_film.astype(np.complex128))
    layers_d.append(float(thickness_nm))
    layers_n.append(n_sub.astype(np.complex128))

    return _reflectance_from_stack_normal_incidence(
        layers_n,
        layers_d,
        wavelengths,
        n_incident=1.0 + 0.0j,
    )


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
