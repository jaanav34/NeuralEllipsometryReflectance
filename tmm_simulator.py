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
