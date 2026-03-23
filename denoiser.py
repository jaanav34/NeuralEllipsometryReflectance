"""
Denoising Autoencoder for reflectance spectra.

Trained on clean spectra from dataset_v2.npz with random Gaussian noise
injection (std uniformly sampled from [0.002, 0.025] per batch).
Works on raw reflectance values in [0, 1] — no normalization needed.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 200),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        return self.decoder(z)


if __name__ == "__main__":
    import os
    from tmm_simulator import simulate_reflectance_batch
    from train import SpectraNet

    WAVELENGTHS = np.linspace(400, 800, 200).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    print(f"Device: {device}")
    print(f"Denoiser parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ──────────────────────────────────────────────
    # 1. Training (skip if denoiser.pt exists)
    # ──────────────────────────────────────────────

    if not os.path.exists("denoiser.pt"):
        data = np.load("dataset_v2.npz")
        X_all = data["X"].astype(np.float32)

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X_all))
        n_train = int(0.8 * len(X_all))
        n_val = int(0.1 * len(X_all))

        X_train = X_all[indices[:n_train]]
        X_val = X_all[indices[n_train:n_train + n_val]]

        BATCH_SIZE = 1024
        train_ds = TensorDataset(torch.from_numpy(X_train))
        val_ds = TensorDataset(torch.from_numpy(X_val))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True,
                                  persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                num_workers=4, pin_memory=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        MAX_EPOCHS = 100
        EARLY_STOP_PATIENCE = 15

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()

            model.train()
            running_loss = 0.0
            for (clean_batch,) in train_loader:
                clean_batch = clean_batch.to(device)
                noise_std = torch.empty(1, device=device).uniform_(0.002, 0.025)
                noisy = clean_batch + torch.randn_like(clean_batch) * noise_std
                noisy = noisy.clamp(0.0, 1.0)

                denoised = model(noisy)
                loss = criterion(denoised, clean_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(clean_batch)
            train_loss = running_loss / len(train_ds)

            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for (clean_batch,) in val_loader:
                    clean_batch = clean_batch.to(device)
                    noisy = clean_batch + torch.randn_like(clean_batch) * 0.010
                    noisy = noisy.clamp(0.0, 1.0)
                    denoised = model(noisy)
                    loss = criterion(denoised, clean_batch)
                    running_loss += loss.item() * len(clean_batch)
            val_loss = running_loss / len(val_ds)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), "denoiser.pt")
            else:
                epochs_without_improvement += 1

            epoch_time = time.time() - t0
            if epoch % 10 == 0 or epoch == 1 or epochs_without_improvement == 0:
                print(f"Epoch {epoch:3d} | train {train_loss:.6f} | "
                      f"val {val_loss:.6f} | lr {current_lr:.1e} | "
                      f"{epoch_time:.1f}s"
                      f"{' *' if epochs_without_improvement == 0 else ''}")

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        print(f"\nBest validation loss: {best_val_loss:.6f}")
    else:
        print("Found existing denoiser.pt, skipping training.")

    # Load best weights for evaluation
    model.load_state_dict(torch.load("denoiser.pt", weights_only=True,
                                     map_location=device))
    model.eval()

    # ──────────────────────────────────────────────
    # 2. Out-of-distribution validation
    # ──────────────────────────────────────────────

    # Generate 1000 fresh spectra the denoiser has never seen
    ood_rng = np.random.default_rng(999)
    ood_thick = ood_rng.uniform(10, 300, 1000).astype(np.float32)
    ood_n = ood_rng.uniform(1.3, 2.5, 1000).astype(np.float32)
    ood_k = ood_rng.uniform(0.0, 0.5, 1000).astype(np.float32)
    ood_spectra = simulate_reflectance_batch(
        ood_thick, ood_n, ood_k, WAVELENGTHS
    ).astype(np.float32)
    ood_params = np.stack([ood_thick, ood_n, ood_k], axis=1)  # (1000, 3)

    ood_t = torch.from_numpy(ood_spectra).to(device)
    noise_levels = [0.005, 0.010, 0.020]

    print("\n" + "=" * 65)
    print("OOD TEST RESULTS (1000 fresh spectra, seed=999)")
    print("=" * 65)
    print(f"  {'Noise std':>12s} {'Input MSE':>12s} {'Output MSE':>12s} "
          f"{'Improvement':>12s}")
    print("-" * 65)

    torch.manual_seed(99)
    for std in noise_levels:
        noisy_t = ood_t + torch.randn_like(ood_t) * std
        noisy_t = noisy_t.clamp(0.0, 1.0)

        with torch.no_grad():
            denoised_t = model(noisy_t)

        input_mse = nn.functional.mse_loss(noisy_t, ood_t).item()
        output_mse = nn.functional.mse_loss(denoised_t, ood_t).item()
        improvement = (1.0 - output_mse / input_mse) * 100

        print(f"  {std:12.3f} {input_mse:12.6f} {output_mse:12.6f} "
              f"{improvement:11.1f}%")

    print("=" * 65)

    # ──────────────────────────────────────────────
    # 3. SpectraNet V4 inversion: clean vs noisy vs denoised
    # ──────────────────────────────────────────────

    # Load SpectraNet V4 and normalization stats
    spectranet = SpectraNet()
    spectranet.load_state_dict(torch.load("spectranet_v4.pt",
                                          map_location="cpu",
                                          weights_only=True))
    spectranet.eval()

    norm_data = np.load("spectra_norm_v4.npz")
    X_mean = norm_data["mean"]
    X_std = norm_data["std"]

    PARAM_BOUNDS = np.array([[10.0, 300.0], [1.3, 2.5], [0.0, 0.5]],
                            dtype=np.float32)
    param_min = PARAM_BOUNDS[:, 0]
    param_range = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

    def run_spectranet(spectra_np):
        """Run SpectraNet V4 on raw spectra, return physical predictions."""
        normed = (spectra_np - X_mean) / X_std
        with torch.no_grad():
            pred_norm = spectranet(
                torch.from_numpy(normed.astype(np.float32))
            ).numpy()
        return pred_norm * param_range + param_min

    # Three versions of OOD spectra at std=0.010
    torch.manual_seed(77)
    ood_noisy_t = ood_t + torch.randn_like(ood_t) * 0.010
    ood_noisy_t = ood_noisy_t.clamp(0.0, 1.0)
    ood_noisy_np = ood_noisy_t.cpu().numpy()

    with torch.no_grad():
        ood_denoised_t = model(ood_noisy_t)
    ood_denoised_np = ood_denoised_t.cpu().numpy()

    pred_clean = run_spectranet(ood_spectra)
    pred_noisy = run_spectranet(ood_noisy_np)
    pred_denoised = run_spectranet(ood_denoised_np)

    mae_clean = np.mean(np.abs(pred_clean - ood_params), axis=0)
    mae_noisy = np.mean(np.abs(pred_noisy - ood_params), axis=0)
    mae_denoised = np.mean(np.abs(pred_denoised - ood_params), axis=0)

    labels = ["thickness (nm)", "n", "k"]

    print("\n" + "=" * 70)
    print("SPECTRANET V4 — EFFECT OF DENOISER (OOD, noise std=0.010)")
    print("=" * 70)
    print(f"  {'Parameter':20s} {'Clean MAE':>12s} {'Noisy MAE':>12s} "
          f"{'Denoised MAE':>14s}")
    print("-" * 70)
    for i, name in enumerate(labels):
        print(f"  {name:20s} {mae_clean[i]:12.4f} {mae_noisy[i]:12.4f} "
              f"{mae_denoised[i]:14.4f}")
    print("-" * 70)
    # Overall: average MAE across normalized parameters for fair comparison
    norm_mae_clean = (mae_clean / param_range).mean()
    norm_mae_noisy = (mae_noisy / param_range).mean()
    norm_mae_denoised = (mae_denoised / param_range).mean()
    print(f"  {'Norm. avg MAE':20s} {norm_mae_clean:12.4f} "
          f"{norm_mae_noisy:12.4f} {norm_mae_denoised:14.4f}")
    print("=" * 70)

    # ──────────────────────────────────────────────
    # 4. Example plot at std=0.010
    # ──────────────────────────────────────────────

    torch.manual_seed(42)
    # Use OOD spectra for the example plot too
    plot_noisy_t = ood_t + torch.randn_like(ood_t) * 0.010
    plot_noisy_t = plot_noisy_t.clamp(0.0, 1.0)

    with torch.no_grad():
        plot_denoised_t = model(plot_noisy_t)

    plot_noisy_np = plot_noisy_t.cpu().numpy()
    plot_denoised_np = plot_denoised_t.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    rng_plot = np.random.default_rng(7)
    example_idxs = rng_plot.choice(1000, 3, replace=False)

    for ax, idx in zip(axes, example_idxs):
        ax.plot(WAVELENGTHS, ood_spectra[idx], "k-", label="Clean",
                linewidth=1.5)
        ax.plot(WAVELENGTHS, plot_noisy_np[idx], alpha=0.5, label="Noisy",
                linewidth=0.8)
        ax.plot(WAVELENGTHS, plot_denoised_np[idx], "--", color="#2ca02c",
                label="Denoised", linewidth=1.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title(f"OOD Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Denoiser Examples — OOD spectra (noise std = 0.010)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("denoiser_examples.png", dpi=150, bbox_inches="tight")
    print("\nSaved denoiser_examples.png")
