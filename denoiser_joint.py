"""
Physics-aware joint denoiser: trains a DenoisingAutoencoder with a frozen
SpectraNet V4 inverter in the loop. The denoiser learns to produce outputs
that are not only close to the clean spectrum (reconstruction loss) but also
lead to accurate parameter predictions (inversion loss).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from denoiser import DenoisingAutoencoder
from train import SpectraNet
from tmm_simulator import simulate_reflectance_batch

if __name__ == "__main__":
    WAVELENGTHS = np.linspace(400, 800, 200).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ──────────────────────────────────────────────
    # 1. Load denoiser (warm start from denoiser.pt)
    # ──────────────────────────────────────────────

    denoiser = DenoisingAutoencoder().to(device)
    denoiser.load_state_dict(torch.load("denoiser.pt", weights_only=True,
                                        map_location=device))
    print(f"Denoiser parameters: "
          f"{sum(p.numel() for p in denoiser.parameters()):,}")

    # ──────────────────────────────────────────────
    # 2. Load frozen SpectraNet V4
    # ──────────────────────────────────────────────

    model_v4 = SpectraNet().to(device)
    model_v4.load_state_dict(torch.load("spectranet_v4.pt", weights_only=True,
                                        map_location=device))
    model_v4.eval()
    for p in model_v4.parameters():
        p.requires_grad = False

    norm_data = np.load("spectra_norm_v4.npz")
    X_mean_np = norm_data["mean"].astype(np.float32)
    X_std_np = norm_data["std"].astype(np.float32)
    X_mean_t = torch.from_numpy(X_mean_np).to(device)
    X_std_t = torch.from_numpy(X_std_np).to(device)

    PARAM_BOUNDS = np.array([[10.0, 300.0], [1.3, 2.5], [0.0, 0.5]],
                            dtype=np.float32)
    param_min = PARAM_BOUNDS[:, 0]
    param_range = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    param_min_t = torch.from_numpy(param_min).to(device)
    param_range_t = torch.from_numpy(param_range).to(device)

    # ──────────────────────────────────────────────
    # 3. Data loading
    # ──────────────────────────────────────────────

    data = np.load("dataset_v2.npz")
    X_all = data["X"].astype(np.float32)
    y_all = data["y"].astype(np.float32)

    # Normalize y to [0,1] for MSE against SpectraNet's sigmoid output
    y_all_norm = (y_all - param_min) / param_range

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_all))
    n_train = int(0.8 * len(X_all))
    n_val = int(0.1 * len(X_all))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    X_train, y_train = X_all[train_idx], y_all_norm[train_idx]
    X_val, y_val = X_all[val_idx], y_all_norm[val_idx]

    BATCH_SIZE = 1024

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val),
                           torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            num_workers=4, pin_memory=True)

    # ──────────────────────────────────────────────
    # 4. Training
    # ──────────────────────────────────────────────

    LAMBDA_INVERSION = 2.0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5
    )

    MAX_EPOCHS = 60
    EARLY_STOP_PATIENCE = 12

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # --- Train ---
        denoiser.train()
        running_loss = 0.0
        for clean_batch, y_batch in train_loader:
            clean_batch = clean_batch.to(device)
            y_batch = y_batch.to(device)

            # Inject noise
            noise_std = torch.empty(1, device=device).uniform_(0.002, 0.025)
            noisy = (clean_batch +
                     torch.randn_like(clean_batch) * noise_std).clamp(0.0, 1.0)

            # Denoise
            denoised = denoiser(noisy)

            # Reconstruction loss
            recon_loss = criterion(denoised, clean_batch)

            # Inversion loss: pass denoised through frozen SpectraNet
            denoised_norm = (denoised - X_mean_t) / X_std_t
            pred_params = model_v4(denoised_norm)
            inversion_loss = criterion(pred_params, y_batch)

            loss = recon_loss + LAMBDA_INVERSION * inversion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(clean_batch)
        train_loss = running_loss / len(train_ds)

        # --- Validate ---
        denoiser.eval()
        running_loss = 0.0
        with torch.no_grad():
            for clean_batch, y_batch in val_loader:
                clean_batch = clean_batch.to(device)
                y_batch = y_batch.to(device)

                noisy = (clean_batch +
                         torch.randn_like(clean_batch) * 0.010).clamp(0.0, 1.0)
                denoised = denoiser(noisy)

                recon_loss = criterion(denoised, clean_batch)
                denoised_norm = (denoised - X_mean_t) / X_std_t
                pred_params = model_v4(denoised_norm)
                inversion_loss = criterion(pred_params, y_batch)

                loss = recon_loss + LAMBDA_INVERSION * inversion_loss
                running_loss += loss.item() * len(clean_batch)
        val_loss = running_loss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(denoiser.state_dict(), "denoiser_joint.pt")
        else:
            epochs_without_improvement += 1

        epoch_time = time.time() - t0
        if epoch % 5 == 0 or epoch == 1 or epochs_without_improvement == 0:
            print(f"Epoch {epoch:3d} | train {train_loss:.6f} | "
                  f"val {val_loss:.6f} | lr {current_lr:.1e} | "
                  f"{epoch_time:.1f}s"
                  f"{' *' if epochs_without_improvement == 0 else ''}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # ──────────────────────────────────────────────
    # 5. OOD Evaluation — four-way comparison
    # ──────────────────────────────────────────────

    # Generate 1000 fresh OOD spectra
    ood_rng = np.random.default_rng(999)
    ood_thick = ood_rng.uniform(10, 300, 1000).astype(np.float32)
    ood_n = ood_rng.uniform(1.3, 2.5, 1000).astype(np.float32)
    ood_k = ood_rng.uniform(0.0, 0.5, 1000).astype(np.float32)
    ood_spectra = simulate_reflectance_batch(
        ood_thick, ood_n, ood_k, WAVELENGTHS
    ).astype(np.float32)
    ood_params = np.stack([ood_thick, ood_n, ood_k], axis=1)

    ood_t = torch.from_numpy(ood_spectra).to(device)

    # Add noise at std=0.010
    torch.manual_seed(77)
    ood_noisy_t = (ood_t + torch.randn_like(ood_t) * 0.010).clamp(0.0, 1.0)

    # Standard denoiser
    dae_std = DenoisingAutoencoder().to(device)
    dae_std.load_state_dict(torch.load("denoiser.pt", weights_only=True,
                                       map_location=device))
    dae_std.eval()

    # Joint denoiser
    dae_joint = DenoisingAutoencoder().to(device)
    dae_joint.load_state_dict(torch.load("denoiser_joint.pt", weights_only=True,
                                         map_location=device))
    dae_joint.eval()

    with torch.no_grad():
        denoised_std_t = dae_std(ood_noisy_t)
        denoised_joint_t = dae_joint(ood_noisy_t)

    # Run SpectraNet V4 on all four versions
    def run_v4(spectra_t):
        normed = (spectra_t - X_mean_t) / X_std_t
        with torch.no_grad():
            pred_norm = model_v4(normed).cpu().numpy()
        return pred_norm * param_range + param_min

    pred_clean = run_v4(ood_t)
    pred_noisy = run_v4(ood_noisy_t)
    pred_std = run_v4(denoised_std_t)
    pred_joint = run_v4(denoised_joint_t)

    mae_clean = np.mean(np.abs(pred_clean - ood_params), axis=0)
    mae_noisy = np.mean(np.abs(pred_noisy - ood_params), axis=0)
    mae_std = np.mean(np.abs(pred_std - ood_params), axis=0)
    mae_joint = np.mean(np.abs(pred_joint - ood_params), axis=0)

    labels = ["thickness (nm)", "n", "k"]

    print("\n" + "=" * 82)
    print("SPECTRANET V4 — FOUR-WAY COMPARISON (OOD, noise std=0.010)")
    print("=" * 82)
    print(f"  {'Parameter':20s} {'Clean':>10s} {'Noisy':>10s} "
          f"{'Std Denoiser':>14s} {'Joint Denoiser':>16s}")
    print("-" * 82)
    for i, name in enumerate(labels):
        print(f"  {name:20s} {mae_clean[i]:10.4f} {mae_noisy[i]:10.4f} "
              f"{mae_std[i]:14.4f} {mae_joint[i]:16.4f}")
    print("-" * 82)
    norm_clean = (mae_clean / param_range).mean()
    norm_noisy = (mae_noisy / param_range).mean()
    norm_std = (mae_std / param_range).mean()
    norm_joint = (mae_joint / param_range).mean()
    print(f"  {'Norm. avg MAE':20s} {norm_clean:10.4f} {norm_noisy:10.4f} "
          f"{norm_std:14.4f} {norm_joint:16.4f}")
    print("=" * 82)
