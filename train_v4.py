"""
Train SpectraNet V4: differentiable physics-informed loss using the
PyTorch TMM simulator. Gradients flow from the re-simulated spectrum
through the TMM back to the network weights.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

from train import SpectraNet
from tmm_simulator import simulate_reflectance_torch

if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # 1. Data loading and preprocessing
    # ──────────────────────────────────────────────

    data = np.load("dataset_v2.npz")
    X_all = data["X"].astype(np.float32)  # (500000, 200)
    y_all = data["y"].astype(np.float32)  # (500000, 3)

    WAVELENGTHS_NP = np.linspace(400, 800, 200).astype(np.float32)

    # Fixed physical bounds for parameter normalization → [0, 1]
    PARAM_BOUNDS = np.array([[10.0, 300.0],    # thickness (nm)
                             [1.3, 2.5],       # n
                             [0.0, 0.5]],      # k
                            dtype=np.float32)
    param_min = PARAM_BOUNDS[:, 0]
    param_range = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

    # Reproducible 80/10/10 split (same seed as V3)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_all))
    n_train = int(0.8 * len(X_all))
    n_val = int(0.1 * len(X_all))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
    X_val_raw, y_val = X_all[val_idx], y_all[val_idx]
    X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

    # Normalize spectra: zero mean, unit variance (fit on train only)
    X_mean = X_train_raw.mean(axis=0)
    X_std = X_train_raw.std(axis=0)
    X_std[X_std < 1e-8] = 1.0

    X_train_norm = (X_train_raw - X_mean) / X_std
    X_val_norm = (X_val_raw - X_mean) / X_std
    X_test_norm = (X_test_raw - X_mean) / X_std

    # Normalize parameters to [0, 1]
    y_train_norm = (y_train - param_min) / param_range
    y_val_norm = (y_val - param_min) / param_range
    y_test_norm = (y_test - param_min) / param_range

    # DataLoaders — train/val include raw spectra for physics loss
    BATCH_SIZE = 512

    train_ds = TensorDataset(torch.from_numpy(X_train_norm),
                             torch.from_numpy(y_train_norm),
                             torch.from_numpy(X_train_raw))
    val_ds = TensorDataset(torch.from_numpy(X_val_norm),
                           torch.from_numpy(y_val_norm),
                           torch.from_numpy(X_val_raw))
    test_ds = TensorDataset(torch.from_numpy(X_test_norm),
                            torch.from_numpy(y_test_norm))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             num_workers=4, pin_memory=True)

    # ──────────────────────────────────────────────
    # 2. Model and physics loss setup
    # ──────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraNet().to(device)
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Denormalization constants on device (for differentiable physics loss)
    t_param_min = torch.from_numpy(param_min).to(device)
    t_param_range = torch.from_numpy(param_range).to(device)
    t_wavelengths = torch.from_numpy(WAVELENGTHS_NP).to(device)

    LAMBDA_PHYSICS = 0.05
    criterion = nn.MSELoss()

    # ──────────────────────────────────────────────
    # 3. Training loop
    # ──────────────────────────────────────────────

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    MAX_EPOCHS = 300
    EARLY_STOP_PATIENCE = 30

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        for xb, yb, xb_raw in train_loader:
            xb, yb, xb_raw = xb.to(device), yb.to(device), xb_raw.to(device)

            pred_norm = model(xb)

            # Parameter MSE (normalized space)
            param_loss = criterion(pred_norm, yb)

            # Differentiable physics loss: denormalize → simulate → compare
            pred_phys = pred_norm * t_param_range + t_param_min
            re_sim = simulate_reflectance_torch(
                pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2],
                t_wavelengths
            )
            physics_loss = criterion(re_sim, xb_raw)

            loss = param_loss + LAMBDA_PHYSICS * physics_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(train_ds)

        # --- Validate ---
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for xb, yb, xb_raw in val_loader:
                xb, yb, xb_raw = xb.to(device), yb.to(device), xb_raw.to(device)
                pred_norm = model(xb)
                param_loss = criterion(pred_norm, yb)
                pred_phys = pred_norm * t_param_range + t_param_min
                re_sim = simulate_reflectance_torch(
                    pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2],
                    t_wavelengths
                )
                physics_loss = criterion(re_sim, xb_raw)
                loss = param_loss + LAMBDA_PHYSICS * physics_loss
                running_loss += loss.item() * len(xb)
        val_loss = running_loss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "spectranet_v4.pt")
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

    # ──────────────────────────────────────────────
    # 4. Evaluation
    # ──────────────────────────────────────────────

    def evaluate_model(net, loader):
        """Run inference, denormalize, compute MAE and R² per parameter."""
        net.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                xb, yb = batch[0].to(device), batch[1]
                pred = net(xb)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.numpy())

        pred_norm = np.concatenate(all_preds)
        true_norm = np.concatenate(all_targets)

        pred_phys = pred_norm * param_range + param_min
        true_phys = true_norm * param_range + param_min

        mae = np.mean(np.abs(pred_phys - true_phys), axis=0)
        ss_res = np.sum((true_phys - pred_phys) ** 2, axis=0)
        ss_tot = np.sum((true_phys - true_phys.mean(axis=0)) ** 2, axis=0)
        r2_per = 1.0 - ss_res / ss_tot
        r2_all = r2_per.mean()

        return pred_phys, true_phys, mae, r2_per, r2_all

    # V4 evaluation
    model.load_state_dict(torch.load("spectranet_v4.pt", weights_only=True))
    pred_v4, true_v4, mae_v4, r2_per_v4, r2_all_v4 = evaluate_model(
        model, test_loader
    )

    # V3 evaluation on the same test split
    # V3 and V4 use the same dataset (dataset_v2.npz) and same split
    # (seed=42), so normalization stats are identical. We can evaluate
    # V3 directly on V4's test_loader without re-normalization.
    model_v3 = SpectraNet().to(device)
    model_v3.load_state_dict(torch.load("spectranet_v3.pt", weights_only=True))
    _, _, mae_v3, r2_per_v3, r2_all_v3 = evaluate_model(model_v3, test_loader)

    # ──────────────────────────────────────────────
    # 5. Comparison table
    # ──────────────────────────────────────────────

    labels = ["thickness (nm)", "n", "k"]

    print("\n" + "=" * 65)
    print("V3 vs V4 (same test split, dataset_v2.npz)")
    print("  V3: SpectraNet, 500k, param loss only")
    print("  V4: SpectraNet, 500k, param + differentiable physics loss")
    print("=" * 65)
    print(f"  {'Parameter':20s} {'V3 MAE':>10s} {'V4 MAE':>10s} "
          f"{'V3 R²':>8s} {'V4 R²':>8s}")
    print("-" * 65)
    for i, name in enumerate(labels):
        print(f"  {name:20s} {mae_v3[i]:10.4f} {mae_v4[i]:10.4f} "
              f"{r2_per_v3[i]:8.4f} {r2_per_v4[i]:8.4f}")
    print("-" * 65)
    print(f"  {'Overall R²':20s} {'':>10s} {'':>10s} "
          f"{r2_all_v3:8.4f} {r2_all_v4:8.4f}")
    print("=" * 65)

    # ──────────────────────────────────────────────
    # 6. Plots
    # ──────────────────────────────────────────────

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (param MSE + 0.05 * physics MSE)")
    ax.set_title("SpectraNet V4 — Training and Validation Loss")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_curve_v4.png", dpi=150)
    print("\nSaved loss_curve_v4.png")

    # --- Parity plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    units = ["nm", "", ""]

    for i, (ax, name, unit) in enumerate(zip(axes, labels, units)):
        t = true_v4[:, i]
        p = pred_v4[:, i]
        ax.scatter(t, p, s=1, alpha=0.15, rasterized=True)
        lo, hi = t.min(), t.max()
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.annotate(f"MAE = {mae_v4[i]:.4f}{(' ' + unit) if unit else ''}",
                    xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=10, backgroundcolor="white")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    fig.suptitle("SpectraNet V4 — Predicted vs True (Test Set)", fontsize=13,
                 y=1.02)
    fig.tight_layout()
    fig.savefig("parity_plot_v4.png", dpi=150, bbox_inches="tight")
    print("Saved parity_plot_v4.png")
