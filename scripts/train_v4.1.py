"""
Train SpectraNet V4.1 with safer speed optimizations.

Main upgrades over V4:
1) float32 and complex64 differentiable TMM physics loss
2) non_blocking CPU to GPU transfers
3) AMP for the network forward and parameter loss
4) optional torch.compile(model) with graceful fallback
5) larger default batch size
6) optional physics loss throttling controls
"""

import time
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import artifact_path, ensure_parent_dir
from src.spectranet import SpectraNet
from src.tmm_simulator import simulate_reflectance_torch_fast


if __name__ == "__main__":
    # 1) Data loading and preprocessing
    data = np.load(artifact_path("data", "dataset_v2.npz"))
    X_all = data["X"].astype(np.float32)
    y_all = data["y"].astype(np.float32)

    wavelengths_np = np.linspace(400, 800, 200).astype(np.float32)

    param_bounds = np.array(
        [
            [10.0, 300.0],  # thickness (nm)
            [1.3, 2.5],  # n
            [0.0, 0.5],  # k
        ],
        dtype=np.float32,
    )
    param_min = param_bounds[:, 0]
    param_range = param_bounds[:, 1] - param_bounds[:, 0]

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_all))
    n_train = int(0.8 * len(X_all))
    n_val = int(0.1 * len(X_all))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
    X_val_raw, y_val = X_all[val_idx], y_all[val_idx]
    X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

    X_mean = X_train_raw.mean(axis=0)
    X_std = X_train_raw.std(axis=0)
    X_std[X_std < 1e-8] = 1.0

    np.savez(
        ensure_parent_dir(artifact_path("data", "spectra_norm_v4_1.npz")),
        mean=X_mean,
        std=X_std,
    )

    X_train_norm = (X_train_raw - X_mean) / X_std
    X_val_norm = (X_val_raw - X_mean) / X_std
    X_test_norm = (X_test_raw - X_mean) / X_std

    y_train_norm = (y_train - param_min) / param_range
    y_val_norm = (y_val - param_min) / param_range
    y_test_norm = (y_test - param_min) / param_range

    # 2) Speed and training controls
    BATCH_SIZE = 1024
    NUM_WORKERS = 4
    LAMBDA_PHYSICS = 0.05
    MAX_EPOCHS = 300
    EARLY_STOP_PATIENCE = 30

    USE_AMP = False
    USE_MODEL_COMPILE = False
    PHYSICS_EVERY_N_BATCHES = 1
    PHYSICS_START_EPOCH = 1
    VALIDATE_EVERY = 1

    train_ds = TensorDataset(
        torch.from_numpy(X_train_norm),
        torch.from_numpy(y_train_norm),
        torch.from_numpy(X_train_raw),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_norm),
        torch.from_numpy(y_val_norm),
        torch.from_numpy(X_val_raw),
    )
    test_ds = TensorDataset(torch.from_numpy(X_test_norm), torch.from_numpy(y_test_norm))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # 3) Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_blocking = device.type == "cuda"
    use_amp = USE_AMP and device.type == "cuda"

    base_model = SpectraNet().to(device)
    model = base_model
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if USE_MODEL_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        try:
            os.environ.setdefault(
                "TRITON_CACHE_DIR",
                str(ensure_parent_dir(artifact_path("tmp", "triton_cache", ".keep")).parent),
            )
            model = torch.compile(base_model)
            with torch.no_grad():
                dummy = torch.zeros((4, 200), device=device)
                _ = model(dummy)
            torch.cuda.synchronize()
            print("Enabled torch.compile for model.")
        except Exception as exc:
            base_model = SpectraNet().to(device)
            model = base_model
            print(f"torch.compile unavailable, using eager mode: {exc}")

    # We could also compile the complex TMM function for extra speed, but we do
    # not do that here because complex autograd graphs can be unstable across
    # PyTorch and CUDA versions and may reduce training reliability.

    t_param_min = torch.from_numpy(param_min).to(device)
    t_param_range = torch.from_numpy(param_range).to(device)
    t_wavelengths = torch.from_numpy(wavelengths_np).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=10,
        factor=0.5,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 4) Train loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(
        "Settings: "
        f"batch={BATCH_SIZE}, amp={use_amp}, compile={USE_MODEL_COMPILE}, "
        f"physics_every={PHYSICS_EVERY_N_BATCHES}, physics_start={PHYSICS_START_EPOCH}, "
        f"validate_every={VALIDATE_EVERY}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        for batch_idx, (xb, yb, xb_raw) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            xb_raw = xb_raw.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                pred_norm = model(xb)
                param_loss = criterion(pred_norm, yb)

            use_physics = (epoch >= PHYSICS_START_EPOCH) and (batch_idx % PHYSICS_EVERY_N_BATCHES == 0)
            if use_physics:
                pred_phys = pred_norm.float() * t_param_range + t_param_min
                re_sim = simulate_reflectance_torch_fast(
                    pred_phys[:, 0],
                    pred_phys[:, 1],
                    pred_phys[:, 2],
                    t_wavelengths,
                )
                physics_loss = criterion(re_sim, xb_raw.float())
                loss = param_loss.float() + LAMBDA_PHYSICS * physics_loss
            else:
                loss = param_loss.float()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(xb)

        train_loss = running_loss / len(train_ds)

        did_validate = epoch % VALIDATE_EVERY == 0 or epoch == 1
        if did_validate:
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for xb, yb, xb_raw in val_loader:
                    xb = xb.to(device, non_blocking=non_blocking)
                    yb = yb.to(device, non_blocking=non_blocking)
                    xb_raw = xb_raw.to(device, non_blocking=non_blocking)

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        pred_norm = model(xb)
                        param_loss = criterion(pred_norm, yb)

                    pred_phys = pred_norm.float() * t_param_range + t_param_min
                    re_sim = simulate_reflectance_torch_fast(
                        pred_phys[:, 0],
                        pred_phys[:, 1],
                        pred_phys[:, 2],
                        t_wavelengths,
                    )
                    physics_loss = criterion(re_sim, xb_raw.float())
                    loss = param_loss.float() + LAMBDA_PHYSICS * physics_loss
                    running_loss += loss.item() * len(xb)

            val_loss = running_loss / len(val_ds)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
        else:
            val_loss = val_losses[-1] if val_losses else float("inf")

        train_losses.append(train_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if did_validate:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    base_model.state_dict(),
                    ensure_parent_dir(artifact_path("models", "spectranet_v4_1.pt")),
                )
            else:
                epochs_without_improvement += 1

        epoch_time = time.time() - epoch_start
        if epoch % 10 == 0 or epoch == 1 or epochs_without_improvement == 0:
            star = " *" if epochs_without_improvement == 0 else ""
            print(
                f"Epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                f"lr {current_lr:.1e} | {epoch_time:.1f}s{star}"
            )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {EARLY_STOP_PATIENCE} epochs)"
            )
            break

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # 5) Evaluation
    def evaluate_model(net, loader):
        net.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=non_blocking)
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

    model_v4_1_eval = SpectraNet().to(device)
    model_v4_1_eval.load_state_dict(
        torch.load(artifact_path("models", "spectranet_v4_1.pt"), weights_only=True)
    )
    pred_v4_1, true_v4_1, mae_v4_1, r2_per_v4_1, r2_all_v4_1 = evaluate_model(
        model_v4_1_eval, test_loader
    )

    model_v4 = SpectraNet().to(device)
    model_v4.load_state_dict(torch.load(artifact_path("models", "spectranet_v4.pt"), weights_only=True))
    _, _, mae_v4, r2_per_v4, r2_all_v4 = evaluate_model(model_v4, test_loader)

    labels = ["thickness (nm)", "n", "k"]
    print("\n" + "=" * 70)
    print("V4 vs V4.1 (same test split, dataset_v2.npz)")
    print("  V4:   physics loss with complex128 TMM")
    print("  V4.1: speed-up config with AMP + fast TMM")
    print("=" * 70)
    print(
        f"  {'Parameter':20s} {'V4 MAE':>10s} {'V4.1 MAE':>10s} "
        f"{'V4 R^2':>10s} {'V4.1 R^2':>10s}"
    )
    print("-" * 70)
    for i, name in enumerate(labels):
        print(
            f"  {name:20s} {mae_v4[i]:10.4f} {mae_v4_1[i]:10.4f} "
            f"{r2_per_v4[i]:10.4f} {r2_per_v4_1[i]:10.4f}"
        )
    print("-" * 70)
    print(
        f"  {'Overall R^2':20s} {'':>10s} {'':>10s} "
        f"{r2_all_v4:10.4f} {r2_all_v4_1:10.4f}"
    )
    print("=" * 70)

    # 6) Plots
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (param MSE + 0.05 * physics MSE)")
    ax.set_title("SpectraNet V4.1 Training and Validation Loss")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = ensure_parent_dir(artifact_path("figures", "loss_curve_v4_1.png"))
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    units = ["nm", "", ""]

    for i, (ax, name, unit) in enumerate(zip(axes, labels, units)):
        t = true_v4_1[:, i]
        p = pred_v4_1[:, i]
        ax.scatter(t, p, s=1, alpha=0.15, rasterized=True)
        lo, hi = t.min(), t.max()
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        unit_suffix = f" {unit}" if unit else ""
        ax.annotate(
            f"MAE = {mae_v4_1[i]:.4f}{unit_suffix}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=10,
            backgroundcolor="white",
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    fig.suptitle("SpectraNet V4.1 Predicted vs True (Test Set)", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = ensure_parent_dir(artifact_path("figures", "parity_plot_v4_1.png"))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
