"""
Train SpectraNetV2: improved architecture with BatchNorm, residual connection,
and consistency regularization via input perturbation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time


# ──────────────────────────────────────────────
# 1. Data loading and preprocessing
# ──────────────────────────────────────────────

data = np.load("dataset.npz")
X_all = data["X"].astype(np.float32)  # (100000, 200)
y_all = data["y"].astype(np.float32)  # (100000, 3)

# Fixed physical bounds for parameter normalization → [0, 1]
PARAM_BOUNDS = np.array([[10.0, 300.0],    # thickness (nm)
                         [1.3, 2.5],       # n
                         [0.0, 0.5]],      # k
                        dtype=np.float32)
param_min = PARAM_BOUNDS[:, 0]
param_range = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

# Reproducible 80/10/10 split (same seed as train.py)
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_all))
n_train = int(0.8 * len(X_all))
n_val = int(0.1 * len(X_all))

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

# Normalize spectra: zero mean, unit variance (fit on train only)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std < 1e-8] = 1.0

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Normalize parameters to [0, 1]
y_train_norm = (y_train - param_min) / param_range
y_val_norm = (y_val - param_min) / param_range
y_test_norm = (y_test - param_min) / param_range

# DataLoaders
BATCH_SIZE = 512

train_ds = TensorDataset(torch.from_numpy(X_train),
                         torch.from_numpy(y_train_norm))
val_ds = TensorDataset(torch.from_numpy(X_val),
                       torch.from_numpy(y_val_norm))
test_ds = TensorDataset(torch.from_numpy(X_test),
                        torch.from_numpy(y_test_norm))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ──────────────────────────────────────────────
# 2. Model: SpectraNetV2
# ──────────────────────────────────────────────

class SpectraNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Main trunk: 200 → 512 → 512 → 256 → 256 → 128 → 64
        hidden_sizes = [512, 512, 256, 256, 128, 64]
        layers = []
        in_dim = 200
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.25),
            ])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Skip connection: project input (200) down to 64 to match last
        # hidden layer output, then add before the output head
        self.skip_proj = nn.Linear(200, 64)

        # Output head: 64 → 3 with Sigmoid
        self.head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.trunk(x)              # (B, 64)
        skip = self.skip_proj(x)       # (B, 64)
        h = h + skip                   # residual addition
        return self.head(h)            # (B, 3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectraNetV2().to(device)
print(f"Training on {device}")
print(f"SpectraNetV2 parameters: {sum(p.numel() for p in model.parameters()):,}")

# ──────────────────────────────────────────────
# 3. Training loop with consistency regularization
# ──────────────────────────────────────────────

NOISE_STD = 0.005  # Gaussian noise std for input perturbation

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)

MAX_EPOCHS = 250
EARLY_STOP_PATIENCE = 30

train_losses = []
val_losses = []
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    # --- Train with consistency regularization ---
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # Prediction on clean input
        pred_clean = model(xb)
        loss_clean = criterion(pred_clean, yb)

        # Prediction on noisy input (consistency regularization)
        noise = torch.randn_like(xb) * NOISE_STD
        pred_noisy = model(xb + noise)
        loss_noisy = criterion(pred_noisy, yb)

        # Average both losses
        loss = 0.5 * (loss_clean + loss_noisy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
    train_loss = running_loss / len(train_ds)

    # --- Validate (clean only) ---
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            running_loss += loss.item() * len(xb)
    val_loss = running_loss / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    # Early stopping and checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "spectranet_v2.pt")
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
# 5. Test evaluation — V2
# ──────────────────────────────────────────────

def evaluate_model(net, loader, tag=""):
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


# Load best V2 weights and evaluate
model.load_state_dict(torch.load("spectranet_v2.pt", weights_only=True))
pred_v2, true_v2, mae_v2, r2_per_v2, r2_all_v2 = evaluate_model(
    model, test_loader, "V2"
)

# ──────────────────────────────────────────────
# 6. Evaluate V1 on the same test split
# ──────────────────────────────────────────────

# Import V1 architecture
from train import SpectraNet

model_v1 = SpectraNet().to(device)
model_v1.load_state_dict(torch.load("spectranet.pt", weights_only=True))
_, _, mae_v1, r2_per_v1, r2_all_v1 = evaluate_model(model_v1, test_loader, "V1")

# ──────────────────────────────────────────────
# 7. Print comparison table
# ──────────────────────────────────────────────

labels = ["thickness (nm)", "n", "k"]

print("\n" + "=" * 65)
print("SIDE-BY-SIDE COMPARISON: V1 vs V2 (test set)")
print("=" * 65)
print(f"  {'Parameter':20s} {'V1 MAE':>10s} {'V2 MAE':>10s} "
      f"{'V1 R²':>8s} {'V2 R²':>8s}")
print("-" * 65)
for i, name in enumerate(labels):
    print(f"  {name:20s} {mae_v1[i]:10.4f} {mae_v2[i]:10.4f} "
          f"{r2_per_v1[i]:8.4f} {r2_per_v2[i]:8.4f}")
print("-" * 65)
print(f"  {'Overall R²':20s} {'':>10s} {'':>10s} "
      f"{r2_all_v1:8.4f} {r2_all_v2:8.4f}")
print("=" * 65)

# ──────────────────────────────────────────────
# 8. Plots
# ──────────────────────────────────────────────

# --- Loss curve ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses, label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("SpectraNetV2 — Training and Validation Loss")
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("loss_curve_v2.png", dpi=150)
print("\nSaved loss_curve_v2.png")

# --- Parity plots ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
units = ["nm", "", ""]

for i, (ax, name, unit) in enumerate(zip(axes, labels, units)):
    t = true_v2[:, i]
    p = pred_v2[:, i]
    ax.scatter(t, p, s=1, alpha=0.15, rasterized=True)
    lo, hi = t.min(), t.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel(f"True {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.set_title(name)
    ax.annotate(f"MAE = {mae_v2[i]:.4f}{(' ' + unit) if unit else ''}",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, backgroundcolor="white")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

fig.suptitle("SpectraNetV2 — Predicted vs True (Test Set)", fontsize=13,
             y=1.02)
fig.tight_layout()
fig.savefig("parity_plot_v2.png", dpi=150, bbox_inches="tight")
print("Saved parity_plot_v2.png")
