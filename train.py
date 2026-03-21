"""
Train SpectraNet: a neural network that inverts reflectance spectra
back into thin-film parameters [thickness_nm, n, k].
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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

# Reproducible 80/10/10 split
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
X_std[X_std < 1e-8] = 1.0  # guard against zero-variance channels

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Save normalization statistics for inference later
np.savez("spectra_norm.npz", mean=X_mean, std=X_std)

# Normalize parameters to [0, 1] using fixed physical bounds
y_train = (y_train - param_min) / param_range
y_val = (y_val - param_min) / param_range
y_test_norm = (y_test - param_min) / param_range

# DataLoaders
BATCH_SIZE = 512

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_ds = TensorDataset(torch.from_numpy(X_test - 0 * X_test),
                         torch.from_numpy(y_test_norm))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ──────────────────────────────────────────────
# 2. Model
# ──────────────────────────────────────────────

class SpectraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectraNet().to(device)
print(f"Training on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ──────────────────────────────────────────────
# 3. Training
# ──────────────────────────────────────────────

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)

MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 20

train_losses = []
val_losses = []
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # --- Train ---
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
    train_loss = running_loss / len(train_ds)

    # --- Validate ---
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
        torch.save(model.state_dict(), "spectranet.pt")
    else:
        epochs_without_improvement += 1

    if epoch % 10 == 0 or epoch == 1 or epochs_without_improvement == 0:
        print(f"Epoch {epoch:3d} | train {train_loss:.6f} | "
              f"val {val_loss:.6f} | lr {current_lr:.1e}"
              f"{' *' if epochs_without_improvement == 0 else ''}")

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
        break

print(f"\nBest validation loss: {best_val_loss:.6f}")

# ──────────────────────────────────────────────
# 4. Test evaluation
# ──────────────────────────────────────────────

# Load best weights
model.load_state_dict(torch.load("spectranet.pt", weights_only=True))
model.eval()

all_preds = []
all_targets = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(yb.numpy())

pred_norm = np.concatenate(all_preds)    # (N_test, 3) in [0, 1]
true_norm = np.concatenate(all_targets)  # (N_test, 3) in [0, 1]

# Denormalize to original physical units
pred_phys = pred_norm * param_range + param_min
true_phys = true_norm * param_range + param_min

# MAE per parameter in original units
mae = np.mean(np.abs(pred_phys - true_phys), axis=0)
labels = ["thickness (nm)", "n", "k"]

# R² across all outputs (treating each output independently, then averaging)
ss_res = np.sum((true_phys - pred_phys) ** 2, axis=0)
ss_tot = np.sum((true_phys - true_phys.mean(axis=0)) ** 2, axis=0)
r2_per_param = 1.0 - ss_res / ss_tot
r2_overall = r2_per_param.mean()

print("\n" + "=" * 50)
print("TEST SET RESULTS")
print("=" * 50)
for i, name in enumerate(labels):
    print(f"  {name:20s}  MAE = {mae[i]:.4f}   R² = {r2_per_param[i]:.4f}")
print(f"  {'Overall R²':20s}        {r2_overall:.4f}")
print("=" * 50)

# ──────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────

# --- Loss curve ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses, label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training and Validation Loss")
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("loss_curve.png", dpi=150)
print("Saved loss_curve.png")

# --- Parity plots ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
units = ["nm", "", ""]

for i, (ax, name, unit) in enumerate(zip(axes, labels, units)):
    t = true_phys[:, i]
    p = pred_phys[:, i]
    ax.scatter(t, p, s=1, alpha=0.15, rasterized=True)
    lo, hi = t.min(), t.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel(f"True {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.set_title(name)
    ax.annotate(f"MAE = {mae[i]:.4f}{(' ' + unit) if unit else ''}",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, backgroundcolor="white")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

fig.suptitle("Predicted vs True — Test Set", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("parity_plot.png", dpi=150, bbox_inches="tight")
print("Saved parity_plot.png")
