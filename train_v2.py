"""
Train SpectraNetV2: improved architecture with BatchNorm, residual connection,
and a physics-informed loss that re-simulates spectra from predicted parameters.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from tmm_simulator import simulate_reflectance_batch

# ──────────────────────────────────────────────
# 1. Data loading and preprocessing
# ──────────────────────────────────────────────

WAVELENGTHS = np.linspace(400, 800, 200)

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

# DataLoaders — include raw spectra as third element for physics loss
BATCH_SIZE = 512

train_ds = TensorDataset(torch.from_numpy(X_train_norm),
                         torch.from_numpy(y_train_norm),
                         torch.from_numpy(X_train_raw))
val_ds = TensorDataset(torch.from_numpy(X_val_norm),
                       torch.from_numpy(y_val_norm),
                       torch.from_numpy(X_val_raw))
test_ds = TensorDataset(torch.from_numpy(X_test_norm),
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
                nn.Dropout(0.15),
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
# 3. Physics-informed loss helper
# ──────────────────────────────────────────────

LAMBDA_PHYSICS = 0.1

# Torch tensors for denormalization (on device)
t_param_min = torch.from_numpy(param_min).to(device)
t_param_range = torch.from_numpy(param_range).to(device)


def compute_physics_loss(pred_norm, raw_spectra):
    """
    Denormalize predicted params, re-simulate spectra via vectorized TMM,
    and return MSE vs the original (raw, unnormalized) input spectra.

    Both pred_norm and raw_spectra are detached for the numpy TMM call.
    The returned loss is a plain tensor (no grad) used as a regularizer.
    """
    # Denormalize predictions to physical units
    pred_phys = pred_norm.detach().cpu().numpy() * param_range + param_min

    # Re-simulate all spectra in one vectorized call — no Python loop
    re_simulated = simulate_reflectance_batch(
        pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2], WAVELENGTHS
    ).astype(np.float32)

    # MSE between re-simulated and original raw spectra
    re_sim_t = torch.from_numpy(re_simulated).to(device)
    raw_t = raw_spectra.to(device)
    physics_mse = nn.functional.mse_loss(re_sim_t, raw_t)
    return physics_mse


# ──────────────────────────────────────────────
# 4. Training loop
# ──────────────────────────────────────────────

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
    # --- Train ---
    model.train()
    running_loss = 0.0
    for xb, yb, xb_raw in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)

        # Parameter MSE (differentiable, drives learning)
        param_loss = criterion(pred, yb)

        # Physics consistency loss (non-differentiable regularizer)
        physics_loss = compute_physics_loss(pred, xb_raw)

        loss = param_loss + LAMBDA_PHYSICS * physics_loss

        optimizer.zero_grad()
        param_loss.backward()  # only param_loss has gradients
        optimizer.step()
        running_loss += loss.item() * len(xb)
    train_loss = running_loss / len(train_ds)

    # --- Validate ---
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb, xb_raw in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            param_loss = criterion(pred, yb)
            physics_loss = compute_physics_loss(pred, xb_raw)
            loss = param_loss + LAMBDA_PHYSICS * physics_loss
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
ax.set_ylabel("Loss (param MSE + 0.1 * physics MSE)")
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
