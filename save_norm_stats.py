"""Save V4 normalization statistics (computed from dataset_v2.npz train split)."""

import numpy as np

data = np.load("dataset_v2.npz")
X_all = data["X"].astype(np.float32)

rng = np.random.default_rng(42)
indices = rng.permutation(len(X_all))
n_train = int(0.8 * len(X_all))
X_train = X_all[indices[:n_train]]

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std < 1e-8] = 1.0

np.savez("spectra_norm_v4.npz", mean=X_mean, std=X_std)
print("Saved spectra_norm_v4.npz")
print(f"  mean shape: {X_mean.shape}, std shape: {X_std.shape}")
