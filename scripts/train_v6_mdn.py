"""
Train SpectraNet V6 MDN posterior model.

Predicts a posterior distribution over (thickness, n, k) instead of one point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import artifact_path, ensure_parent_dir
from src.spectranet_mdn import SpectraNetMDN, mdn_negative_log_likelihood

PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float32)
PARAM_RANGE = np.array([290.0, 1.2, 0.5], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpectraNet MDN posterior model.")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_train = int(0.8 * len(x))
    n_val = int(0.1 * len(x))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]


def _normalize_params(y: np.ndarray) -> np.ndarray:
    return (y - PARAM_MIN) / PARAM_RANGE


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(artifact_path("data", "dataset_v2.npz"))
    x_all = data["X"].astype(np.float32)
    y_all = data["y"].astype(np.float32)
    x_train_raw, y_train_phys, x_val_raw, y_val_phys, x_test_raw, y_test_phys = _split(x_all, y_all, args.seed)

    x_mean = x_train_raw.mean(axis=0)
    x_std = x_train_raw.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    x_train = (x_train_raw - x_mean) / x_std
    x_val = (x_val_raw - x_mean) / x_std
    x_test = (x_test_raw - x_mean) / x_std

    y_train = _normalize_params(y_train_phys)
    y_val = _normalize_params(y_val_phys)
    y_test = _normalize_params(y_test_phys)

    np.savez(
        ensure_parent_dir(artifact_path("data", "spectra_norm_v6_mdn.npz")),
        mean=x_mean,
        std=x_std,
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraNetMDN(n_components=args.components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val = float("inf")
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            out = model(xb)
            loss = mdn_negative_log_likelihood(out["logits"], out["means"], out["scales"], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                out = model(xb)
                val_loss += float(
                    mdn_negative_log_likelihood(out["logits"], out["means"], out["scales"], yb).item()
                ) * len(xb)
            val_loss /= len(val_ds)

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ensure_parent_dir(artifact_path("models", "spectranet_v6_mdn.pt")))

        if epoch % 10 == 0 or epoch == 1 or epoch == best_epoch:
            print(f"epoch {epoch:03d} train_nll={train_loss:.6f} val_nll={val_loss:.6f}")

    model.load_state_dict(torch.load(artifact_path("models", "spectranet_v6_mdn.pt"), map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_nll = 0.0
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            out = model(xb)
            test_nll += float(
                mdn_negative_log_likelihood(out["logits"], out["means"], out["scales"], yb).item()
            ) * len(xb)
        test_nll /= len(test_ds)

    summary = {
        "components": args.components,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_nll": float(best_val),
        "test_nll": float(test_nll),
        "train_minutes": float((time.time() - start) / 60.0),
    }
    out_summary = ensure_parent_dir(artifact_path("benchmarks", "spectranet_v6_mdn_summary.json"))
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {artifact_path('models', 'spectranet_v6_mdn.pt')}")
    print(f"Saved summary: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
