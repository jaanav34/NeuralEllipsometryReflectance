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
from src.tmm_simulator import simulate_reflectance_batch

PARAM_MIN = np.array([10.0, 1.3, 0.0], dtype=np.float32)
PARAM_RANGE = np.array([290.0, 1.2, 0.5], dtype=np.float32)


def _overall_r2(pred: np.ndarray, true: np.ndarray) -> float:
    r2_vals: list[float] = []
    for i in range(pred.shape[1]):
        ss_res = float(np.sum((true[:, i] - pred[:, i]) ** 2))
        ss_tot = float(np.sum((true[:, i] - np.mean(true[:, i])) ** 2))
        r2_vals.append(1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan"))
    return float(np.nanmean(np.asarray(r2_vals, dtype=np.float64)))


def _catastrophic_rate(pred_phys: np.ndarray, true_phys: np.ndarray) -> float:
    abs_err = np.abs(pred_phys - true_phys)
    catastrophic = (abs_err[:, 0] > 25.0) | (abs_err[:, 1] > 0.25) | (abs_err[:, 2] > 0.10)
    return float(np.mean(catastrophic))


def _normalized_mean_abs_err(pred_phys: np.ndarray, true_phys: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_phys - true_phys) / PARAM_RANGE))


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _tmm_ranked_component_selection(
    means_norm: np.ndarray,
    spectra_raw: np.ndarray,
    wavelengths: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    n, k, _ = means_norm.shape
    chosen = np.empty((n, 3), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_means = means_norm[start:end]  # (B,K,3)
        flat = chunk_means.reshape(-1, 3)
        params_phys = flat * PARAM_RANGE + PARAM_MIN
        sim = simulate_reflectance_batch(
            params_phys[:, 0].astype(np.float64),
            params_phys[:, 1].astype(np.float64),
            params_phys[:, 2].astype(np.float64),
            wavelengths.astype(np.float64),
        )
        sim = sim.reshape(end - start, k, -1)
        mae = np.mean(np.abs(sim - spectra_raw[start:end, None, :].astype(np.float64)), axis=2)
        best = np.argmin(mae, axis=1)
        chosen[start:end] = chunk_means[np.arange(end - start), best]
    return chosen


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
        all_logits: list[np.ndarray] = []
        all_means: list[np.ndarray] = []
        all_scales: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            out = model(xb)
            test_nll += float(
                mdn_negative_log_likelihood(out["logits"], out["means"], out["scales"], yb).item()
            ) * len(xb)
            all_logits.append(out["logits"].cpu().numpy())
            all_means.append(out["means"].cpu().numpy())
            all_scales.append(out["scales"].cpu().numpy())
            all_true.append(yb.cpu().numpy())
        test_nll /= len(test_ds)

    logits_np = np.concatenate(all_logits, axis=0)
    means_np = np.concatenate(all_means, axis=0)
    scales_np = np.concatenate(all_scales, axis=0)
    true_norm = np.concatenate(all_true, axis=0)

    logits_t = torch.from_numpy(logits_np)
    means_t = torch.from_numpy(means_np)
    scales_t = torch.from_numpy(scales_np)
    weights_t = torch.softmax(logits_t, dim=1)
    mix_mean_norm = torch.sum(weights_t.unsqueeze(-1) * means_t, dim=1).numpy().astype(np.float32)
    top1_idx = torch.argmax(weights_t, dim=1)
    top1_norm = means_t[torch.arange(len(means_t)), top1_idx].numpy().astype(np.float32)

    true_t = torch.from_numpy(true_norm)
    comp_err = torch.mean(torch.abs(means_t - true_t.unsqueeze(1)), dim=2)
    oracle_idx = torch.argmin(comp_err, dim=1)
    oracle_norm = means_t[torch.arange(len(means_t)), oracle_idx].numpy().astype(np.float32)

    tmm_ranked_norm = _tmm_ranked_component_selection(
        means_norm=means_np,
        spectra_raw=x_test_raw.astype(np.float32, copy=False),
        wavelengths=np.linspace(400, 800, 200, dtype=np.float32),
    )

    entropy = (-(weights_t * torch.log(torch.clamp(weights_t, min=1e-12))).sum(dim=1)).numpy().astype(np.float32)

    true_phys = true_norm * PARAM_RANGE + PARAM_MIN
    mix_mean_phys = mix_mean_norm * PARAM_RANGE + PARAM_MIN
    top1_phys = top1_norm * PARAM_RANGE + PARAM_MIN
    oracle_phys = oracle_norm * PARAM_RANGE + PARAM_MIN
    tmm_ranked_phys = tmm_ranked_norm * PARAM_RANGE + PARAM_MIN

    top1_norm_err = np.mean(np.abs(top1_phys - true_phys) / PARAM_RANGE, axis=1)
    mix_norm_err = np.mean(np.abs(mix_mean_phys - true_phys) / PARAM_RANGE, axis=1)
    tmm_norm_err = np.mean(np.abs(tmm_ranked_phys - true_phys) / PARAM_RANGE, axis=1)
    oracle_norm_err = np.mean(np.abs(oracle_phys - true_phys) / PARAM_RANGE, axis=1)
    catastrophic_top1 = (
        (np.abs(top1_phys[:, 0] - true_phys[:, 0]) > 25.0)
        | (np.abs(top1_phys[:, 1] - true_phys[:, 1]) > 0.25)
        | (np.abs(top1_phys[:, 2] - true_phys[:, 2]) > 0.10)
    )
    oracle_hit = oracle_norm_err < 0.05

    summary = {
        "components": args.components,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_nll": float(best_val),
        "test_nll": float(test_nll),
        "top1_mae_thickness_nm": float(np.mean(np.abs(top1_phys[:, 0] - true_phys[:, 0]))),
        "top1_mae_n": float(np.mean(np.abs(top1_phys[:, 1] - true_phys[:, 1]))),
        "top1_mae_k": float(np.mean(np.abs(top1_phys[:, 2] - true_phys[:, 2]))),
        "top1_overall_r2": _overall_r2(top1_phys, true_phys),
        "top1_normalized_mae": _normalized_mean_abs_err(top1_phys, true_phys),
        "top1_catastrophic_rate": _catastrophic_rate(top1_phys, true_phys),
        "mixture_mean_normalized_mae": _normalized_mean_abs_err(mix_mean_phys, true_phys),
        "tmm_ranked_component_normalized_mae": _normalized_mean_abs_err(tmm_ranked_phys, true_phys),
        "oracle_component_normalized_mae": _normalized_mean_abs_err(oracle_phys, true_phys),
        "oracle_component_hit_rate_normerr_lt_0p05": float(np.mean(oracle_hit)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p90": float(np.percentile(entropy, 90)),
        "entropy_corr_with_top1_norm_error": _corrcoef_safe(entropy, top1_norm_err),
        "entropy_corr_with_mix_norm_error": _corrcoef_safe(entropy, mix_norm_err),
        "entropy_corr_with_tmm_ranked_norm_error": _corrcoef_safe(entropy, tmm_norm_err),
        "entropy_corr_with_top1_catastrophic": _corrcoef_safe(entropy, catastrophic_top1.astype(np.float32)),
        "train_minutes": float((time.time() - start) / 60.0),
    }
    out_summary = ensure_parent_dir(artifact_path("benchmarks", "spectranet_v6_mdn_summary.json"))
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {artifact_path('models', 'spectranet_v6_mdn.pt')}")
    print(f"Saved summary: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
