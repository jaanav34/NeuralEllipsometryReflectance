"""
Train SpectraNet V5 reliability model.

This script adds:
  1) hard-case curriculum sampling
  2) weighted training for thin and low-identifiability regimes
  3) explicit risk head for catastrophic failure prediction
  4) uncertainty calibration export
  5) catastrophic rate tracking as a first-class metric
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import artifact_path, ensure_parent_dir
from src.reliability_training import (
    PARAM_MIN,
    PARAM_RANGE,
    calibrate_ci_scale,
    catastrophic_flags_from_error,
    hard_case_weight_map,
    sample_curriculum_params,
    save_calibration,
)
from src.spectranet_reliability import SpectraNetReliability
from src.tmm_simulator import simulate_reflectance_batch


def _binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool)
    s = score.astype(np.float64)
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sum_ranks_pos = float(np.sum(ranks[y]))
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def _binary_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool)
    s = score.astype(np.float64)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _selective_rejection_stats(catastrophic: np.ndarray, risk_prob: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    base = float(np.mean(catastrophic))
    out["base_catastrophic_rate"] = base
    for frac in (0.10, 0.20, 0.30):
        cutoff = float(np.quantile(risk_prob, 1.0 - frac))
        keep = risk_prob < cutoff
        kept_rate = float(np.mean(catastrophic[keep])) if np.any(keep) else float("nan")
        out[f"reject_top_{int(frac * 100)}pct_kept_catastrophic_rate"] = kept_rate
        if np.isfinite(kept_rate) and base > 0:
            out[f"reject_top_{int(frac * 100)}pct_relative_drop"] = float((base - kept_rate) / base)
        else:
            out[f"reject_top_{int(frac * 100)}pct_relative_drop"] = float("nan")
    return out


def _risk_bucket_calibration(catastrophic: np.ndarray, risk_prob: np.ndarray) -> list[dict[str, float]]:
    bins = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            mask = (risk_prob >= lo) & (risk_prob < hi)
        else:
            mask = (risk_prob >= lo) & (risk_prob <= hi)
        if not np.any(mask):
            rows.append(
                {
                    "risk_lo": float(lo),
                    "risk_hi": float(hi),
                    "count": 0.0,
                    "mean_risk": float("nan"),
                    "catastrophic_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "risk_lo": float(lo),
                "risk_hi": float(hi),
                "count": float(np.sum(mask)),
                "mean_risk": float(np.mean(risk_prob[mask])),
                "catastrophic_rate": float(np.mean(catastrophic[mask])),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reliability-first SpectraNet V5.")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-risk", type=float, default=0.5)
    parser.add_argument("--lambda-uncert", type=float, default=0.25)
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--curriculum-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_params(params_phys: np.ndarray) -> np.ndarray:
    return (params_phys - PARAM_MIN) / PARAM_RANGE


def _load_dataset(use_curriculum: bool, curriculum_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if use_curriculum:
        hard_cases = artifact_path(
            "benchmarks",
            "reliability_matrix_smoke_nn_only",
            "reliability_matrix_smoke_nn_only_top_failures.csv",
        )
        params = sample_curriculum_params(curriculum_size, seed=seed, hard_cases_csv=hard_cases)
        spectra = simulate_reflectance_batch(
            params[:, 0].astype(np.float64),
            params[:, 1].astype(np.float64),
            params[:, 2].astype(np.float64),
            np.linspace(400, 800, 200, dtype=np.float64),
        ).astype(np.float32)
        return spectra, params.astype(np.float32)
    data = np.load(artifact_path("data", "dataset_v2.npz"))
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


def _split_data(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_train = int(0.8 * len(x))
    n_val = int(0.1 * len(x))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]


def _evaluate(
    model: SpectraNetReliability,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    all_mean: list[np.ndarray] = []
    all_std: list[np.ndarray] = []
    all_risk: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            all_mean.append(out["param_mean"].cpu().numpy())
            all_std.append(out["param_std"].cpu().numpy())
            all_risk.append(torch.sigmoid(out["risk_logit"]).cpu().numpy())
            all_true.append(yb.numpy())
    mean_norm = np.concatenate(all_mean)
    std_norm = np.concatenate(all_std)
    risk_prob = np.concatenate(all_risk)
    true_norm = np.concatenate(all_true)
    mean_phys = mean_norm * PARAM_RANGE + PARAM_MIN
    std_phys = std_norm * PARAM_RANGE
    true_phys = true_norm * PARAM_RANGE + PARAM_MIN
    abs_err = np.abs(mean_phys - true_phys)
    catastrophic = catastrophic_flags_from_error(abs_err)
    catastrophic_rate = float(np.mean(catastrophic))
    return {
        "pred_mean_phys": mean_phys,
        "pred_std_phys": std_phys,
        "risk_prob": risk_prob,
        "true_phys": true_phys,
        "abs_err": abs_err,
        "catastrophic_rate": catastrophic_rate,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_all, y_all = _load_dataset(args.use_curriculum, args.curriculum_size, args.seed)
    x_train_raw, y_train_phys, x_val_raw, y_val_phys, x_test_raw, y_test_phys = _split_data(x_all, y_all, args.seed)

    x_mean = x_train_raw.mean(axis=0)
    x_std = x_train_raw.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    np.savez(
        ensure_parent_dir(artifact_path("data", "spectra_norm_v5_reliability.npz")),
        mean=x_mean,
        std=x_std,
    )

    y_train_norm = _normalize_params(y_train_phys)
    y_val_norm = _normalize_params(y_val_phys)
    y_test_norm = _normalize_params(y_test_phys)

    x_train_norm = (x_train_raw - x_mean) / x_std
    x_val_norm = (x_val_raw - x_mean) / x_std
    x_test_norm = (x_test_raw - x_mean) / x_std

    train_weights = hard_case_weight_map(y_train_phys)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_norm),
        torch.from_numpy(y_train_norm),
        torch.from_numpy(train_weights),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_norm),
        torch.from_numpy(y_val_norm),
        torch.ones(len(x_val_norm), dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test_norm),
        torch.from_numpy(y_test_norm),
        torch.ones(len(x_test_norm), dtype=torch.float32),
    )

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
    model = SpectraNetReliability().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, wb in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            wb = wb.to(device, non_blocking=device.type == "cuda")

            out = model(xb)
            pred = out["param_mean"]
            pred_std = out["param_std"]
            risk_logit = out["risk_logit"]

            err = pred - yb
            mse_per = (err * err).mean(dim=1)
            weighted_mse = (mse_per * wb).mean()

            nll_param = 0.5 * (((err / pred_std) ** 2) + 2.0 * torch.log(pred_std)).mean(dim=1)
            weighted_nll = (nll_param * wb).mean()

            with torch.no_grad():
                abs_err_norm = torch.abs(err)
                catastrophic = (
                    (abs_err_norm[:, 0] > (25.0 / PARAM_RANGE[0]))
                    | (abs_err_norm[:, 1] > (0.25 / PARAM_RANGE[1]))
                    | (abs_err_norm[:, 2] > (0.10 / PARAM_RANGE[2]))
                ).float()

            risk_loss = f.binary_cross_entropy_with_logits(risk_logit, catastrophic)
            loss = weighted_mse + (args.lambda_uncert * weighted_nll) + (args.lambda_risk * risk_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb, wb in val_loader:
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                out = model(xb)
                err = out["param_mean"] - yb
                mse = (err * err).mean(dim=1)
                val_loss += float(mse.mean().item()) * len(xb)
            val_loss /= len(val_ds)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ensure_parent_dir(artifact_path("models", "spectranet_v5_reliability.pt")))

        if epoch % 10 == 0 or epoch == 1 or epoch == best_epoch:
            print(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

    model.load_state_dict(torch.load(artifact_path("models", "spectranet_v5_reliability.pt"), map_location=device, weights_only=True))
    val_eval = _evaluate(model, val_loader, device)
    test_eval = _evaluate(model, test_loader, device)

    ci_scales = calibrate_ci_scale(
        abs_err=val_eval["abs_err"],  # type: ignore[index]
        pred_std=val_eval["pred_std_phys"],  # type: ignore[index]
        target_coverage=0.95,
    )
    calibration_path = save_calibration(ci_scales, name="risk_ci_calibration_v5.npz")

    test_risk = np.asarray(test_eval["risk_prob"], dtype=np.float64)
    test_catastrophic = catastrophic_flags_from_error(np.asarray(test_eval["abs_err"], dtype=np.float64))
    auroc = _binary_auroc(test_catastrophic, test_risk)
    auprc = _binary_auprc(test_catastrophic, test_risk)
    selective = _selective_rejection_stats(test_catastrophic, test_risk)
    risk_buckets = _risk_bucket_calibration(test_catastrophic, test_risk)

    summary = {
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "train_minutes": float((time.time() - start) / 60.0),
        "use_curriculum": bool(args.use_curriculum),
        "curriculum_size": int(args.curriculum_size),
        "test_mae_thickness_nm": float(np.mean(test_eval["abs_err"][:, 0])),  # type: ignore[index]
        "test_mae_n": float(np.mean(test_eval["abs_err"][:, 1])),  # type: ignore[index]
        "test_mae_k": float(np.mean(test_eval["abs_err"][:, 2])),  # type: ignore[index]
        "test_catastrophic_rate": float(test_eval["catastrophic_rate"]),  # type: ignore[arg-type]
        "val_catastrophic_rate": float(val_eval["catastrophic_rate"]),  # type: ignore[arg-type]
        "test_catastrophic_base_rate": float(np.mean(test_catastrophic)),
        "risk_auroc": float(auroc),
        "risk_auprc": float(auprc),
        "risk_base_positive_rate": float(np.mean(test_catastrophic)),
        "selective_rejection": selective,
        "risk_bucket_calibration": risk_buckets,
        "ci_calibration_scales": [float(x) for x in ci_scales],
        "ci_calibration_path": str(calibration_path),
        "history_tail": history[-10:],
    }
    summary_path = ensure_parent_dir(artifact_path("benchmarks", "spectranet_v5_reliability_summary.json"))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {artifact_path('models', 'spectranet_v5_reliability.pt')}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
