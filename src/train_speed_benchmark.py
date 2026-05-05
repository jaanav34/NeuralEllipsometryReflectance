from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.paths import artifact_path, ensure_parent_dir
from src.spectranet import SpectraNet
from src.tmm_simulator import simulate_reflectance_torch, simulate_reflectance_torch_fast


@dataclass(frozen=True)
class BenchmarkStage:
    name: str
    batch_size: int
    non_blocking_transfer: bool
    use_fast_tmm: bool
    use_amp: bool
    use_compile: bool
    physics_every_n_batches: int


@dataclass(frozen=True)
class StageMetrics:
    stage_name: str
    mean_step_ms: float
    std_step_ms: float
    mean_transfer_ms: float
    mean_compute_ms: float
    mean_optimize_ms: float
    samples_per_sec: float
    speedup_vs_baseline: float
    speedup_vs_previous: float


def _load_subset(
    subset_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(artifact_path("data", "dataset_v2.npz"))
    X_all = data["X"].astype(np.float32)
    y_all = data["y"].astype(np.float32)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_all))
    n_train = int(0.8 * len(X_all))
    train_idx = indices[:n_train]

    if subset_size > len(train_idx):
        subset_size = len(train_idx)
    train_idx = train_idx[:subset_size]

    X_train_raw = X_all[train_idx]
    y_train = y_all[train_idx]

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_train_norm = (X_train_raw - x_mean) / x_std

    param_bounds = np.array(
        [
            [10.0, 300.0],
            [1.3, 2.5],
            [0.0, 0.5],
        ],
        dtype=np.float32,
    )
    param_min = param_bounds[:, 0]
    param_range = param_bounds[:, 1] - param_bounds[:, 0]
    y_train_norm = (y_train - param_min) / param_range
    wavelengths = np.linspace(400, 800, 200).astype(np.float32)

    return X_train_norm, y_train_norm, X_train_raw, param_min, param_range, wavelengths


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _build_model(device: torch.device, use_compile: bool, batch_size: int) -> torch.nn.Module:
    model = SpectraNet().to(device)
    if use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            os.environ.setdefault(
                "TRITON_CACHE_DIR",
                str(ensure_parent_dir(artifact_path("tmp", "triton_cache", ".keep")).parent),
            )
            model = torch.compile(model)
            with torch.no_grad():
                dummy = torch.zeros((max(1, min(batch_size, 4)), 200), device=device)
                _ = model(dummy)
            _sync_if_cuda(device)
        except Exception:
            model = SpectraNet().to(device)
    return model


def _run_single_stage(
    stage: BenchmarkStage,
    device: torch.device,
    num_workers: int,
    num_steps: int,
    warmup_steps: int,
    X_train_norm: np.ndarray,
    y_train_norm: np.ndarray,
    X_train_raw: np.ndarray,
    param_min: np.ndarray,
    param_range: np.ndarray,
    wavelengths: np.ndarray,
) -> dict[str, float]:
    dataset = TensorDataset(
        torch.from_numpy(X_train_norm),
        torch.from_numpy(y_train_norm),
        torch.from_numpy(X_train_raw),
    )
    loader = DataLoader(
        dataset,
        batch_size=stage.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    loader_iter = iter(loader)

    model = _build_model(device, stage.use_compile, stage.batch_size)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = stage.use_amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    non_blocking = stage.non_blocking_transfer and device.type == "cuda"

    t_param_min = torch.from_numpy(param_min).to(device)
    t_param_range = torch.from_numpy(param_range).to(device)
    t_wavelengths = torch.from_numpy(wavelengths).to(device)
    tmm_fn = simulate_reflectance_torch_fast if stage.use_fast_tmm else simulate_reflectance_torch
    lambda_physics = 0.05

    transfer_times = []
    compute_times = []
    optimize_times = []
    total_times = []
    sample_count = 0

    max_total_steps = num_steps + warmup_steps
    for step_index in range(max_total_steps):
        try:
            xb, yb, xb_raw = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            xb, yb, xb_raw = next(loader_iter)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        xb = xb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking)
        xb_raw = xb_raw.to(device, non_blocking=non_blocking)
        _sync_if_cuda(device)
        t1 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_norm = model(xb)
            param_loss = criterion(pred_norm, yb)

        use_physics = step_index % stage.physics_every_n_batches == 0
        if use_physics:
            pred_phys = pred_norm.float() * t_param_range + t_param_min
            re_sim = tmm_fn(
                pred_phys[:, 0],
                pred_phys[:, 1],
                pred_phys[:, 2],
                t_wavelengths,
            )
            physics_loss = criterion(re_sim, xb_raw.float())
            loss = param_loss.float() + lambda_physics * physics_loss
        else:
            loss = param_loss.float()

        _sync_if_cuda(device)
        t2 = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _sync_if_cuda(device)
        t3 = time.perf_counter()

        if step_index >= warmup_steps:
            transfer_times.append((t1 - t0) * 1000.0)
            compute_times.append((t2 - t1) * 1000.0)
            optimize_times.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)
            sample_count += len(xb)

    mean_total_ms = float(mean(total_times))
    total_sec = sum(total_times) / 1000.0
    throughput = sample_count / total_sec if total_sec > 0 else 0.0

    return {
        "step_ms": mean_total_ms,
        "step_ms_std": float(stdev(total_times)) if len(total_times) > 1 else 0.0,
        "transfer_ms": float(mean(transfer_times)),
        "compute_ms": float(mean(compute_times)),
        "optimize_ms": float(mean(optimize_times)),
        "samples_per_sec": throughput,
    }


def run_speed_benchmark_pipeline(
    stages: list[BenchmarkStage],
    subset_size: int = 65536,
    repeats: int = 3,
    num_steps: int = 120,
    warmup_steps: int = 20,
    num_workers: int = 4,
) -> list[StageMetrics]:
    if not stages:
        raise ValueError("stages must not be empty")

    (
        X_train_norm,
        y_train_norm,
        X_train_raw,
        param_min,
        param_range,
        wavelengths,
    ) = _load_subset(subset_size=subset_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    stage_runs: list[dict[str, float]] = []
    for stage in stages:
        per_repeat = []
        for _ in range(repeats):
            metrics = _run_single_stage(
                stage=stage,
                device=device,
                num_workers=num_workers,
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                X_train_norm=X_train_norm,
                y_train_norm=y_train_norm,
                X_train_raw=X_train_raw,
                param_min=param_min,
                param_range=param_range,
                wavelengths=wavelengths,
            )
            per_repeat.append(metrics)

        aggregated = {
            "stage_name": stage.name,
            "mean_step_ms": float(mean(item["step_ms"] for item in per_repeat)),
            "std_step_ms": float(mean(item["step_ms_std"] for item in per_repeat)),
            "mean_transfer_ms": float(mean(item["transfer_ms"] for item in per_repeat)),
            "mean_compute_ms": float(mean(item["compute_ms"] for item in per_repeat)),
            "mean_optimize_ms": float(mean(item["optimize_ms"] for item in per_repeat)),
            "samples_per_sec": float(mean(item["samples_per_sec"] for item in per_repeat)),
        }
        stage_runs.append(aggregated)

    baseline_step_ms = stage_runs[0]["mean_step_ms"]
    results: list[StageMetrics] = []
    for index, metrics in enumerate(stage_runs):
        prev_step_ms = baseline_step_ms if index == 0 else stage_runs[index - 1]["mean_step_ms"]
        result = StageMetrics(
            stage_name=metrics["stage_name"],
            mean_step_ms=metrics["mean_step_ms"],
            std_step_ms=metrics["std_step_ms"],
            mean_transfer_ms=metrics["mean_transfer_ms"],
            mean_compute_ms=metrics["mean_compute_ms"],
            mean_optimize_ms=metrics["mean_optimize_ms"],
            samples_per_sec=metrics["samples_per_sec"],
            speedup_vs_baseline=baseline_step_ms / metrics["mean_step_ms"],
            speedup_vs_previous=prev_step_ms / metrics["mean_step_ms"],
        )
        results.append(result)

    return results


def save_benchmark_results(
    metrics: list[StageMetrics],
    stages: list[BenchmarkStage],
    output_stem: str = "v4_to_v4_1_speed",
) -> tuple[str, str]:
    base_dir = ensure_parent_dir(artifact_path("benchmarks", f"{output_stem}.json")).parent
    json_path = str(base_dir / f"{output_stem}.json")
    csv_path = str(base_dir / f"{output_stem}.csv")

    payload = {
        "stages": [asdict(stage) for stage in stages],
        "metrics": [asdict(item) for item in metrics],
    }
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(payload, f_json, indent=2)

    fieldnames = list(asdict(metrics[0]).keys()) if metrics else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))

    return json_path, csv_path
