"""
External benchmark pipeline for V4 to V4.1 speed upgrades.

This script measures each change incrementally so we can quantify
where runtime improvements come from.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_speed_benchmark import (
    BenchmarkStage,
    run_speed_benchmark_pipeline,
    save_benchmark_results,
)


def build_default_stages(base_batch_size: int = 512, target_batch_size: int = 1024) -> list[BenchmarkStage]:
    return [
        BenchmarkStage(
            name="baseline_v4_like",
            batch_size=base_batch_size,
            non_blocking_transfer=False,
            use_fast_tmm=False,
            use_amp=False,
            use_compile=False,
            physics_every_n_batches=1,
        ),
        BenchmarkStage(
            name="change_1_fast_tmm_c64",
            batch_size=base_batch_size,
            non_blocking_transfer=False,
            use_fast_tmm=True,
            use_amp=False,
            use_compile=False,
            physics_every_n_batches=1,
        ),
        BenchmarkStage(
            name="change_2_non_blocking_transfer",
            batch_size=base_batch_size,
            non_blocking_transfer=True,
            use_fast_tmm=True,
            use_amp=False,
            use_compile=False,
            physics_every_n_batches=1,
        ),
        BenchmarkStage(
            name=f"change_3_batch_{target_batch_size}",
            batch_size=target_batch_size,
            non_blocking_transfer=True,
            use_fast_tmm=True,
            use_amp=False,
            use_compile=False,
            physics_every_n_batches=1,
        ),
        BenchmarkStage(
            name="change_4_amp_network",
            batch_size=target_batch_size,
            non_blocking_transfer=True,
            use_fast_tmm=True,
            use_amp=True,
            use_compile=False,
            physics_every_n_batches=1,
        ),
        BenchmarkStage(
            name="change_5_compile_model",
            batch_size=target_batch_size,
            non_blocking_transfer=True,
            use_fast_tmm=True,
            use_amp=True,
            use_compile=True,
            physics_every_n_batches=1,
        ),
        BenchmarkStage(
            name="optional_riskier_physics_every_2",
            batch_size=target_batch_size,
            non_blocking_transfer=True,
            use_fast_tmm=True,
            use_amp=True,
            use_compile=True,
            physics_every_n_batches=2,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark V4 to V4.1 speed changes.")
    parser.add_argument("--subset-size", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-stem", type=str, default="v4_to_v4_1_speed")
    parser.add_argument("--base-batch-size", type=int, default=512)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size is not None:
        base_batch_size = args.batch_size
        target_batch_size = args.batch_size
    else:
        base_batch_size = args.base_batch_size
        target_batch_size = args.target_batch_size

    stages = build_default_stages(
        base_batch_size=base_batch_size,
        target_batch_size=target_batch_size,
    )
    metrics = run_speed_benchmark_pipeline(
        stages=stages,
        subset_size=args.subset_size,
        repeats=args.repeats,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
    )
    json_path, csv_path = save_benchmark_results(metrics, stages, output_stem=args.output_stem)

    print("\nV4 to V4.1 speed benchmark")
    print("=" * 116)
    print(
        f"{'stage':36s} {'step ms':>10s} {'std ms':>10s} {'samples/s':>12s} "
        f"{'x baseline':>12s} {'x previous':>12s}"
    )
    print("-" * 116)
    for item in metrics:
        print(
            f"{item.stage_name:36s} "
            f"{item.mean_step_ms:10.3f} "
            f"{item.std_step_ms:10.3f} "
            f"{item.samples_per_sec:12.1f} "
            f"{item.speedup_vs_baseline:12.3f} "
            f"{item.speedup_vs_previous:12.3f}"
        )
    print("=" * 116)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
