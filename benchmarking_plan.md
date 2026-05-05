# Neural Thin-Film Metrology Failure Benchmarking Plan

## Goal

The goal is not speed. The goal is to expose hidden technical debt in the inverse pipeline:

1. where the neural network fails,
2. where MC Dropout correctly warns that it is uncertain,
3. where the denoiser helps or hurts the initial guess,
4. where the L-BFGS-B refiner improves spectral fit but worsens physical parameters,
5. which regions of parameter space are physically non-identifiable.

The key benchmark principle is: **run the same path as the app**.

```text
spectrum -> optional denoiser -> MC Dropout SpectraNet mean -> refiner against raw spectrum
```

## Why the manual failures are happening

The examples are concentrated around 30-70 nm films. In that region, the 400-800 nm spectrum often has fewer than one full interference fringe. The spectrum becomes smooth, and multiple `(thickness, n, k)` triples can produce nearly the same reflectance curve.

That means the refiner can lower spectral MAE while moving away from truth. It is not optimizing parameter accuracy. It only sees spectral residual. If two physically different parameter sets give similar spectra, the optimizer may prefer the wrong basin.

MC Dropout is already telling you this in several cases. CIs like `±28 nm` and `±0.17 n` are not minor. They are the model saying the inverse is underdetermined.

## Benchmark structure

### Tier 1: NN + denoiser + uncertainty map

Run this over hundreds of thousands to millions of samples. This is GPU-friendly.

Outputs:

- per-sample truth
- NN MC Dropout mean
- NN MC Dropout std and 95% CI
- spectral MAE from NN re-simulation
- catastrophic flags
- slice metrics by thickness, n, k, and approximate fringe count

### Tier 2: Refiner stress subset

Running SciPy L-BFGS-B on millions of samples is possible only as an overnight CPU job and is usually not the best first move. The better approach is to refine:

- all manual probe cases,
- top NN failures,
- thin-film region failures,
- high-uncertainty cases,
- random calibration subset.

This tells you whether the refiner is actually fixing hard cases or simply finding alternate spectra-equivalent solutions.

### Tier 3: Robust refiner diagnosis

For the worst failures, run multi-start refinement and optional coarse-grid initialization. If many starts land in different parameter basins with similar spectral MSE, the problem is non-identifiable under the current measurement setup.

That is an important scientific result, not just a bug.

## First commands

Fast but meaningful smoke test:

```bash
python scripts/benchmark_inference_suite.py --n-random 20000 --noise-levels 0 0.001 --mc-samples 30 --include-probes --refine-strategy catastrophic --max-refine 1000 --output-stem smoke_failure_atlas
```

Full NN uncertainty atlas:

```bash
python scripts/benchmark_inference_suite.py --n-random 1000000 --noise-levels 0 0.001 0.005 --mc-samples 30 --refine-strategy none --output-stem million_nn_atlas
```

Hard-case refiner benchmark:

```bash
python scripts/benchmark_inference_suite.py --n-random 200000 --noise-levels 0 0.001 0.005 --mc-samples 50 --include-probes --refine-strategy catastrophic --max-refine 20000 --robust-refiner --refiner-workers 8 --output-stem refiner_hard_cases
```

Dense grid atlas:

```bash
python scripts/benchmark_inference_suite.py --n-random 0 --grid --grid-thickness 120 --grid-n 80 --grid-k 80 --noise-levels 0 --mc-samples 20 --refine-strategy none --output-stem grid_768k_nn_atlas
```

Plot one NPZ output:

```bash
python scripts/plot_benchmark_results.py artifacts/benchmarks/grid_768k_nn_atlas/grid_768k_nn_atlas_noise_0p0.npz
```

## Acceptance criteria

The model does not need to be perfect everywhere. It needs to be honest and stable.

Suggested thresholds:

- thickness MAE under 3 nm globally on clean spectra,
- n MAE under 0.045 globally,
- k MAE under 0.008 globally,
- catastrophic failure rate under 1% outside thin/low-fringe region,
- CI miss rate near or below 5-10% for each parameter if using 95% intervals,
- refiner should not increase normalized parameter error on more than 20% of selected hard cases,
- cases with good spectral fit but bad parameters should be explicitly flagged as non-identifiable.

## Likely fixes after benchmarking

1. Add a reliability gate to the app: if CI is wide or approximate fringe count is low, show an ambiguity warning.
2. Change the refiner from single-start to diagnostic multi-start for hard cases.
3. Add a `refiner_hurt` warning if the refined solution is far from the NN mean but only marginally improves spectral MAE.
4. Retrain with a harder curriculum around thin films and low-fringe spectra.
5. Add a second output head for uncertainty or ambiguity classification.
6. Add wavelength-dependent silicon substrate optical constants if aiming for physical realism beyond the synthetic toy stack.
