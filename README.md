# Neural Thin-Film Metrology

A physics-informed neural network system that inverts optical reflectance spectra to recover thin-film parameters: thickness, refractive index (n), and extinction coefficient (k). In semiconductor manufacturing, every deposited film must be measured to verify it meets spec. Spectral reflectometry is fast and non-contact, but extracting physical parameters from the measured spectrum is an ill-posed inverse problem where multiple parameter combinations can produce nearly identical spectra. This project solves that problem by combining a neural network for fast global estimation, a differentiable Transfer Matrix Method for physics-constrained training, Monte Carlo Dropout for uncertainty quantification, a physics-aware denoiser for noise robustness, and an L-BFGS-B optimizer for final precision. The result is a complete inference pipeline that runs in under 100ms per sample with sub-3nm thickness accuracy and calibrated confidence intervals.

## Live Demo

**[https://kla-neural-jaanav.streamlit.app/](https://kla-neural-jaanav.streamlit.app/)**

The interactive app includes a forward TMM simulator, an inverse predictor with uncertainty estimates and spectral refinement, and a technical report on model development.

## How It Works

### 1. TMM Forward Simulator

The Transfer Matrix Method computes reflectance spectra from first principles for a single thin film on a silicon substrate at normal incidence. The implementation exists in three forms: a scalar NumPy version for single samples, a vectorized NumPy batch version that processes thousands of spectra simultaneously via broadcasting, and a fully differentiable PyTorch version where gradients flow through the complex-valued matrix operations. The PyTorch variant is the key enabler for the physics-informed training loss.

### 2. Synthetic Dataset (500k Stratified Samples)

The training dataset contains 500,000 simulated reflectance spectra. Rather than sampling parameters purely at random, the dataset uses stratified sampling: 60% uniform random, 20% thickness-stratified (equal representation across 10 thickness bins from 10 to 300nm), and 20% n-stratified (equal representation across 10 refractive index bins from 1.3 to 2.5). This ensures thin films and extreme refractive indices are well-represented rather than undersampled, which proved more impactful than any architecture change.

### 3. SpectraNet V4 (Differentiable Physics Loss)

SpectraNet is a 275k-parameter MLP with hidden layers [512, 256, 128, 64], ReLU activations, Dropout(0.2), and a Sigmoid output that maps to normalized parameter space. The V4 training loss combines standard parameter MSE with a physics term: predicted parameters are fed back through the differentiable PyTorch TMM to re-simulate the spectrum, and the MSE between that re-simulation and the input spectrum is added to the loss with gradients flowing end-to-end. This forces the network to learn physically self-consistent mappings rather than just statistical correlations, and is most impactful on refractive index and extinction coefficient where the physics constraint resolves ambiguities that pure data fitting cannot.

### 4. MC Dropout Uncertainty

At inference time, dropout layers remain active across 100 stochastic forward passes (Monte Carlo Dropout). The spread of predictions across these passes provides calibrated 95% confidence intervals for each parameter. Wide intervals correctly flag cases where the spectrum is ambiguous, such as very thin films where the n-thickness degeneracy makes the inverse problem fundamentally ill-posed.

### 5. Physics-Aware Joint Denoiser

A denoising autoencoder (encoder-bottleneck-decoder architecture, 200 to 32 to 200 dimensions with Sigmoid output) is trained jointly with a frozen copy of SpectraNet V4 in the loop. The loss combines spectral reconstruction MSE with an inversion MSE that measures how accurately the denoised spectrum produces correct parameter predictions. This makes the denoiser preserve spectral features that matter for parameter recovery rather than simply smoothing noise. On out-of-distribution test data at noise std=0.010, the joint denoiser recovers approximately 30% of noise-induced prediction error compared to approximately 8% for a standard reconstruction-only denoiser.

### 6. L-BFGS-B Spectral Refiner

The neural network prediction serves as the initial guess for a box-constrained L-BFGS-B optimizer that minimizes the MSE between the TMM-simulated spectrum and the original measured spectrum. The refiner always operates on the raw input spectrum (before denoising) because the denoiser may subtly reshape spectral features, and the refiner's job is to find parameters that match the actual measurement exactly. This typically converges in 20-30 iterations and improves the spectral residual by 80-90% on clean inputs.

## Results

| Version | Description | Thickness MAE (nm) | n R² | k R² | Overall R² |
|---------|-------------|--------------------:|-----:|-----:|-----------:|
| V1 | 100k uniform, baseline MLP | 3.65 | 0.871 | 0.992 | 0.952 |
| V3 | 500k stratified dataset | 2.79 | 0.915 | 0.990 | 0.967 |
| V4 | 500k + differentiable physics loss | 2.62 | 0.930 | 0.996 | 0.975 |

| Denoiser | Noise Recovery (std=0.010, OOD) |
|----------|--------------------------------:|
| Standard (reconstruction only) | ~8% of noise-induced error |
| Joint (with frozen SpectraNet) | ~30% of noise-induced error |

## Key Findings

Data quality matters more than architecture complexity. The jump from V1 to V3 (R² 0.952 to 0.967) came entirely from increasing the dataset size from 100k to 500k with stratified sampling. Intermediate experiments with deeper networks, BatchNorm, and residual connections (V2) all performed worse than the simple V1 architecture on the same data. The lesson is that a simple model on good data beats a complex model on mediocre data.

The differentiable physics loss is qualitatively different from non-differentiable alternatives. Early attempts used a NumPy-based TMM in the loss function, which required detaching gradients and produced no training signal. The PyTorch reimplementation allows gradients to flow from the re-simulated spectrum through the TMM math back to the network weights. This acts as a strong inductive bias: the network cannot settle on parameter combinations that minimize training MSE but violate thin-film physics.

Refractive index is harder to predict than thickness or extinction coefficient. Thickness controls fringe count and spacing, producing a strong and unambiguous spectral signature. Extinction coefficient controls amplitude damping, which is also distinctive. Refractive index affects fringe contrast and subtle shifts in position, but these effects overlap with thickness changes, creating a fundamental degeneracy. The physics loss helps most here (n R² improved from 0.915 to 0.930) because it penalizes parameter combinations that are statistically plausible but physically inconsistent.

The hybrid neural network plus optimizer pipeline outperforms either approach alone. The neural network provides a fast, globally-informed initial guess that avoids the local minima that trap optimization-only methods. The L-BFGS-B refiner then exploits exact TMM physics to achieve precision that the network alone cannot reach. Together they deliver the speed of neural inference (under 1ms) with the accuracy of iterative optimization (under 100ms total).

## Limitations

The n-thickness degeneracy is a fundamental physical limitation at low film thickness (below 50nm). When fewer than one interference fringe appears in the visible wavelength range, the spectrum becomes a smooth curve that multiple (thickness, n) pairs can fit equally well. The model correctly flags these cases with wide uncertainty intervals, but the point predictions may be unreliable. This is not a model deficiency; it reflects genuine information loss in the measurement.

The extinction coefficient is never predicted as exactly zero. Because the training distribution samples k uniformly from 0.0 to 0.5, the network learns a continuous mapping and floors at approximately 0.003 to 0.005 for truly transparent films. A classification head could address this but was not implemented. At high noise levels (std above 0.015), spectral features are irreversibly corrupted and the denoiser cannot recover destroyed information. The model's uncertainty estimates correctly widen in this regime.

This project models a single thin film on a silicon substrate at normal incidence. Real semiconductor stacks have multiple layers, and production tools measure at multiple angles with polarization sensitivity. Extending to multi-layer stacks and oblique incidence is straightforward in the TMM formalism but dramatically increases the dimensionality of the inverse problem. Spectroscopic ellipsometry, which measures polarization change rather than intensity alone, is the standard approach for breaking the remaining degeneracies in production settings.

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI: forward simulator, inverse predictor, model performance report |
| `tmm_simulator.py` | TMM implementations: scalar NumPy, vectorized batch, differentiable PyTorch |
| `refiner.py` | L-BFGS-B spectral residual optimizer with box constraints |
| `denoiser.py` | Standard denoising autoencoder with OOD evaluation |
| `denoiser_joint.py` | Physics-aware joint denoiser trained with frozen SpectraNet |
| `train.py` | SpectraNet class definition and V1 training script |
| `train_v2.py` | V2: BatchNorm, residual connections, consistency regularization |
| `train_v3.py` | V3: V1 architecture on 500k stratified dataset |
| `train_v4.py` | V4: differentiable physics-informed loss via PyTorch TMM |
| `dataset_generator.py` | 100k uniform random dataset generator |
| `dataset_generator_v2.py` | 500k stratified dataset generator |
| `save_norm_stats.py` | Standalone script to regenerate normalization statistics |
| `spectranet_v4.pt` | Best model weights (V4, 275k parameters) |
| `denoiser.pt` | Standard denoiser weights |
| `denoiser_joint.pt` | Joint denoiser weights |
| `spectra_norm_v4.npz` | Input normalization statistics (mean, std from training set) |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `spectranet_v4.pt`, `denoiser_joint.pt`, and `spectra_norm_v4.npz` at startup. Training scripts and dataset files are not required for the UI.

## Background

The Transfer Matrix Method is the standard analytical framework for computing optical properties of thin-film stacks. It models light propagation through layered media using 2x2 complex matrices that encode phase accumulation and interface reflections, producing exact reflectance spectra from physical parameters. This is the same physics underlying KLA's SpectraFilm and SpectraShape product lines, where spectral measurements on production wafers must be inverted into film thicknesses and optical constants at high throughput. This project demonstrates that neural inversion with differentiable physics constraints can approach the accuracy of traditional model-based fitting while running orders of magnitude faster.
