# Neural Thin-Film Metrology

A physics-informed neural network that inverts optical reflectance spectra to recover thin-film parameters (thickness, refractive index n, extinction coefficient k).

## What it does

Given a reflectance spectrum measured from a thin film on a silicon substrate, this tool predicts the film's physical properties in milliseconds. It combines a neural network for fast initial prediction with MC Dropout uncertainty quantification and an optional L-BFGS-B spectral optimizer that refines results to near-exact accuracy. An interactive Streamlit UI lets you explore the forward physics (Transfer Matrix Method), run inverse predictions, and inspect model performance.

## How it works

- **TMM Simulator**: Transfer Matrix Method computes reflectance spectra from film parameters at normal incidence (air / thin film / silicon substrate), implemented in both NumPy and a differentiable PyTorch version.
- **Dataset Generation**: 500k synthetic spectra with stratified sampling across thickness and refractive index ranges.
- **SpectraNet Architecture**: MLP with layers [512, 256, 128, 64], ReLU activations, Dropout(0.2), and Sigmoid output mapping to normalized parameter space.
- **Differentiable Physics Loss (V4)**: The PyTorch TMM simulator enables a physics-informed loss term — predicted parameters are re-simulated and compared against the input spectrum, with gradients flowing end-to-end through the TMM math.
- **MC Dropout Uncertainty**: At inference, dropout stays active across 100 stochastic forward passes to estimate prediction confidence intervals.
- **Spectral Refiner**: L-BFGS-B optimizer uses the NN prediction as initial guess and minimizes the spectral residual with physical box constraints.

## Results

| Version | Description | Thickness MAE (nm) | n R² | k R² | Overall R² |
|---------|-------------|--------------------:|-----:|-----:|-----------:|
| V1 | 100k uniform, baseline | 3.65 | 0.871 | 0.992 | 0.952 |
| V3 | 500k stratified | 2.79 | 0.915 | 0.990 | 0.967 |
| V4 | 500k + physics loss | 2.62 | 0.930 | 0.996 | 0.975 |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI — forward simulator, inverse predictor, model performance |
| `tmm_simulator.py` | Transfer Matrix Method: single, batch (NumPy), and differentiable (PyTorch) |
| `refiner.py` | L-BFGS-B spectral residual optimizer for post-NN refinement |
| `train.py` | SpectraNet class definition and V1 training script |
| `train_v2.py` | V2: BatchNorm + residual + consistency regularization |
| `train_v3.py` | V3: V1 architecture on 500k stratified dataset |
| `train_v4.py` | V4: differentiable physics-informed loss |
| `dataset_generator.py` | 100k uniform random dataset generator |
| `dataset_generator_v2.py` | 500k stratified dataset generator |
| `spectranet_v4.pt` | Best model weights (V4) |
| `spectra_norm_v4.npz` | Input normalization statistics (mean, std) |
