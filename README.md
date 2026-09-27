# SEIRD PINN-RK4

This  fits a national-level SEIRD model using a neural network to learn time-varying epidemiological parameters.

The model uses time, Fourier features, and nine COVID-related covariates such as vaccination, testing, mobility, stringency, and infection lags. A small neural network learns **β, σ, γ, and μ**, which are then used in the SEIRD equations.

Instead of solving the equations directly with a basic numerical update, the model uses **RK4** for the SEIRD rollout. Training minimizes the difference between simulated and observed \(I, R,\) and \(D\), while a smoothness term keeps the learned parameters from changing too abruptly.

**ADVANTAGES:**
1. The model learns: beta ,gamma, u,sigma as a function of time.

2. The model is enforced to learn the real physics by fitting onto the RK4 equation .Model here cannot depend on the data loss. This is a stronger form of physics incorporation.

The model cannot arbitrarily invent a curve that violates the SEIRD dynamics, because is produced by the numerical integration of the SEIRD equations

### Main settings
- 26 input features
- 2 hidden layers, 64 neurons each
- Tanh activation
- Adam optimizer, learning rate 0.002
- 3000 training epochs
- RK4 solver
- Smoothness weight = 5
- Gradient clipping = 5


The code also saves the fitted SEIRD trajectories, learned parameters, and training-loss plot for analysis.
