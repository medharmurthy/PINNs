# Plain SIR (USING INVERSE PINNS)

 SIR — classical SIR assumes:
 
dS/dt = -beta * S * I

dI/dt = +beta * S * I - gamma * I

dR/dt = +gamma * I

 with beta and gamma as exactly TWO SINGLE SCALAR CONSTANTS for the entire
beta and gamma are nn.Parameter() scalars: two numbers, full stop.

The neural network ONLY maps (x, y, t) -> (S, I, R). No covariates (stringency_index, vaccination_rate, mobility_score, etc.) are used anywhere, neither as network inputs nor as physics-loss modulators.

The PINN's job is the classical "inverse problem": given noisy/partial S, I, R observations, find the single best-fit (beta, gamma) pair and the underlying smooth S(t), I(t), R(t) curves that satisfy the ODE.

Disadvantages:

* 550 districts and 991 days fit to ONE
global beta and ONE global gamma, the fit will be a national-average
compromise - it will not capture per-district heterogeneity.

*Physics Data loss is very small with respect to the data loss so model just tries to reduce the data loss greedily and achieves a good rmse and mse  score.

