# Plain SIR (USING INVERSE PINNS)

 SIR — classical SIR assumes:
 
dS/dt = -beta * S * I

dI/dt = +beta * S * I - gamma * I

dR/dt = +gamma * I

 with beta and gamma as exactly TWO SINGLE SCALAR CONSTANTS for the entire
beta and gamma are nn.Parameter() scalars: two numbers, full stop.

**APPROACH 1**

The neural network ONLY maps (x, y, t) -> (S, I, R). No covariates (stringency_index, vaccination_rate, mobility_score, etc.) are used anywhere, neither as network inputs nor as physics-loss modulators.

The PINN's job is the classical "inverse problem": given noisy/partial S, I, R observations, find the single best-fit (beta, gamma) pair and the underlying smooth S(t), I(t), R(t) curves that satisfy the ODE.

Disadvantages:

* 550 districts and 991 days fit to ONE
global beta and ONE global gamma, the fit will be a national-average
compromise - it will not capture per-district heterogeneity.

*Physics Data loss is very small with respect to the data loss so model just tries to reduce the data loss greedily and achieves a good rmse and mse  score.

**APPROACH 2**

We improve the network's ability to fit the epidemic curves through a few key upgrades: 

Fourier Features: We introduced a TimeFourierFeatures layer that projects the time variable t into sine and cosine transformations, which drastically helps the network resolve the shape of the time series.   

Increased Training Loops: We increased the training time to 17,000 epochs to allow the model to fully converge with the new architecture. 

Peak Weighting: We applied a dynamic weight scale to prioritize rows of data near each district's infection peak.

RK4 INTRODUCTION:

CLASSICAL SIR ODE SOLUTION USING THE LEARNED (beta, gamma)

Sanity check: solve the plain ODE system directly (RK4, no neural network)
using the learned beta/gamma and compare to the average data trend. 

This confirms beta/gamma are behaving like genuine SIR parameters and not just absorbing fitting error.
