# PINNs for COVID-19 in India

Modelling how COVID-19 spread across Indian districts using physics-informed neural networks (PINNs).

This is Phase 1 of the IITG.AI project. The basic question: can we take messy district-level case data and back out the *hidden* epidemic parameters (how fast it spreads, how fast people recover, how many die) as things that change over time and place, instead of assuming they're constant for the whole country?

We tried three different models to get at this. None of them is perfect, and each one is good at something the others aren't. The full write-up, with all the plots and numbers, is in `PINNs_COVID19_Unified_Phase1_Report.pdf`.

---

## The idea in two paragraphs

Classic compartmental models (SIR, SEIR, SIRD) split a population into buckets: Susceptible, Exposed, Infectious, Recovered, Deceased. A few ODEs describe how people move between them. Usually you fit these with fixed rates. That falls apart for a country like India, where 700+ districts had outbreaks at different times, under different lockdowns, with different mobility and vaccination.

A PINN gets around this by training a network on two things at once:

1. Fit the observed data.
2. Obey the ODEs.

Because both losses share the same weights, the rates the network learns have to explain the data *and* make physical sense. At least, that's the theory. Most of what we learned in Phase 1 is about where that theory breaks down.

---

## The three models

### 1. Unified SIRD-PINN (district level)

One network with a shared trunk and two heads:

- **State head**: outputs S, I, R, D through a softmax, so they always add up to exactly 1.
- **Rate head**: outputs β, γ, μ through Softplus, so they're always positive and don't saturate the way Sigmoid does.

It covers 593 districts (587,663 rows). Districts talk to each other through a k-nearest-neighbour "neighbour pressure" term weighted by mobility. An earlier version used a Laplacian diffusion term, which treated infection like heat spreading through metal and didn't make much sense.

Things worth knowing:

- Train/validation is split **by district** (474 / 119), not by row. The first version split by row, which leaked neighbouring days of the same district into validation and made every number meaningless.
- There was a nasty chain-rule bug. Time was normalised twice, so the learned rates came out ~495× too big. Fixed by scaling the autograd time derivatives by 2/990.
- On unseen districts, direct predictions are good: R² of 0.97 / 0.79 / 0.96 / 0.98 for S / I / R / D.
- **But** if you throw away the state head and just integrate the learned rates forward with RK4, it blows up within ~15% of the timeline. This is the big open problem.

### 2. Domain-Decomposed SEIR-PINN (per district)

Each district's timeline is chopped into ~30-day windows. Every window gets its own pair of networks:

- **StateNet**: maps (t, x, y) to (S, E, I, R).
- **ParamNet**: maps 8 context features (stringency, density, case velocity/acceleration, regime signals, and so on) to (β, σ, γ).

Neighbouring windows overlap by 10%, with a continuity loss so the curve doesn't jump at the edges. Collocation points are biased toward the parts of the curve that change fastest.

This one went through a lot of debugging. The report lists all 13 changes. The highlights:

- fp16 mixed precision was silently skipping every optimizer step. Switched to fp32.
- ParamNet was barely training because its gradients never got past the clip threshold. Fixed by ramping up the physics loss weight.
- Sigmoid on the rate outputs got stuck at its limits. Switched to Softplus.
- A multi-GPU device mismatch only showed up when the RK4 rollout crossed a window boundary.

Results (shown on Ernakulam): the state fit is near-perfect, and the RK4 rollout actually follows the timing of the infection waves, which is better than the other residual-based model. The catch is that the learned rates are often way too large (β and γ up to ~60/day) and jump around between windows.

### 3. RK4-SEIRD (national)

This one ditches the standard PINN setup entirely. There's no state network. A small MLP predicts time-varying β, σ, γ, μ from time, Fourier features and 9 covariates. Those rates go straight into a **differentiable RK4 solver**, and we backprop through the solver.

The upside: every trajectory it produces obeys SEIRD exactly, by construction. It can't cheat by fitting the data while ignoring the physics.

It took four tries to get here:

1. Plain SIR with one global β and γ. It fit a national average and missed every peak.
2. Added Fourier features and per-district peak weighting. Better, but the RK4 check still failed.
3. SIRD with rates driven by covariates. RK4 rollout stayed flat, so no epidemic at all.
4. SEIRD with the solver inside the loss. This one worked.

The final loss is a variance-weighted MSE on I, R and D plus a smoothness penalty on the rates. Loss dropped ~95% over 3,000 epochs (~2.8 hours), and the simulated curve tracks the national waves. The downside is that it only works on the national average, so all the district-level variation is gone.

---

## Quick comparison

|                     | Unified SIRD-PINN             | Domain-Decomposed SEIR-PINN   | RK4-SEIRD                  |
| ------------------- | ----------------------------- | ----------------------------- | -------------------------- |
| Scope               | 593 districts                 | one district at a time        | national average           |
| How physics is used | residual loss (soft)          | residual + boundary (soft)    | ODE solver (exact)         |
| Direct fit          | very good                     | very good                     | good                       |
| RK4 rollout         | diverges                      | partly works                  | works (it *is* the model)  |
| Main problem        | rates don't hold up under RK4 | rates too big, jumpy          | no district detail         |

---

## Things we learned the hard way

- **A good fit doesn't mean good physics.** Every model with a separate state network fit the data nicely while its rates were nonsense when integrated. Always run the RK4 check.
- **Watch your time scaling.** If you normalise time, your derivatives are in normalised units, not days. Two of the three models hit this independently.
- **Don't use Sigmoid for rate outputs.** It saturates and gets stuck. Softplus is the safer choice.
- **Fourier features help on time, hurt on space.** District coordinates are basically IDs, not a smooth field.
- **I and D are tiny compared to S.** Weight the losses, or the model will happily ignore them.
- **Split by district or by time, never by random row.**

---

## Data

`india_processed_final.csv` is Google's COVID-19 Open Data joined with Oxford stringency and Google/Apple mobility indices. It's daily and district-level, with lat/long, cumulative cases, recoveries, deaths, and covariates like stringency, mobility, population density, vaccination and testing.

Two gotchas we found:

- `stringency_index` is all zeros. Use `stringency_index_alternate`.
- `growth_rate` has some `inf` values.

---

## Running things

The work lives in Jupyter notebooks, mostly run on Colab and Kaggle. You'll need roughly:

```
python >= 3.9
torch
numpy
pandas
scikit-learn
matplotlib
pywavelets   # only for the SEIR-PINN spike detector
```

Point the data path in each notebook at your copy of `india_processed_final.csv` and run top to bottom. Some tips:

- The Unified SIRD-PINN trains in ~48 minutes on CPU (60 epochs).
- The SEIR-PINN really wants a GPU and supports multiple GPUs. 7,000 epochs per district isn't quick.
- The RK4-SEIRD model runs for about 2.8 hours for 3,000 epochs.

---

## What's next

The obvious move is to combine the three: use the differentiable-solver core from RK4-SEIRD, run it per district on the Unified model's leak-free pipeline and neighbour coupling, and borrow the time-window decomposition from the SEIR-PINN. Other things on the list:

- Tie neighbouring districts' and neighbouring windows' rates together so they can't drift apart.
- Report every rate in per-day units and sanity-check it against known values (recovery in roughly 1–2 weeks, not hours).
- Evaluate all models on the same held-out districts with the same metrics.
- Test whether β actually responds to lockdown stringency at the district level, since the national average washes it out.
- Swap the k-NN graph for real mobility flow data.

<img width="1178" height="4422" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/28fac813-8d57-4028-b185-607ec6883e93" />
