# -*- coding: utf-8 -*-
"""
main.py — entry point for the SEIR PINN pipeline (Kaggle version, fp32, no AMP).

This package was split out of a single monolithic script into:
  config.py            paths, device setup, all hyperparameters
  data_preprocessing.py data loading, feature engineering, train/val/test
                        split, tensor prep, collocation sampling
  models.py             StateNet, ParamNet, FourierFeatures, SEIR residuals
  training.py           loss, checkpointing, diagnostics, training loop
  rk4_simulator.py       post-training RK4 rollout evaluation
  main.py (this file)    orchestration + the full modification history below

Full modification history (mods #1-#16), preserved here as the single
source of truth rather than duplicated across every file:

Modifications for Colab (original):
 1. Added Google Drive mounting.
 2. Checkpoints and CSV outputs are routed directly to Google Drive.
 3. Checkpointing is implemented at the end of every epoch.
 4. Plots an RK4 snapshot on the train set every 500 epochs during training.

Modification (fp32):
 5. Removed torch.cuda.amp (autocast + GradScaler) entirely. The loss
    formulation here (relative errors dividing by targets that can be as
    small as ~1e-6, combined with 1e5-scale loss weights) was overflowing
    fp16's representable range under autocast. GradScaler silently skips
    optimizer.step() whenever it detects an inf/nan during unscale_(), and
    if that happens every epoch you get a training loop that runs, prints,
    and checkpoints normally while never actually updating any weights.
    Training now runs fully in fp32: forward, backward, and the optimizer
    step are unconditional, with no scaler in the loop.

Modification (ParamNet optimization, superseded by #7 below):
 6. Split the single Adam optimizer / single gradient-clip call into two
    separate parameter groups: one for the ParamNet(s), one for the
    per-domain state_nets (see training.train_district). ParamNet
    parameters and StateNet parameters get their own learning rate and
    their own clip_grad_norm_ call.

Modification (per-domain ParamNet):
 7. ParamNet is no longer a single network shared across every domain.
    Since the RK4 rollout being fit here is only ever evaluated on the
    train set itself (each domain sees its own p_c/p_d rows and nothing
    else), there's no cross-domain generalization requirement forcing a
    shared beta/sigma/gamma mapping — each ~30-day domain now gets its own
    ParamNet, mirroring the existing per-domain StateNet setup. This is a
    strictly more expressive setup at the cost of losing any
    parameter-sharing regularization across domains. All ParamNet
    parameters across all domains are still pooled into one optimizer
    parameter group (with PARAM_NET_LR) and clipped together as a group,
    separately from the pooled StateNet parameters.

BUGFIX (rk4 weight ramp — superseded by #9 below):
 8. Previous behavior: at the single epoch == WARMUP_EPOCHS boundary,
    weights['ode']/['alg'] jumped 0.0 -> 1.0 AND weights['rk4'] jumped
    0.0 -> 1e5 in the same instant param_nets were unfrozen — causing a
    training stall right at epoch 1000. Fix at the time: only 'rk4' was
    changed to ramp linearly from 0.0 to RK4_TARGET_WEIGHT over
    RK4_RAMP_EPOCHS instead of jumping instantly.

REMOVAL (rk4 loss):
 9. The RK4-rollout loss term has been removed entirely: the 'rk4' loss
    weight, its ramp schedule, loss_rk4 inside compute_loss, and the
    train-set RK4 diagnostic logging/plotting that ran alongside it during
    training. param_nets are still unfrozen at epoch == WARMUP_EPOCHS,
    because loss_ode still needs a live gradient path into param_net —
    that requirement was never specific to RK4. Total training loss is now
    just data + ode + alg + boundary. The post-training RK4 simulation
    (rk4_simulator.py) is untouched by this removal — it's a separate,
    post-hoc evaluation, not part of the training loss.

BUGFIX (multi-GPU device mismatch in post-training RK4 rollout):
 10. rk4_simulator.query_param_net was raising a device-mismatch
    RuntimeError under NUM_GPUS >= 2, because different domains' ParamNets
    can live on different physical GPUs, but the input tensor was being
    built on a single global `device` argument rather than param_net's own
    device. Fix: build the input tensor on param_net's own device (via
    _net_device), resolved after _select_domain_model picks the correct
    per-domain ParamNet.

ADDITION (post-training diagnostics):
 12. Three read-only reporting functions run once at the end of
    train_district (training.py), after the epoch loop: report_learned_params
    (per-domain beta/sigma/gamma mean±std), report_state_net_fit (StateNet's
    direct MSE/MAE fit, no ODE integration), and plot_state_net_fit (the
    plotting counterpart). These isolate "does StateNet fit the data" from
    "is the learned ODE/rollout consistent."

BUGFIX (ode/alg weight ramp):
 11. param_net's only gradient path to beta/sigma/gamma runs through
    loss_ode/loss_alg. Those jumped to weight 1.0 at WARMUP_EPOCHS and
    stayed there, while data/boundary sit at 1e5. clip_grad_norm_(max_norm=1.0)
    only caps gradients already above 1.0 — with ode/alg stuck at weight
    1.0, param_net's raw gradient norm sat well under that threshold, so
    its higher LR (PARAM_NET_LR) mostly multiplied a near-zero gradient by
    a bigger number. Net effect: param_net barely trained. Fix: ramp
    weights['ode']/['alg'] linearly from 1.0 up to ODE_ALG_TARGET_WEIGHT
    over ODE_ALG_RAMP_EPOCHS (mirroring the now-removed rk4 ramp pattern),
    deliberately kept below the 1e5 data/boundary weight.

BUGFIX (ParamNet output saturation — Sigmoid -> Softplus):
 13. Post-ramp learned beta/gamma were still pinned at exactly the min/max
    of their sigmoid-scaled range, with near-zero within-domain std despite
    genuinely varying contextual features — the signature of a saturated
    sigmoid (once a gradient pushes a pre-activation into the flat tail,
    the local gradient collapses toward 0 and the unit gets stuck). Fix:
    replace ParamNet's final nn.Sigmoid() with nn.Softplus() (models.py),
    which has no flat, exactly-zero-gradient region. The *2.0/*0.5/*0.5
    scale multipliers are unchanged, now functioning as soft scale factors
    rather than hard ceilings.

BUGFIX (post-training RK4 rollout was leaking the true trajectory into its
own "forecast"):
 14. simulate_district_rk4 (rk4_simulator.py) correctly starts from a
    single initial condition (StateNet's prediction at t0) and propagates
    state purely through rk4_step + seir_rhs. The bug was in what
    query_param_net fed ParamNet at every RK4 stage: build_param_interpolators
    built np.interp lookups straight from the true, observed per-district
    series for ALL param_features, including velocity/acceleration/
    regime_signal/spike_signal — all four derived directly from the true I
    curve. So at every RK4 stage, ParamNet was told what the true infection
    curve was actually doing at that exact time — a running answer key
    baked into the rollout's own input features. Fix: split param_features
    into EXO_PARAM_FEATURES (x, y, stringency_smooth, pop_density — read
    from the true series via interpolators, since these are genuinely
    exogenous) and TRAJ_PARAM_FEATURES (velocity, acceleration,
    regime_signal, spike_signal — produced online by a new
    OnlineFeatureTracker, seeded with the single initial (t0, I0) and
    updated only from the simulated trajectory, never the true I series).

ADDITION (Fourier features on ParamNet's input only):
 15. StateNet is untouched. ParamNet's raw 8-dim contextual input vector is
    now passed through a random Fourier feature encoding (Tancik et al.,
    2020) before the first Linear layer (models.FourierFeatures), to
    counter the spectral bias plain Tanh-MLPs have toward low-frequency
    functions. The random frequency matrix B is a fixed, non-trainable
    buffer, sampled once per ParamNet instance (so each per-domain ParamNet
    gets its own independently-sampled B). forward() concatenates the raw
    input alongside sin/cos of the projection, not replacing it.

BUGFIX (train/inference feature mismatch behind post-mod-#14 RK4 rollout
drift):
 16. After mod #14 removed the true-trajectory leak, the rollout's I/S/R
    curves showed a sustained drift plus a high-frequency wobble in I.
    Diagnosis: OnlineFeatureTracker computed velocity/acceleration as a raw
    one-step backward difference of the SIMULATED I, refreshed every fine
    RK4 substep — but calculate_epidemic_dynamics (data_preprocessing.py)
    computed ParamNet's training-time velocity/acceleration as a 7-point
    CENTERED-rolling-median-smoothed signal, at one row per real daily
    observation. Even with the leak gone, ParamNet's rollout-time inputs
    came from a different distribution than its training-time inputs,
    which fed back into a closed loop (noisy feature -> shifted params ->
    perturbed next I -> noisier feature) that produced the drift/wobble.
    Fix: (a) data_preprocessing.calculate_epidemic_dynamics' velocity/
    acceleration smoothing changed from centered to TRAILING rolling
    median — a genuine preprocessing change requiring ParamNet retraining;
    (b) rk4_simulator.compute_regime_thresholds mirrors the same trailing
    change for consistency; (c) OnlineFeatureTracker reworked to maintain
    trailing deques of raw daily velocity/acceleration and report their
    median, reproducing the trailing rolling median causally, point for
    point; (d) simulate_district_rk4's loop restructured around real
    observation intervals so the tracker updates once per real day (not
    per fine substep), matching training's feature cadence exactly.

Minor cleanup made during the file split: unused imports carried over from
the original notebook (IPython.display, sys, random, torch.utils.checkpoint,
torch.nn.functional, matplotlib.patches, matplotlib.gridspec) were dropped,
since none of them were referenced anywhere in the pipeline.
"""

import numpy as np

import config
from data_preprocessing import load_and_preprocess_data
from rk4_simulator import run_rk4_simulations_for_district
from training import train_single_district


def main():
    # ── Load & preprocess data ──────────────────────────────────────────
    df_dataset, df_train, df_val, df_test, T_DAYS = load_and_preprocess_data()

    # N_DOMAINS is data-dependent (ceil(T_DAYS / SUBDOMAIN_LENGTH_DAYS)), so
    # it's set here, after loading data, rather than being a static config
    # constant. Every downstream module reads it via `config.N_DOMAINS`
    # (not a value captured at import time), so this assignment is visible
    # to training.train_district and anywhere else that needs it.
    config.T_DAYS = T_DAYS
    config.N_DOMAINS = max(1, int(np.ceil(T_DAYS / config.SUBDOMAIN_LENGTH_DAYS)))

    # ── Pick target district ────────────────────────────────────────────
    target_district = config.TARGET_DISTRICT
    if target_district is None:
        candidate_keys = df_dataset['location_key'].unique()
        target_district = next(
            dk for dk in candidate_keys
            if len(df_dataset[df_dataset['location_key'] == dk]) >= config.MIN_ROWS_PER_DISTRICT
        )

    # ── Train ────────────────────────────────────────────────────────────
    all_district_models = train_single_district(df_dataset, target_district, df_val=df_val)

    # ── Post-training RK4 rollout evaluation ────────────────────────────
    rk4_results = run_rk4_simulations_for_district(
        target_district, all_district_models, df_train, df_val, df_test,
        df_dataset, config.DATA_COLUMNS['param_features'], config.device,
        splits=('Train', 'Test')
    )
    return rk4_results


if __name__ == "__main__":
    main()
