# -*- coding: utf-8 -*-
"""
rk4_simulator.py — post-training RK4 rollout evaluation using the learned
ParamNet(s): a separate, post-hoc evaluation, not part of the training loss
(mod #9 removed the RK4 term from training entirely; everything here is
diagnostic).

This file is where mods #10, #14, and #16 live (see main.py's module
docstring for the full history):
  - mod #10: query_param_net builds its input tensor on param_net's own
    device (multi-GPU device-mismatch fix).
  - mod #14: EXO_PARAM_FEATURES (x, y, stringency_smooth, pop_density) are
    read from the true series via interpolators; TRAJ_PARAM_FEATURES
    (velocity, acceleration, regime_signal, spike_signal) are produced
    online by OnlineFeatureTracker from the SIMULATED trajectory only —
    the true I curve is never read after t0. This is what makes the
    rollout a genuine forecast rather than a trajectory leak.
  - mod #16: OnlineFeatureTracker reproduces the exact same TRAILING (not
    centered) 7-point rolling-median smoothing that
    data_preprocessing.calculate_epidemic_dynamics now uses at training
    time, updated once per real observation interval (not per fine RK4
    substep) — closing the train/inference feature-distribution gap that
    caused rollout drift/wobble after mod #14 alone.

Do not change the trailing-window smoothing in data_preprocessing.py
without updating compute_regime_thresholds and OnlineFeatureTracker here to
match, or the mod #16 fix will silently regress.
"""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from models import _net_device


# ─────────────────────────────────────────────────────────────
# EXOGENOUS-FEATURE INTERPOLATORS (mod #14)
# ─────────────────────────────────────────────────────────────
def build_param_interpolators(df_dist_all, param_features=None):
    """Builds np.interp lookups ONLY for genuinely exogenous features
    (policy stringency, population density, static coordinates).
    Trajectory-derived features (velocity/acceleration/regime/spike) are
    intentionally excluded here — see OnlineFeatureTracker, which produces
    them causally from the simulated state instead of the true series."""
    param_features = param_features or config.EXO_PARAM_FEATURES
    df_sorted = df_dist_all.sort_values('t')
    t_vals = df_sorted['t'].values
    interpolators = {}
    for feat in param_features:
        f_vals = df_sorted[feat].values
        interpolators[feat] = lambda t_query, _t=t_vals, _f=f_vals: np.interp(t_query, _t, _f)
    return interpolators


def compute_regime_thresholds(df_dist_all, target_col=None):
    """Reproduces the near_zero_tol / spike_threshold / spike_scale
    normalization constants from data_preprocessing.calculate_epidemic_dynamics,
    so the online regime_signal/spike_signal computed during the RK4
    rollout use the same scale as what ParamNet was trained on. These three
    are scalar normalization constants for the district, not the trajectory
    itself, so reusing them here does not reintroduce the per-step
    trajectory leak that mod #14 fixes.

    mod #16: uses the same TRAILING (not centered) rolling median as
    data_preprocessing.calculate_epidemic_dynamics — must stay in lockstep
    with that function or acc_std (and therefore near_zero_tol/
    spike_threshold/spike_scale) would be computed on a different smoothing
    than what ParamNet was actually trained against."""
    target_col = target_col or config.TARGET_COL
    df_sorted = df_dist_all.sort_values('t')
    dt = df_sorted['t'].diff().replace(0, np.nan)
    d_target = df_sorted[target_col].diff()
    velocity = (d_target / dt).bfill().ffill().fillna(0)
    acceleration = (velocity.diff() / dt).bfill().ffill().fillna(0)
    velocity = velocity.rolling(7, min_periods=1).median()
    acceleration = acceleration.rolling(7, min_periods=1).median()
    acc_std = acceleration.std()
    near_zero_tol = 0.05 * (acc_std if acc_std > 0 else 1.0)
    spike_threshold = 2.0 * (acc_std if acc_std > 0 else 1.0)
    spike_scale = 0.1 * (spike_threshold if spike_threshold > 0 else 1.0)
    return near_zero_tol, spike_threshold, spike_scale


# ─────────────────────────────────────────────────────────────
# ONLINE TRAJECTORY-DERIVED FEATURE TRACKER (mods #14, #16)
# ─────────────────────────────────────────────────────────────
class OnlineFeatureTracker:
    """Produces velocity/acceleration/regime_signal/spike_signal causally
    from the SIMULATED trajectory only, so the RK4 rollout never reads the
    true I curve it is supposed to be forecasting (mod #14).

    mod #16: maintains a maxlen-7 trailing buffer of RAW daily velocity
    estimates and a maxlen-7 trailing buffer of RAW daily acceleration
    estimates (mirroring calculate_epidemic_dynamics' two-stage
    raw-then-smoothed computation exactly: acceleration is diff of RAW
    velocity, not diff of smoothed velocity), and reports the MEDIAN of
    each buffer — reproducing `rolling(7, min_periods=1).median()`
    causally, point for point, since a trailing window's value at row i
    only ever depends on rows <= i.

    update() should be called once per REAL daily observation (see
    simulate_district_rk4), not once per fine RK4 substep — matching the
    daily resolution calculate_epidemic_dynamics computed these features at
    during training. Values are frozen across every fine RK4 substep within
    a real day, exactly mirroring how one real day's feature row was a
    single fixed value throughout training.

    Seeded once with the single initial (t0, I0); everything after that is
    derived only from previously-simulated points, never the true series.
    """

    def __init__(self, near_zero_tol, spike_threshold, spike_scale, window=7):
        self.near_zero_tol = near_zero_tol
        self.spike_threshold = spike_threshold
        self.spike_scale = spike_scale
        self.t_hist = []
        self.I_hist = []
        self.raw_velocity_hist = deque(maxlen=window)
        self.raw_acceleration_hist = deque(maxlen=window)
        self.velocity = 0.0
        self.acceleration = 0.0

    def update(self, t, I):
        self.t_hist.append(t)
        self.I_hist.append(I)
        if len(self.t_hist) < 2:
            return
        dt = self.t_hist[-1] - self.t_hist[-2]
        if dt <= 0:
            return
        raw_velocity = (self.I_hist[-1] - self.I_hist[-2]) / dt
        self.raw_velocity_hist.append(raw_velocity)
        if len(self.raw_velocity_hist) >= 2:
            # Mirrors training: acceleration is the diff of RAW velocity
            # (not the already-smoothed velocity), divided by the same dt.
            raw_acceleration = (self.raw_velocity_hist[-1] - self.raw_velocity_hist[-2]) / dt
            self.raw_acceleration_hist.append(raw_acceleration)
        # Trailing median over up to `window` points — equivalent to
        # pandas .rolling(window, min_periods=1).median() at this row,
        # since a trailing window never needs points beyond the current one.
        self.velocity = float(np.median(self.raw_velocity_hist)) if self.raw_velocity_hist else 0.0
        self.acceleration = float(np.median(self.raw_acceleration_hist)) if self.raw_acceleration_hist else 0.0

    def current_features(self):
        regime_signal = np.tanh(self.acceleration / (self.near_zero_tol + 1e-8))
        spike_signal = 1.0 / (1.0 + np.exp(
            -(abs(self.acceleration) - self.spike_threshold) / (self.spike_scale + 1e-8)
        ))
        return {
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'regime_signal': regime_signal,
            'spike_signal': spike_signal,
        }


# ─────────────────────────────────────────────────────────────
# PARAMNET QUERY (mods #10, #14)
# ─────────────────────────────────────────────────────────────
def query_param_net(param_nets, domain_edges, interpolators, tracker, param_features, t_query, device):
    """mod #14: feature vector is assembled from two disjoint sources —
    config.EXO_PARAM_FEATURES via the true-series `interpolators` (fine,
    genuinely exogenous), and config.TRAJ_PARAM_FEATURES via `tracker`,
    which knows only about the simulated trajectory so far.

    `device` is accepted for call-signature compatibility with the rest of
    the RK4 code path but is intentionally unused here — per mod #10, the
    input tensor is built on param_net's own device, not the passed-in
    `device`, to avoid a multi-GPU device mismatch when different domains'
    ParamNets live on different physical GPUs."""
    param_net, _, _ = _select_domain_model(param_nets, domain_edges, t_query)
    param_device = _net_device(param_net)
    traj_feats = tracker.current_features()
    values = []
    for feat in param_features:
        if feat in config.TRAJ_PARAM_FEATURES:
            values.append(traj_feats[feat])
        else:
            values.append(interpolators[feat](t_query))
    feat_vec = np.array(values, dtype=np.float32)
    p_t = torch.tensor(feat_vec, dtype=torch.float32, device=param_device).view(1, -1)
    with torch.no_grad():
        beta, sigma, gamma = param_net(p_t).squeeze(0).cpu().numpy()
    return float(beta), float(sigma), float(gamma)


# ─────────────────────────────────────────────────────────────
# SEIR ODE + RK4 STEP
# ─────────────────────────────────────────────────────────────
def seir_rhs(state, beta, sigma, gamma):
    S, E, I, R = state
    dS = -beta * S * I
    dE = beta * S * I - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])


def rk4_step(state, t, dt, param_nets, domain_edges, interpolators, tracker, param_features, device):
    # mod #14: `tracker` is queried, not updated, at every stage here —
    # its velocity/acceleration are frozen at whatever the last COMPLETED
    # real interval set them to (see OnlineFeatureTracker). It is only
    # advanced once, by the caller (simulate_district_rk4), after a whole
    # real interval's substeps complete — never mid-interval, since the
    # true future I is not (and must not be) available at intermediate
    # query times.
    b1, s1, g1 = query_param_net(param_nets, domain_edges, interpolators, tracker, param_features, t, device)
    k1 = seir_rhs(state, b1, s1, g1)

    b2, s2, g2 = query_param_net(param_nets, domain_edges, interpolators, tracker, param_features, t + dt / 2, device)
    k2 = seir_rhs(state + dt / 2 * k1, b2, s2, g2)

    b3, s3, g3 = query_param_net(param_nets, domain_edges, interpolators, tracker, param_features, t + dt / 2, device)
    k3 = seir_rhs(state + dt / 2 * k2, b3, s3, g3)

    b4, s4, g4 = query_param_net(param_nets, domain_edges, interpolators, tracker, param_features, t + dt, device)
    k4 = seir_rhs(state + dt * k3, b4, s4, g4)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _select_domain_model(models, domain_edges, t_value):
    k = np.searchsorted(domain_edges, t_value, side='right') - 1
    k = int(np.clip(k, 0, len(models) - 1))
    return models[k], domain_edges[k], domain_edges[k + 1]


def get_initial_state(state_nets, domain_edges, t0, x0, y0, device):
    """The RK4 rollout's ONLY read of the trained model for its initial
    condition — a single point, from StateNet's own prediction (not the
    true observed S/I/R at t0). Everything after this comes from
    rk4_step's pure ODE integration."""
    state_net, t_start, t_end = _select_domain_model(state_nets, domain_edges, t0)
    k = state_nets.index(state_net)
    overlap_width = config.OVERLAP_FRAC * (t_end - t_start)
    t_start_overlap = max(float(domain_edges[0]), t_start - overlap_width) if k > 0 else t_start
    dt_real = t_end - t_start_overlap + 1e-8
    t_norm = 2.0 * (t0 - t_start_overlap) / dt_real - 1.0

    t_t = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
    x_t = torch.tensor([[x0]], dtype=torch.float32, device=device)
    y_t = torch.tensor([[y0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = state_net(t_t, x_t, y_t).cpu().numpy().squeeze()

    S0, E0, I0, R0 = out
    return np.array([S0, E0, I0, R0])


def simulate_district_rk4(district_key, bundle, df_dist_split, df_dist_all, param_features, device):
    state_nets = bundle['state_nets']
    param_nets = bundle['param_nets']
    domain_edges = bundle['domain_edges']

    df_dist_split = df_dist_split.sort_values('t')
    t_vals = df_dist_split['t'].values
    if len(t_vals) < 2:
        return None

    x0, y0 = float(df_dist_split.iloc[0]['x']), float(df_dist_split.iloc[0]['y'])
    # mod #14: interpolators cover only EXO_PARAM_FEATURES (truly exogenous
    # covariates); velocity/acceleration/regime/spike come from `tracker`
    # instead, seeded with the single initial (t0, I0) and advanced only
    # from the simulated trajectory from here on.
    interpolators = build_param_interpolators(df_dist_all, config.EXO_PARAM_FEATURES)
    thresholds = compute_regime_thresholds(df_dist_all)
    tracker = OnlineFeatureTracker(*thresholds)

    state = get_initial_state(state_nets, domain_edges, t_vals[0], x0, y0, device)
    tracker.update(t_vals[0], state[2])  # state = [S, E, I, R]; seed with I0 only

    n_intervals = max(1, len(t_vals) - 1)

    sim_t = [t_vals[0]]
    sim_states = [state.copy()]
    # mod #16: loop restructured around real observation intervals
    # (t_vals[i] -> t_vals[i+1]) rather than one flat pass over all fine
    # substeps with a single dataset-wide dt:
    #   (1) tracker.update() is called exactly once per real interval,
    #       after that interval's substeps complete — matching the daily
    #       resolution at which velocity/acceleration/regime_signal/
    #       spike_signal were computed during training. tracker.current_
    #       features() therefore returns the SAME values for every one of
    #       the RK4_STEPS_PER_INTERVAL substeps (and all 4 RK4 stages
    #       within each substep) inside a given real interval.
    #   (2) dt is computed per-interval from that interval's own real span
    #       rather than a single dataset-wide average — more accurate if
    #       real observations aren't perfectly evenly spaced.
    t_cur = t_vals[0]
    for i in range(n_intervals):
        t_start_i, t_end_i = t_vals[i], t_vals[i + 1]
        dt = (t_end_i - t_start_i) / config.RK4_STEPS_PER_INTERVAL
        for _ in range(config.RK4_STEPS_PER_INTERVAL):
            state = rk4_step(state, t_cur, dt, param_nets, domain_edges, interpolators, tracker, param_features, device)
            t_cur += dt
            sim_t.append(t_cur)
            sim_states.append(state.copy())
        # Advance the tracker only now, once this real interval's worth of
        # substeps is complete — not per fine substep.
        tracker.update(t_end_i, state[2])

    sim_t = np.array(sim_t)
    sim_states = np.array(sim_states)

    sim_S = np.interp(t_vals, sim_t, sim_states[:, 0])
    sim_E = np.interp(t_vals, sim_t, sim_states[:, 1])
    sim_I = np.interp(t_vals, sim_t, sim_states[:, 2])
    sim_R = np.interp(t_vals, sim_t, sim_states[:, 3])

    return {
        't': t_vals,
        'true_S': df_dist_split['S'].values, 'true_I': df_dist_split['I'].values, 'true_R': df_dist_split['R'].values,
        'sim_S': sim_S, 'sim_E': sim_E, 'sim_I': sim_I, 'sim_R': sim_R,
        'fine_t': sim_t, 'fine_states': sim_states,
    }


def plot_rk4_simulation(result, district_key, split_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ['S', 'I', 'R']
    true_keys = ['true_S', 'true_I', 'true_R']
    sim_keys = ['sim_S', 'sim_I', 'sim_R']
    for ax, label, tk, sk in zip(axes, labels, true_keys, sim_keys):
        ax.plot(result['t'], result[tk], 'o', ms=3, label='true')
        ax.plot(result['t'], result[sk], '-', label='RK4 sim (ParamNet)')
        ax.set_title(f"{district_key} — {label} ({split_name}, RK4 rollout)")
        ax.set_xlabel('t (normalized, dataset-wide)')
        ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def run_rk4_simulations_for_district(district_key, all_district_models, df_train, df_val, df_test,
                                      df_dataset, param_features, device, splits=('Train',)):
    if district_key not in all_district_models:
        print(f"No trained model found for '{district_key}'")
        return {}

    bundle = all_district_models[district_key]
    df_dist_all = df_dataset[df_dataset['location_key'] == district_key].sort_values('t')
    if df_dist_all.empty:
        return {}

    results = {}

    if 'Train' in splits:
        df_dist_train = df_train[df_train['location_key'] == district_key]
        if not df_dist_train.empty:
            res = simulate_district_rk4(district_key, bundle, df_dist_train, df_dist_all, param_features, device)
            if res is not None:
                results['Train'] = res
                plot_rk4_simulation(res, district_key, 'Train')

    if 'Validation' in splits:
        df_dist_val = df_val[df_val['location_key'] == district_key]
        if not df_dist_val.empty:
            res = simulate_district_rk4(district_key, bundle, df_dist_val, df_dist_all, param_features, device)
            if res is not None:
                results['Validation'] = res
                plot_rk4_simulation(res, district_key, 'Validation')

    if 'Test' in splits:
        df_dist_test = df_test[df_test['location_key'] == district_key]
        if not df_dist_test.empty:
            res = simulate_district_rk4(district_key, bundle, df_dist_test, df_dist_all, param_features, device)
            if res is not None:
                results['Test'] = res
                plot_rk4_simulation(res, district_key, 'Test')

    return results
