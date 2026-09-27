# -*- coding: utf-8 -*-
"""
training.py — loss computation, checkpointing, post-training diagnostics,
and the domain-decomposed training loop (train_district / train_single_district).

Trains fully in fp32 (mod #5 — no torch.cuda.amp), with two Adam parameter
groups (mod #6, superseded structurally by #7's per-domain ParamNets) and
two independent clip_grad_norm_ calls. See main.py's module docstring for
the complete mod history (#1-#16); the mods most relevant to this file are
#5, #6, #7, #8/#9 (rk4 loss added then fully removed — total loss here is
just data + ode + alg + boundary), #11 (ode/alg weight ramp), and #12
(post-training diagnostics).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config
from data_preprocessing import build_colloc_curvature_weights, sample_colloc_t
from models import ParamNet, StateNet, _net_device, seir_residuals


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def compute_loss(state_net, param_net, t_d, x_d, y_d, p_d, tgt_d,
                  t_c, x_c, y_c, p_c, prev_state_net, t_bound_curr, t_bound_prev,
                  x_bound_curr, y_bound_curr, x_bound_prev, y_bound_prev,
                  weights, scales, dt_scale):

    domain_device = t_d.device
    param_device = _net_device(param_net)

    pred_states = state_net(t_d, x_d, y_d)
    eps = 1e-6
    rel_data_err = (pred_states[:, [0, 2, 3]] - tgt_d) / (tgt_d + eps)
    loss_data = torch.mean(rel_data_err ** 2)

    loss_boundary = torch.tensor(0.0, device=domain_device)
    if prev_state_net is not None and t_bound_curr is not None:
        prev_pred = prev_state_net(t_bound_prev, x_bound_prev, y_bound_prev).to(domain_device)
        curr_pred = state_net(t_bound_curr, x_bound_curr, y_bound_curr)
        rel_bound_err = (curr_pred - prev_pred) / (prev_pred + eps)
        loss_boundary = torch.mean(rel_bound_err ** 2)

    pred_c = state_net(t_c, x_c, y_c)
    S, E, I, R = pred_c[:, 0:1], pred_c[:, 1:2], pred_c[:, 2:3], pred_c[:, 3:4]

    grads = [
        torch.autograd.grad(var, t_c, torch.ones_like(var),
                             create_graph=True, retain_graph=True)[0] * dt_scale
        for var in (S, E, I, R)
    ]

    beta, sigma, gamma = param_net(p_c.to(param_device)).split(1, dim=1)
    beta, sigma, gamma = beta.to(domain_device), sigma.to(domain_device), gamma.to(domain_device)

    res = seir_residuals(S, E, I, R, *grads, beta, sigma, gamma)

    loss_ode = torch.mean(sum((r / scales[k]) ** 2 for r, k in zip(res[:4], ['S', 'E', 'I', 'R'])))
    loss_alg = torch.mean(sum(g ** 2 for g in res[4:]))

    loss_total = (weights['data'] * loss_data +
                  weights['ode'] * loss_ode +
                  weights['alg'] * loss_alg +
                  weights['boundary'] * loss_boundary)

    return loss_total, loss_data, loss_ode, loss_alg, loss_boundary


# ─────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────
def save_checkpoint(state_net, param_net, optimizer, epoch, district_id, domain_id, filepath=None,
                     overwrite=True):
    if filepath is None:
        filepath = os.path.join(config.OUTPUT_BASE, "checkpoints", district_id, f"domain_{domain_id}")
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)

    filename = f"{filepath}/model_latest.pt" if overwrite else f"{filepath}/model_epoch{epoch}.pt"
    checkpoint = {
        'epoch': epoch,
        'state_net_state_dict': state_net.state_dict(),
        'param_net_state_dict': param_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


# ─────────────────────────────────────────────────────────────
# POST-TRAINING DIAGNOSTICS (mod #12)
# ─────────────────────────────────────────────────────────────
def report_learned_params(param_nets, domain_data, district_key='unknown'):
    """Prints per-domain mean±std of the learned beta/sigma/gamma, evaluated
    over that domain's own training rows (d['p'])."""
    print(f"\n{'='*78}\nLearned ODE parameters (beta, sigma, gamma) per domain — {district_key}\n{'='*78}")
    header = (f"{'Domain':>6} | {'N rows':>7} | {'beta (mean ± std)':>20} | "
              f"{'sigma (mean ± std)':>20} | {'gamma (mean ± std)':>20}")
    print(header)
    print('-' * len(header))
    for k, (pn, d) in enumerate(zip(param_nets, domain_data)):
        with torch.no_grad():
            out = pn(d['p'].to(_net_device(pn)))
        beta = out[:, 0].cpu().numpy()
        sigma = out[:, 1].cpu().numpy()
        gamma = out[:, 2].cpu().numpy()
        print(f"{k:>6} | {len(d['p']):>7} | "
              f"{beta.mean():>7.4f} ± {beta.std():<8.4f} | "
              f"{sigma.mean():>7.4f} ± {sigma.std():<8.4f} | "
              f"{gamma.mean():>7.4f} ± {gamma.std():<8.4f}")
    print('=' * 78)


def report_state_net_fit(state_nets, domain_data, district_key='unknown'):
    """Prints per-domain and overall MSE/MAE of state_net(t, x, y) against
    the true (S, I, R) training targets — a direct data-fit check with no
    RK4 integration involved."""
    print(f"\n{'='*90}\nStateNet direct fit to training data (no RK4 rollout) — {district_key}\n{'='*90}")
    header = (f"{'Domain':>6} | {'N rows':>7} | {'MSE(S)':>10} | {'MSE(I)':>10} | {'MSE(R)':>10} | "
              f"{'MAE(S)':>10} | {'MAE(I)':>10} | {'MAE(R)':>10}")
    print(header)
    print('-' * len(header))

    all_pred, all_tgt = [], []
    for k, (sn, d) in enumerate(zip(state_nets, domain_data)):
        with torch.no_grad():
            pred = sn(d['t_norm'], d['x'], d['y'])
        pred_sir = pred[:, [0, 2, 3]].cpu().numpy()
        tgt = d['tgt'].cpu().numpy()
        mse = ((pred_sir - tgt) ** 2).mean(axis=0)
        mae = np.abs(pred_sir - tgt).mean(axis=0)
        print(f"{k:>6} | {len(d['p']):>7} | "
              f"{mse[0]:>10.2e} | {mse[1]:>10.2e} | {mse[2]:>10.2e} | "
              f"{mae[0]:>10.2e} | {mae[1]:>10.2e} | {mae[2]:>10.2e}")
        all_pred.append(pred_sir)
        all_tgt.append(tgt)

    all_pred = np.concatenate(all_pred, axis=0)
    all_tgt = np.concatenate(all_tgt, axis=0)
    overall_mse = ((all_pred - all_tgt) ** 2).mean(axis=0)
    overall_mae = np.abs(all_pred - all_tgt).mean(axis=0)
    print('-' * len(header))
    print(f"{'ALL':>6} | {len(all_tgt):>7} | "
          f"{overall_mse[0]:>10.2e} | {overall_mse[1]:>10.2e} | {overall_mse[2]:>10.2e} | "
          f"{overall_mae[0]:>10.2e} | {overall_mae[1]:>10.2e} | {overall_mae[2]:>10.2e}")
    print('=' * 90)


def plot_state_net_fit(state_nets, domain_data, district_key='unknown'):
    """Plots state_net's raw prediction vs. true S/I/R across the full
    training set, stitched across domains, ordered by real time (t_raw)."""
    labels = ['S', 'I', 'R']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    all_t = []
    all_pred = {0: [], 1: [], 2: []}
    all_tgt = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for sn, d in zip(state_nets, domain_data):
            pred = sn(d['t_norm'], d['x'], d['y'])
            pred_sir = pred[:, [0, 2, 3]].cpu().numpy()
            tgt = d['tgt'].cpu().numpy()
            t_cpu = d['t_raw'].cpu().numpy().squeeze()

            all_t.extend(np.atleast_1d(t_cpu))
            for c in range(3):
                all_pred[c].extend(pred_sir[:, c])
                all_tgt[c].extend(tgt[:, c])

    if not all_t:
        plt.close(fig)
        return

    order = np.argsort(all_t)
    all_t = np.array(all_t)[order]

    for c, label in enumerate(labels):
        axes[c].plot(all_t, np.array(all_tgt[c])[order], 'o', ms=3, label='true')
        axes[c].plot(all_t, np.array(all_pred[c])[order], '.', ms=3, label='state_net pred')
        axes[c].set_title(f"{district_key} — {label} (StateNet direct fit)")
        axes[c].set_xlabel("t")
        axes[c].legend()

    fig.suptitle("StateNet Direct Fit to Training Data (no RK4 rollout)", fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# TRAINING ROUTER
# ─────────────────────────────────────────────────────────────
def train_district(df_dist, n_param_inputs, district_key='unknown'):
    """Trains one district's per-domain StateNets and ParamNets jointly.

    Reads config.N_DOMAINS, so config.N_DOMAINS (and config.T_DAYS) must be
    set — e.g. by main.py, right after data_preprocessing.load_and_preprocess_data()
    returns T_DAYS — before this function is called.
    """
    t_col, x_col, y_col = config.DATA_COLUMNS['time'], config.DATA_COLUMNS['x'], config.DATA_COLUMNS['y']
    tgt_cols = config.DATA_COLUMNS['targets']
    p_cols = config.DATA_COLUMNS['param_features']
    n_domains = config.N_DOMAINS
    devices = config.DEVICES

    t_all = torch.tensor(df_dist[t_col].values, dtype=torch.float32).view(-1, 1)
    x_all = torch.tensor(df_dist[x_col].values, dtype=torch.float32).view(-1, 1)
    y_all = torch.tensor(df_dist[y_col].values, dtype=torch.float32).view(-1, 1)
    target_all = torch.tensor(df_dist[tgt_cols].values, dtype=torch.float32)
    p_all = torch.tensor(df_dist[p_cols].values, dtype=torch.float32)

    domain_edges = np.linspace(float(t_all.min()), float(t_all.max()), n_domains + 1)

    domain_devices = [devices[k % len(devices)] for k in range(n_domains)]
    state_nets = [StateNet().to(domain_devices[k]) for k in range(n_domains)]
    param_nets = [ParamNet(n_param_inputs).to(domain_devices[k]) for k in range(n_domains)]

    param_net_params = []
    for pn in param_nets:
        param_net_params += list(pn.parameters())
    state_net_params = []
    for sn in state_nets:
        state_net_params += list(sn.parameters())

    optimizer = torch.optim.Adam([
        {'params': param_net_params, 'lr': config.PARAM_NET_LR},
        {'params': state_net_params, 'lr': config.LR},
    ])
    domain_data = []

    for k in range(n_domains):
        dom_device = domain_devices[k]
        t_start, t_end = domain_edges[k], domain_edges[k + 1]
        overlap_width = config.OVERLAP_FRAC * (t_end - t_start)
        t_start_overlap = max(float(t_all.min()), t_start - overlap_width) if k > 0 else t_start
        dt_real = t_end - t_start_overlap + 1e-8
        dt_scale = 2.0 / dt_real

        mask_d = (t_all.squeeze() >= t_start) & (t_all.squeeze() <= t_end)
        t_raw_k = t_all[mask_d].to(dom_device)
        t_d_k_norm = (2.0 * (t_raw_k - t_start_overlap) / dt_real - 1.0).detach()
        x_d_k = x_all[mask_d].detach().to(dom_device)
        y_d_k = y_all[mask_d].detach().to(dom_device)
        tgt_d_k = target_all[mask_d].detach().to(dom_device)
        p_d_k = p_all[mask_d].detach().to(dom_device)

        t_bound_curr, t_bound_prev = None, None
        x_bound_curr, y_bound_curr, x_bound_prev, y_bound_prev = None, None, None, None
        if k > 0:
            prev_device = domain_devices[k - 1]
            mask_b = (t_all.squeeze() >= t_start_overlap) & (t_all.squeeze() <= t_start)
            t_b_raw = t_all[mask_b]

            t_bound_curr = (2.0 * (t_b_raw.to(dom_device) - t_start_overlap) / dt_real - 1.0).detach()

            prev_t_start, prev_t_end = domain_edges[k - 1], domain_edges[k]
            prev_overlap_width = config.OVERLAP_FRAC * (prev_t_end - prev_t_start)
            prev_t_start_overlap = max(float(t_all.min()), prev_t_start - prev_overlap_width) if (k - 1) > 0 else prev_t_start
            prev_dt_real = prev_t_end - prev_t_start_overlap + 1e-8
            t_bound_prev = (2.0 * (t_b_raw.to(prev_device) - prev_t_start_overlap) / prev_dt_real - 1.0).detach()

            x_bound_curr = x_all[mask_b].detach().to(dom_device)
            y_bound_curr = y_all[mask_b].detach().to(dom_device)
            x_bound_prev = x_all[mask_b].detach().to(prev_device)
            y_bound_prev = y_all[mask_b].detach().to(prev_device)

        colloc_weights, _ = build_colloc_curvature_weights(t_d_k_norm, tgt_d_k)

        domain_data.append({
            'device': dom_device,
            't_raw': t_raw_k,
            't_norm': t_d_k_norm, 'x': x_d_k, 'y': y_d_k,
            'tgt': tgt_d_k, 'p': p_d_k, 'dt_scale': dt_scale,
            't_bound_curr': t_bound_curr, 't_bound_prev': t_bound_prev,
            'x_bound_curr': x_bound_curr, 'y_bound_curr': y_bound_curr,
            'x_bound_prev': x_bound_prev, 'y_bound_prev': y_bound_prev,
            'colloc_weights': colloc_weights,
        })

    weights = dict(config.INIT_WEIGHTS)

    pbar = tqdm(range(config.EPOCHS_PER_DOMAIN), desc=f"Training {district_key} ({len(devices)} GPU(s))")

    for epoch in pbar:
        if epoch < config.WARMUP_EPOCHS:
            weights['ode'] = 0.0
            weights['alg'] = 0.0
            for pn in param_nets:
                pn.requires_grad_(False)
        elif epoch == config.WARMUP_EPOCHS:
            weights['ode'] = 1.0
            weights['alg'] = 1.0
            for pn in param_nets:
                pn.requires_grad_(True)

        # mod #11: linear ramp for ode/alg weights, starting the epoch
        # warmup ends. See main.py's module docstring for the full
        # diagnosis of why a flat weight=1.0 left ParamNet's gradient
        # perpetually under clip_grad_norm_'s threshold.
        if epoch >= config.WARMUP_EPOCHS:
            ramp_progress = min(1.0, (epoch - config.WARMUP_EPOCHS) / config.ODE_ALG_RAMP_EPOCHS)
            weights['ode'] = 1.0 + ramp_progress * (config.ODE_ALG_TARGET_WEIGHT - 1.0)
            weights['alg'] = 1.0 + ramp_progress * (config.ODE_ALG_TARGET_WEIGHT - 1.0)

        optimizer.zero_grad(set_to_none=True)

        is_last_epoch = (epoch == config.EPOCHS_PER_DOMAIN - 1)
        do_log = (epoch % config.CHECKPOINT_INTERVAL == 0) or is_last_epoch

        if do_log:
            epoch_loss_value = 0.0
            epoch_l_data = 0.0
            epoch_l_ode = 0.0
            epoch_l_alg = 0.0
            epoch_l_bound = 0.0

        for k in range(n_domains):
            d = domain_data[k]
            dom_device = d['device']
            N_colloc = min(config.N_COLLOC_MAX, len(d['p']) * 5)
            if N_colloc == 0:
                continue

            t_c = sample_colloc_t(d['t_norm'], d['colloc_weights'], N_colloc, dom_device)
            x_c = torch.empty(N_colloc, 1, device=dom_device).uniform_(-1.0, 1.0).requires_grad_(True)
            y_c = torch.empty(N_colloc, 1, device=dom_device).uniform_(-1.0, 1.0).requires_grad_(True)
            idx = np.random.choice(len(d['p']), N_colloc, replace=True)
            p_c = d['p'][idx].detach().clone()

            prev_sn = state_nets[k - 1] if k > 0 else None

            loss_total, l_data, l_ode, l_alg, l_bound = compute_loss(
                state_nets[k], param_nets[k],
                d['t_norm'], d['x'], d['y'], d['p'], d['tgt'],
                t_c, x_c, y_c, p_c,
                prev_sn, d['t_bound_curr'], d['t_bound_prev'],
                d['x_bound_curr'], d['y_bound_curr'], d['x_bound_prev'], d['y_bound_prev'],
                weights, config.SCALES, d['dt_scale'],
            )

            loss_total.backward()

            if do_log:
                epoch_loss_value += loss_total.item()
                epoch_l_data += l_data.item()
                epoch_l_ode += l_ode.item()
                epoch_l_alg += l_alg.item()
                epoch_l_bound += l_bound.item()
                save_checkpoint(state_nets[k], param_nets[k], optimizer, epoch, district_key, k)

        torch.nn.utils.clip_grad_norm_(param_net_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(state_net_params, max_norm=1.0)
        optimizer.step()

        if do_log:
            pbar.set_postfix(
                Tot=f"{epoch_loss_value:.2e}",
                Dat=f"{weights['data']:.0e}*{epoch_l_data:.2e}",
                Bnd=f"{weights['boundary']:.0e}*{epoch_l_bound:.2e}",
                ODE=f"{weights['ode']:.0e}*{epoch_l_ode:.2e}",
                Alg=f"{weights['alg']:.0e}*{epoch_l_alg:.2e}",
            )

    report_learned_params(param_nets, domain_data, district_key)
    report_state_net_fit(state_nets, domain_data, district_key)
    plot_state_net_fit(state_nets, domain_data, district_key)

    return state_nets, param_nets, domain_edges


def train_single_district(df_dataset, district_key, df_val=None):
    n_param_inputs = len(config.DATA_COLUMNS['param_features'])

    df_dist = df_dataset[df_dataset['location_key'] == district_key].sort_values(config.DATA_COLUMNS['time'])

    print(f"--> Training single district: '{district_key}' ({len(df_dist)} rows)")

    state_nets, param_nets, domain_edges = train_district(
        df_dist, n_param_inputs, district_key=district_key
    )

    return {
        district_key: {
            'state_nets': state_nets,
            'param_nets': param_nets,
            'domain_edges': domain_edges,
        }
    }
