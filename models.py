# -*- coding: utf-8 -*-
"""
models.py — network architectures and SEIR ODE residual definitions.

StateNet: learns (S, E, I, R) as a function of (t, x, y). Untouched by
every modification in this pipeline's history except its unrelated role as
a boundary/data-fit target — see main.py's docstring, mod #15, which is
explicit that StateNet's input/architecture is never modified.

ParamNet: learns (beta, sigma, gamma) as a function of the 8-dim contextual
feature vector. Carries two modifications from the history in main.py:
  - mod #13: final activation Sigmoid -> Softplus (fixes saturation-induced
    collapse of the learned parameters to fixed corner values).
  - mod #15: raw input passed through a random Fourier feature encoding
    (FourierFeatures) before the first Linear layer, to counter spectral
    bias. StateNet is not given this encoding.
"""

import numpy as np
import torch
import torch.nn as nn

import config


class StateNet(nn.Module):
    """Learns (S, E, I, R) as a function of (t, x, y). Unmodified throughout
    the pipeline's history."""

    def __init__(self):
        super(StateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 4), nn.Sigmoid()
        )

    def forward(self, t, x, y):
        return self.net(torch.cat([t, x, y], dim=1))


class FourierFeatures(nn.Module):
    """mod #15: random Fourier feature encoding (Tancik et al., 2020),
    applied ONLY to ParamNet's raw contextual input — StateNet's (t, x, y)
    input is untouched. `B` is a fixed (non-trainable) random frequency
    matrix, sampled once per instance and stored as a buffer (so it moves
    with .to(device) but is never updated by the optimizer). Each
    per-domain ParamNet (mod #7) constructs its own FourierFeatures
    instance, so each domain gets its own independently-sampled B —
    consistent with domains already having fully independent parameters.

    forward() concatenates the raw input alongside sin/cos of the random
    projection, rather than replacing it, so the network keeps direct
    access to each feature's actual scale (e.g. pop_density's magnitude) in
    addition to the higher-frequency sin/cos basis that counters the
    spectral bias of a plain Tanh-MLP.
    """

    def __init__(self, in_dim, num_freqs=16, scale=4.0):
        super().__init__()
        B = torch.randn(in_dim, num_freqs) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class ParamNet(nn.Module):
    """Learns (beta, sigma, gamma) as a function of the 8-dim contextual
    feature vector. See module docstring for mods #13 and #15."""

    def __init__(self, n_param_inputs,
                 fourier_num_freqs=config.PARAM_NET_FOURIER_NUM_FREQS,
                 fourier_scale=config.PARAM_NET_FOURIER_SCALE):
        super(ParamNet, self).__init__()
        # mod #15: random Fourier feature encoding on ParamNet's raw
        # contextual input only. Raw input is concatenated alongside the
        # sin/cos encoding (not replaced), so ParamNet retains each
        # feature's actual scale as well as the higher-frequency basis.
        self.fourier = FourierFeatures(n_param_inputs, num_freqs=fourier_num_freqs, scale=fourier_scale)
        encoded_dim = n_param_inputs + 2 * fourier_num_freqs
        # mod #13: Sigmoid -> Softplus. Softplus has no flat, exactly-
        # zero-gradient saturation region the way Sigmoid does at both
        # tails, so a unit that gets pushed to a large pre-activation
        # still has a live (exponentially small, but nonzero) gradient
        # instead of getting stuck. The *2.0/*0.5/*0.5 scale multipliers
        # below are no longer a hard ceiling under Softplus (which is
        # unbounded above), just a scale factor to keep beta/sigma/gamma
        # at roughly their previous order of magnitude.
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Softplus()
        )

    def forward(self, p):
        p_enc = self.fourier(p)
        out = self.net(p_enc)
        return torch.cat([out[:, 0:1] * 2.0, out[:, 1:2] * 0.5, out[:, 2:3] * 0.5], dim=1)


def _net_device(module):
    return next(module.parameters()).device


def seir_residuals(S, E, I, R, dS, dE, dI, dR, beta, sigma, gamma):
    """SEIR ODE + algebraic-consistency residuals, used by
    training.compute_loss for loss_ode/loss_alg."""
    ds_eq = -beta * S * I
    de_eq = beta * S * I - sigma * E
    di_eq = sigma * E - gamma * I
    dr_eq = gamma * I
    return (dS - ds_eq, dE - de_eq, dI - di_eq, dR - dr_eq,
            (dS + dE + dI + dR), (dR - gamma * I), (dE - ds_eq - sigma * E))
