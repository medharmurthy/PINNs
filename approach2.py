
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

CANDIDATES = glob.glob('/kaggle/input/**/india_pinn_comprehensive_data*.csv', recursive=True)
DATA_PATH = CANDIDATES[0] if CANDIDATES else 'india_pinn_comprehensive_data.csv'
print('Reading:', DATA_PATH)

OUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = {
    'IN_AP': 'Andhra Pradesh', 'IN_AR': 'Arunachal Pradesh', 'IN_AS': 'Assam',
    'IN_BR': 'Bihar', 'IN_CT': 'Chhattisgarh', 'IN_DL': 'Delhi',
    'IN_GJ': 'Gujarat', 'IN_HP': 'Himachal Pradesh', 'IN_HR': 'Haryana',
    'IN_JH': 'Jharkhand', 'IN_KA': 'Karnataka', 'IN_KL': 'Kerala',
    'IN_MH': 'Maharashtra', 'IN_ML': 'Meghalaya', 'IN_MN': 'Manipur',
    'IN_MP': 'Madhya Pradesh', 'IN_MZ': 'Mizoram', 'IN_NL': 'Nagaland',
    'IN_OR': 'Odisha', 'IN_PB': 'Punjab', 'IN_RJ': 'Rajasthan',
    'IN_TG': 'Telangana', 'IN_TN': 'Tamil Nadu', 'IN_UP': 'Uttar Pradesh',
    'IN_UT': 'Uttarakhand', 'IN_WB': 'West Bengal',
}


# %% [2] LOAD & CLEAN -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
print('Districts:', df.location_key.nunique()
resid = (df.S + df.I + df.R - 1.0).abs()
print('Max |S+I+R - 1| residual:', resid.max())

peak_I = df.groupby('location_key')['I'].max()
flat_locations = peak_I[peak_I < 1e-6].index.tolist()
print(f'Dropping {len(flat_locations)} flat-signal districts out of {df.location_key.nunique()}')
df = df[~df.location_key.isin(flat_locations)].reset_index(drop=True)

# Build readable labels once, used only for plot titles/legends below — the
# model itself still only ever sees x, y, t, S, I, R (see Section 6 model).
df['district_name'] = df['subregion2_name'].astype(str)
df['state_name'] = df['state_key'].map(STATE_NAMES).fillna(df['state_key'])
df['district_label'] = df['district_name'] + ', ' + df['state_name']

PEAK_WEIGHT_SCALE = 50.0
district_I_max = df.groupby('location_key')['I'].transform('max')
df['peak_weight'] = 1.0 + PEAK_WEIGHT_SCALE * (df['I'] / district_I_max.clip(lower=1e-9))

df = df[['location_key', 'district_name', 'state_name', 'district_label',
          'x', 'y', 't', 'S', 'I', 'R', 'peak_weight']].copy()

locations = df.location_key.unique()
rng = np.random.RandomState(SEED)
rng.shuffle(locations)

n_val = max(1, int(0.15 * len(locations)))
val_locations = set(locations[:n_val])
train_locations = set(locations[n_val:])

train_df = df[df.location_key.isin(train_locations)].reset_index(drop=True)
val_df   = df[df.location_key.isin(val_locations)].reset_index(drop=True)
print(f'Train districts: {len(train_locations)} ({len(train_df)} rows)')
print(f'Val districts:   {len(val_locations)} ({len(val_df)} rows)')


# %% [4] TENSORS ----------------------------------------------------------------------
def make_tensors(d):
    X = torch.tensor(d[['x', 'y', 't']].values, dtype=torch.float32)
    Y = torch.tensor(d[['S', 'I', 'R']].values, dtype=torch.float32)
    W = torch.tensor(d[['peak_weight']].values, dtype=torch.float32)
    return X.to(device), Y.to(device), W.to(device)

X_train, Y_train, W_train = make_tensors(train_df)
X_val,   Y_val,   W_val   = make_tensors(val_df)
N_COLLOC = 20000

x_min, x_max = df.x.min(), df.x.max()
y_min, y_max = df.y.min(), df.y.max()
t_min, t_max = df.t.min(), df.t.max()

Xc = torch.empty(N_COLLOC, 3, device=device)
Xc[:, 0] = torch.empty(N_COLLOC, device=device).uniform_(x_min, x_max)
Xc[:, 1] = torch.empty(N_COLLOC, device=device).uniform_(y_min, y_max)
Xc[:, 2] = torch.empty(N_COLLOC, device=device).uniform_(t_min, t_max)
Xc.requires_grad_(True)
BETA_INIT  = 0.4
GAMMA_INIT = 0.15
    def __init__(self, num_freqs=24, scale=6.0, seed=SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Bt = torch.randn(1, num_freqs, generator=g) * scale
        self.register_buffer('Bt', Bt)
        self.out_dim = 2 + num_freqs * 2   # raw x, y  +  sin/cos(t)

    def forward(self, xyt):
        xy = xyt[:, 0:2]
        t = xyt[:, 2:3]
        proj = 2 * np.pi * (t @ self.Bt)
        t_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return torch.cat([xy, t_feat], dim=-1)


class PlainSIRPinn(nn.Module):
    """
    state_net: (x, y, t) -> [raw x,y ; Fourier(t)] -> MLP -> raw 3-vector
               -> softmax -> (S, I, R), sums to 1.
    beta, gamma: two free scalar parameters, shared by every point in space
                 and time. This is the entire parametric content of classical
                 SIR -- nothing else is learned or conditioned on covariates.
                 The Fourier encoding only changes how state_net resolves
                 SHAPE in t; it has no effect on beta/gamma, which stay
                 exactly two plain scalars as before.
    Both live purely in the dataset's normalized-t coordinate.
    """
    def __init__(self, hidden=128, n_layers=7, num_freqs=24, fourier_scale=6.0):
        super().__init__()

        def mlp(in_dim, out_dim, hidden, n_layers):
            layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, out_dim)]
            return nn.Sequential(*layers)

        self.fourier = TimeFourierFeatures(num_freqs=num_freqs, scale=fourier_scale)
        self.state_net = mlp(self.fourier.out_dim, 3, hidden, n_layers)

        # Raw unconstrained scalars, passed through softplus at use-time so
        # beta, gamma > 0.
        self.beta_raw  = nn.Parameter(torch.tensor(float(BETA_INIT)))
        self.gamma_raw = nn.Parameter(torch.tensor(float(GAMMA_INIT)))

    def beta(self):
        return nn.functional.softplus(self.beta_raw) + 1e-4

    def gamma(self):
        return nn.functional.softplus(self.gamma_raw) + 1e-4

    def forward(self, xyt):
        feat = self.fourier(xyt)
        raw_sir = self.state_net(feat)
        sir = torch.softmax(raw_sir, dim=1)   # enforces S+I+R=1
        return sir


model = PlainSIRPinn(hidden=128, n_layers=7, num_freqs=24, fourier_scale=6.0).to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f'Total trainable parameters: {n_params:,} (includes the 2 scalar beta/gamma)')


# %% [7] LOSS FUNCTIONS ---------------------------------------------------------------
def physics_residual_loss(model, xyt_colloc):
    """
    Enforces the constant-coefficient SIR ODE system at collocation points:
        dS/dt + beta*S*I        = 0
        dI/dt - beta*S*I + gamma*I = 0
        dR/dt - gamma*I          = 0
    beta and gamma are the SAME two scalars at every collocation point —
    no per-point variation, no covariate modulation.
    """
    sir = model(xyt_colloc)
    S, I, R = sir[:, 0:1], sir[:, 1:2], sir[:, 2:3]
    beta, gamma = model.beta(), model.gamma()

    grad_outputs = torch.ones_like(S)
    dS = torch.autograd.grad(S, xyt_colloc, grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True)[0][:, 2:3]
    dI = torch.autograd.grad(I, xyt_colloc, grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True)[0][:, 2:3]
    dR = torch.autograd.grad(R, xyt_colloc, grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True)[0][:, 2:3]

    res_S = dS + beta * S * I
    res_I = dI - beta * S * I + gamma * I
    res_R = dR - gamma * I

    return (res_S.pow(2).mean() + 10*res_I.pow(2).mean() + res_R.pow(2).mean())


def data_loss(model, xyt, y_true, weight):
    """Weighted MSE -- weight upweights rows near each district's own
    infection peak (see PEAK_WEIGHT_SCALE note in Section 2), fixing the
    69%-of-rows-are-baseline imbalance found during debugging."""
    sir_pred = model(xyt)
    return (weight * (sir_pred - y_true).pow(2)).mean()


# %% [8] TRAINING LOOP ------------------------------------------------------------------
LAMBDA_PHYSICS = 1.0
LAMBDA_DATA    = 10.0
N_EPOCHS       = 17000
LR             = 1e-3
PRINT_EVERY    = 500

X_train_g = X_train.clone().requires_grad_(False)
Xc_g = Xc

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

history = {'epoch': [], 'data_loss': [], 'physics_loss': [], 'total_loss': [],
           'val_loss': [], 'beta': [], 'gamma': []}

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    l_data = data_loss(model, X_train_g, Y_train, W_train)
    l_phys = physics_residual_loss(model, Xc_g)
    loss = LAMBDA_DATA * l_data + LAMBDA_PHYSICS * l_phys

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            sir_val_pred = model(X_val)
            l_val = nn.functional.mse_loss(sir_val_pred, Y_val).item()
            cur_beta = model.beta().item()
            cur_gamma = model.gamma().item()

        history['epoch'].append(epoch)
        history['data_loss'].append(l_data.item())
        history['physics_loss'].append(l_phys.item())
        history['total_loss'].append(loss.item())
        history['val_loss'].append(l_val)
        history['beta'].append(cur_beta)
        history['gamma'].append(cur_gamma)

        print(f'Epoch {epoch:5d} | data {l_data.item():.6f} | '
              f'physics {l_phys.item():.6f} | total {loss.item():.6f} | '
              f'val {l_val:.6f} | beta {cur_beta:.4f} | gamma {cur_gamma:.4f} | '
              f'R0 {cur_beta/cur_gamma:.3f}')

print('Training complete.')
print(f'Final learned beta:  {model.beta().item():.4f}  (normalized-t units)')
print(f'Final learned gamma: {model.gamma().item():.4f}  (normalized-t units)')
print(f'Final implied R0:    {(model.beta()/model.gamma()).item():.4f}')


# %% [9] SAVE MODEL & TRAINING HISTORY --------------------------------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'beta': model.beta().item(),
    'gamma': model.gamma().item(),
}, os.path.join(OUT_DIR, 'plain_sir_pinn_model.pt'))

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(OUT_DIR, 'training_history.csv'), index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(hist_df.epoch, hist_df.data_loss, label='Data loss')
axes[0].plot(hist_df.epoch, hist_df.physics_loss, label='Physics residual loss')
axes[0].plot(hist_df.epoch, hist_df.val_loss, label='Validation data loss', linestyle='--')
axes[0].set_yscale('log')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (log scale)')
axes[0].set_title('Plain SIR-PINN Training History')
axes[0].legend()

axes[1].plot(hist_df.epoch, hist_df.beta, label='beta', color='#e67e22')
axes[1].plot(hist_df.epoch, hist_df.gamma, label='gamma', color='#7a7060')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Value (normalized-t units)')
axes[1].set_title('Learned scalar beta, gamma over training')
axes[1].legend()

axes[2].plot(hist_df.epoch, hist_df.beta / hist_df.gamma, color='#5a3a8a', lw=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('R0 = beta / gamma')
axes[2].set_title('Implied R0 over training')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_history.png'), dpi=150)
plt.show()


# %% [10] EVALUATION METRICS -----------------------------------------------------------
def evaluate(model, X, Y, label=''):
    model.eval()
    with torch.no_grad():
        sir_pred = model(X)
        mse = nn.functional.mse_loss(sir_pred, Y).item()
        mae = nn.functional.l1_loss(sir_pred, Y).item()

        names = ['S', 'I', 'R']
        per_compartment = {}
        for i, name in enumerate(names):
            comp_mse = nn.functional.mse_loss(sir_pred[:, i], Y[:, i]).item()
            comp_mae = nn.functional.l1_loss(sir_pred[:, i], Y[:, i]).item()
            per_compartment[name] = {'mse': comp_mse, 'mae': comp_mae}

    print(f'--- {label} ---')
    print(f'Overall MSE: {mse:.6f} | MAE: {mae:.6f}')
    for name, m in per_compartment.items():
        print(f'  {name}: MSE={m["mse"]:.6f}  MAE={m["mae"]:.6f}')
    return mse, mae, per_compartment

train_metrics = evaluate(model, X_train, Y_train, label='TRAIN')
val_metrics   = evaluate(model, X_val,   Y_val,   label='VALIDATION (held-out districts)')

print(f'\nGlobal fitted parameters (shared across ALL {len(train_locations)+len(val_locations)} districts):')
print(f'  beta  = {model.beta().item():.4f}   (normalized-t units)')
print(f'  gamma = {model.gamma().item():.4f}   (normalized-t units)')
print(f'  R0    = {(model.beta()/model.gamma()).item():.4f}')
print(f'  1/gamma (mean infectious period, in normalized-t units) = {(1/model.gamma()).item():.4f}')


# %% [11] WHOLE-INDIA AGGREGATE RESULTS (HEADLINE PLOT) ---------------------------------
# This is the main "all of India at once" result: every one of the 550
# districts (train + held-out) pooled together, plotted as one national
# average true trajectory vs. the model's prediction, using the SAME single
# global (beta, gamma) for every point — i.e. exactly what plain SIR produces
# when applied across the whole country at once rather than state-by-state.
all_df = pd.concat([train_df, val_df], ignore_index=True)

xyt_all = torch.tensor(all_df[['x', 'y', 't']].values, dtype=torch.float32, device=device)
model.eval()
with torch.no_grad():
    sir_pred_all = model(xyt_all).cpu().numpy()
all_df = all_df.copy()
all_df['S_pred'] = sir_pred_all[:, 0]
all_df['I_pred'] = sir_pred_all[:, 1]
all_df['R_pred'] = sir_pred_all[:, 2]

# Average across all districts at each time step -> one national curve
india_true = all_df.groupby('t')[['S', 'I', 'R']].mean().reset_index()
india_pred = all_df.groupby('t')[['S_pred', 'I_pred', 'R_pred']].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(india_true.t, india_true.S, '--', color='#1a4a7a', lw=1.5, alpha=0.7, label='Susceptible (actual, all-India average)')
ax.plot(india_true.t, india_true.I, '--', color='#c0392b', lw=1.5, alpha=0.7, label='Infectious (actual, all-India average)')
ax.plot(india_true.t, india_true.R, '--', color='#1a7a6e', lw=1.5, alpha=0.7, label='Recovered (actual, all-India average)')
ax.plot(india_pred.t, india_pred.S_pred, '-', color='#1a4a7a', lw=2.5, label='Susceptible (PINN-SIR fit)')
ax.plot(india_pred.t, india_pred.I_pred, '-', color='#c0392b', lw=2.5, label='Infectious (PINN-SIR fit)')
ax.plot(india_pred.t, india_pred.R_pred, '-', color='#1a7a6e', lw=2.5, label='Recovered (PINN-SIR fit)')

ax.set_xlabel('Time (normalized, full outbreak window)', fontsize=12)
ax.set_ylabel('Fraction of population', fontsize=12)
ax.set_title(
    f'Whole-India SIR-PINN Fit — All {all_df.location_key.nunique()} Districts Combined\n'
    f'R\u2080 = {(model.beta()/model.gamma()).item():.2f}   |   '
    f'\u03b2={model.beta().item():.4f}, \u03b3={model.gamma().item():.4f} (normalized-t units)',
    fontsize=12
)
ax.legend(fontsize=9, loc='center right')
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'whole_india_sir_fit.png'), dpi=150)
plt.show()
print('Saved: whole_india_sir_fit.png  (headline all-India result)')


# %% [11b] WHOLE-INDIA — SEPARATE S, I, R PLOTS (ONE COMPARTMENT PER PLOT) --------------
# Same style as the reference "Train - <district>: Infection trajectory" plot:
# one compartment per figure, true=solid line, predicted=dashed line.
def plot_compartment_whole_india(true_df, pred_df, compartment, pred_col, color, save=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(true_df.t, true_df[compartment], '-', color=color, lw=1.8, label=f'{compartment} true')
    ax.plot(pred_df.t, pred_df[pred_col], '--', color=color, lw=1.8, label=f'{compartment} pred')
    ax.set_title(f'All-India (550 districts, averaged) — {compartment} trajectory', fontsize=12)
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Fraction of population')
    ax.legend()
    plt.tight_layout()
    if save:
        fname = os.path.join(OUT_DIR, f'whole_india_{compartment}_trajectory.png')
        plt.savefig(fname, dpi=150)
        print('Saved:', fname)
    plt.show()
    plt.close(fig)

plot_compartment_whole_india(india_true, india_pred, 'S', 'S_pred', '#1a4a7a')
plot_compartment_whole_india(india_true, india_pred, 'I', 'I_pred', '#c0392b')
plot_compartment_whole_india(india_true, india_pred, 'R', 'R_pred', '#1a7a6e')


# %% [12] PER-DISTRICT CURVE RECONSTRUCTION & PLOTTING ----------------------------------
def plot_district(model, df_full, location_key, save=True):
    d = df_full[df_full.location_key == location_key].sort_values('t')
    if len(d) == 0:
        print(f'No data for {location_key}')
        return

    # Human-readable label, e.g. "Amroha, Uttar Pradesh" instead of IN_UP_AMN
    readable_name = d['district_label'].iloc[0]

    xyt = torch.tensor(d[['x', 'y', 't']].values, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        sir_pred = model(xyt)
    sir_pred = sir_pred.cpu().numpy()

    t = d.t.values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, d.S.values, 'o', ms=2, alpha=0.3, color='#1a4a7a', label='Susceptible (actual)')
    ax.plot(t, d.I.values, 'o', ms=2, alpha=0.3, color='#c0392b', label='Infectious (actual)')
    ax.plot(t, d.R.values, 'o', ms=2, alpha=0.3, color='#1a7a6e', label='Recovered (actual)')
    ax.plot(t, sir_pred[:, 0], '-', color='#1a4a7a', lw=2, label='Susceptible (PINN fit)')
    ax.plot(t, sir_pred[:, 1], '-', color='#c0392b', lw=2, label='Infectious (PINN fit)')
    ax.plot(t, sir_pred[:, 2], '-', color='#1a7a6e', lw=2, label='Recovered (PINN fit)')
    ax.set_xlabel('Time (normalized)', fontsize=11)
    ax.set_ylabel('Fraction of population', fontsize=11)
    ax.set_title(
        f'{readable_name} — Plain SIR Fit\n'
        f'(national \u03b2={model.beta().item():.3f}, \u03b3={model.gamma().item():.3f})',
        fontsize=12
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    if save:
        # Filenames stay code-based (filesystem-safe), only the in-plot text
        # is the readable name.
        fname = os.path.join(OUT_DIR, f'fit_{location_key}.png')
        plt.savefig(fname, dpi=150)
        print(f'Saved: {fname}   (title: {readable_name})')
    plt.show()
    plt.close(fig)


example_train_locs = list(train_locations)[:3]
example_val_locs   = list(val_locations)[:3]

for loc in example_train_locs:
    plot_district(model, df, loc)

for loc in example_val_locs:
    plot_district(model, df, loc)


# %% [12b] PER-DISTRICT — SEPARATE S, I, R PLOTS (MATCHES REFERENCE STYLE) ---------------
# One compartment per plot, true=solid line, predicted=dashed line, titled
# "<Set> - <District readable name>: <Compartment> trajectory" — same style
# as the "Train - IN_UP_HRP: Infection trajectory" reference plot.
def plot_district_compartments_separate(model, df_full, location_key, set_label='Train', save=True):
    d = df_full[df_full.location_key == location_key].sort_values('t')
    if len(d) == 0:
        print(f'No data for {location_key}')
        return

    readable_name = d['district_label'].iloc[0]
    xyt = torch.tensor(d[['x', 'y', 't']].values, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        sir_pred = model(xyt).cpu().numpy()

    t = d.t.values
    compartments = [
        ('S', d.S.values, sir_pred[:, 0], '#1a4a7a', 'Susceptible'),
        ('I', d.I.values, sir_pred[:, 1], '#c0392b', 'Infection'),
        ('R', d.R.values, sir_pred[:, 2], '#1a7a6e', 'Recovery'),
    ]

    for code, true_vals, pred_vals, color, traj_name in compartments:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, true_vals, '-', color=color, lw=1.5, label=f'{code} true')
        ax.plot(t, pred_vals, '--', color='#e67e22', lw=1.5, label=f'{code} pred')
        ax.set_title(f'{set_label} - {readable_name}: {traj_name} trajectory', fontsize=11)
        ax.legend()
        plt.tight_layout()
        if save:
            fname = os.path.join(
                OUT_DIR, f'{set_label.lower()}_{location_key}_{code}_trajectory.png'
            )
            plt.savefig(fname, dpi=150)
            print(f'Saved: {fname}   (title: {set_label} - {readable_name}: {traj_name} trajectory)')
        plt.show()
        plt.close(fig)


# A handful of train districts and a handful of held-out (validation) districts,
# matching the reference plot's "Train - <district>" / "Val - <district>" framing.
for loc in example_train_locs:
    plot_district_compartments_separate(model, df, loc, set_label='Train')

for loc in example_val_locs:
    plot_district_compartments_separate(model, df, loc, set_label='Val')


# %% [13] CLASSICAL SIR ODE SOLUTION USING THE LEARNED (beta, gamma) ---------------------
# Sanity check: solve the plain ODE system directly (RK4, no neural network)
# using the learned beta/gamma and compare to the average data trend. This
# confirms beta/gamma are behaving like genuine SIR parameters and not just
# absorbing fitting error.
def rk4_sir(beta, gamma, S0, I0, R0, steps=991):
    dt = 1.0 / steps
    S, I, R = S0, I0, R0
    out = [(0.0, S, I, R)]
    for i in range(steps):
        def f(s, i_, r):
            return (-beta*s*i_, beta*s*i_ - gamma*i_, gamma*i_)
        k1 = f(S, I, R)
        k2 = f(S+0.5*dt*k1[0], I+0.5*dt*k1[1], R+0.5*dt*k1[2])
        k3 = f(S+0.5*dt*k2[0], I+0.5*dt*k2[1], R+0.5*dt*k2[2])
        k4 = f(S+dt*k3[0], I+dt*k3[1], R+dt*k3[2])
        S += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        I += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        R += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        out.append(((i+1)*dt, S, I, R))
    return np.array(out)

avg_I0 = train_df[train_df.t == train_df.t.min()].I.mean()
avg_S0 = train_df[train_df.t == train_df.t.min()].S.mean()
avg_R0_init = train_df[train_df.t == train_df.t.min()].R.mean()

ode_curve = rk4_sir(model.beta().item(), model.gamma().item(),
                     avg_S0, max(avg_I0, 1e-6), avg_R0_init)

# Average true trajectory across all training districts at each t
avg_true = train_df.groupby('t')[['S', 'I', 'R']].mean().reset_index()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(avg_true.t, avg_true.S, '--', color='#1a4a7a', alpha=0.6, label='Susceptible (data average)')
ax.plot(avg_true.t, avg_true.I, '--', color='#c0392b', alpha=0.6, label='Infectious (data average)')
ax.plot(avg_true.t, avg_true.R, '--', color='#1a7a6e', alpha=0.6, label='Recovered (data average)')
ax.plot(ode_curve[:, 0], ode_curve[:, 1], '-', color='#1a4a7a', lw=2, label='Susceptible (RK4, fitted \u03b2/\u03b3)')
ax.plot(ode_curve[:, 0], ode_curve[:, 2], '-', color='#c0392b', lw=2, label='Infectious (RK4, fitted \u03b2/\u03b3)')
ax.plot(ode_curve[:, 0], ode_curve[:, 3], '-', color='#1a7a6e', lw=2, label='Recovered (RK4, fitted \u03b2/\u03b3)')
ax.set_xlabel('Time (normalized)', fontsize=11)
ax.set_ylabel('Fraction of population', fontsize=11)
ax.set_title(
    'Classical SIR ODE (solved directly, no neural network) using the\n'
    f'PINN-fitted national \u03b2/\u03b3 (R\u2080={(model.beta()/model.gamma()).item():.2f}), '
    'vs. all-India data average',
    fontsize=11.5
)
ax.legend(fontsize=8)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'classical_ode_vs_data_average.png'), dpi=150)
plt.show()

india_avg=avg_true
plt.figure(figsize=(8,5))

plt.plot(
    india_avg["t"],
    india_avg["S"],
    '--',
    linewidth=2,
    label='Actual India Average S'
)

plt.plot(
    ode_curve[:,0],
    ode_curve[:,1],
    linewidth=2,
    label='RK4 SIR Prediction'
)

plt.title("Whole India Average Susceptible")
plt.xlabel("Normalized Time")
plt.ylabel("Susceptible Fraction")
plt.grid(True)
plt.legend()


plt.show()
plt.figure(figsize=(8,5))

plt.plot(
    india_avg["t"],
    india_avg["I"],
    '--',
    linewidth=2,
    label='Actual India Average I'
)

plt.plot(
    ode_curve[:,0],
    ode_curve[:,2],
    linewidth=2,
    label='RK4 SIR Prediction'
)

plt.title("Whole India Average Infectious")
plt.xlabel("Normalized Time")
plt.ylabel("Infectious Fraction")
plt.grid(True)
plt.legend()

plt.show()
plt.figure(figsize=(8,5))

plt.plot(
    india_avg["t"],
    india_avg["R"],
    '--',
    linewidth=2,
    label='Actual India Average R'
)

plt.plot(
    ode_curve[:,0],
    ode_curve[:,3],
    linewidth=2,
    label='RK4 SIR Prediction'
)

plt.title("Whole India Average Recovered")
plt.xlabel("Normalized Time")
plt.ylabel("Recovered Fraction")
plt.grid(True)
plt.legend()

plt.show()
# %% [14] SUMMARY ----------------------------------------------------------------------
print('=' * 70)
print('SUMMARY — PLAIN SIR-PINN')
print('=' * 70)
print(f'Districts used: {len(train_locations) + len(val_locations)} '
      f'(train={len(train_locations)}, val={len(val_locations)})')
print(f'Train MSE: {train_metrics[0]:.6f} | Val MSE: {val_metrics[0]:.6f}')
print(f'Single global beta:  {model.beta().item():.4f}  (normalized-t units)')
print(f'Single global gamma: {model.gamma().item():.4f}  (normalized-t units)')
print(f'Single global R0:    {(model.beta()/model.gamma()).item():.4f}')
print(f'Outputs written to: {OUT_DIR}')
print(' - whole_india_sir_fit.png          (HEADLINE: all-India combined fit, readable labels)')
print(' - whole_india_<S|I|R>_trajectory.png (whole-India, ONE compartment per plot, true vs pred)')
print(' - <train|val>_<district>_<S|I|R>_trajectory.png (per-district, ONE compartment per plot)')
print(' - plain_sir_pinn_model.pt          (trained weights + final beta/gamma)')
print(' - training_history.csv/.png        (loss curves + beta/gamma convergence)')
print(' - fit_<district>.png               (per-district S/I/R reconstruction, readable titles)')
print(' - classical_ode_vs_data_average.png (RK4 sanity check against data average)')
print()
print('NOTE: this is plain SIR — ONE beta and ONE gamma for the entire dataset.')
print('No covariates were used. This will not capture per-district heterogeneity')
print('(lockdown timing, vaccination differences, etc.) by construction — that')
print('is the defining limitation of the classical model, not a bug.')