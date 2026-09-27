import os, glob, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
# ------- data ----
csv_path = next(iter(glob.glob("/kaggle/input/**/india_processed_final.csv", recursive=True)),
                 "india_processed_final.csv")
outdir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print("csv:", csv_path)

df = pd.read_csv(csv_path)
w = df.groupby("location_key")["pop_density"].first()
w = (w / w.sum()).rename("w")
df = df.merge(w, on="location_key")

agg_cols = ["S", "I", "R", "D", "stringency_index", "vaccination_rate",
            "full_vaccination_rate", "testing_rate", "tests_per_case",
            "mobility_score", "growth_rate", "I_lag_7", "I_lag_14"]
nat = (df.groupby("t")
         .apply(lambda g: pd.Series({c: np.average(g[c], weights=g["w"]) for c in agg_cols}))
         .reset_index().sort_values("t").reset_index(drop=True)
         .fillna(0.0))
nat.to_csv(f"{outdir}/national_series.csv", index=False)
t_np = nat["t"].values.astype(np.float32)
dt = float(t_np[1] - t_np[0])
T = len(t_np)
# ----- features -----
K = 8
fourier = np.concatenate([np.sin(2*np.pi*k*t_np)[:, None] for k in range(1, K+1)] +
                          [np.cos(2*np.pi*k*t_np)[:, None] for k in range(1, K+1)], axis=1)
cov_names = ["stringency_index", "vaccination_rate", "full_vaccination_rate",
             "testing_rate", "tests_per_case", "mobility_score",
             "growth_rate", "I_lag_7", "I_lag_14"]
cov = nat[cov_names].values.astype(np.float32)
cov[~np.isfinite(cov)] = 0.0     
cov = (cov - cov.mean(0)) / (cov.std(0) + 1e-8)
X = np.concatenate([t_np[:, None], fourier, cov], axis=1).astype(np.float32)
X = torch.tensor(X, device=device)
IN_DIM = X.shape[1]

S_t = torch.tensor(nat["S"].values, dtype=torch.float32, device=device)
I_t = torch.tensor(nat["I"].values, dtype=torch.float32, device=device)
R_t = torch.tensor(nat["R"].values, dtype=torch.float32, device=device)
D_t = torch.tensor(nat["D"].values, dtype=torch.float32, device=device)
wI, wR, wD = 1/I_t.var(), 1/R_t.var(), 1/D_t.var()

# ----- model -----
class ParamNet(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(),
                                  nn.Linear(h, h), nn.Tanh(),
                                  nn.Linear(h, 4))
    def forward(self, x):
        o = self.net(x)
        beta  = torch.nn.functional.softplus(o[:, 0]) * 1.2
        sigma = torch.nn.functional.softplus(o[:, 1]) * 0.5 + 0.05
        gamma = torch.nn.functional.softplus(o[:, 2]) * 0.3 + 0.02
        mu    = torch.nn.functional.softplus(o[:, 3]) * 0.02
        return beta, sigma, gamma, mu

net = ParamNet(IN_DIM).to(device)
E0_raw = nn.Parameter(torch.tensor(-6.0, device=device))
opt = torch.optim.Adam(list(net.parameters()) + [E0_raw], lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=800, gamma=0.5)

def deriv(state, beta, sigma, gamma, mu):
    S, E, I, R, D = state
    return torch.stack([-beta*S*I,
                         beta*S*I - sigma*E,
                         sigma*E - (gamma+mu)*I,
                         gamma*I,
                         mu*I])

def rk4(state, beta, sigma, gamma, mu, h):
    k1 = deriv(state, beta, sigma, gamma, mu)
    k2 = deriv(state + h/2*k1, beta, sigma, gamma, mu)
    k3 = deriv(state + h/2*k2, beta, sigma, gamma, mu)
    k4 = deriv(state + h*k3, beta, sigma, gamma, mu)
    return torch.clamp(state + h/6*(k1 + 2*k2 + 2*k3 + k4), 0.0, 1.2)

def rollout():
    beta, sigma, gamma, mu = net(X)                 # one batched forward for all T steps
    E0 = torch.nn.functional.softplus(E0_raw) * 0.01
    state = torch.stack([S_t[0]-E0, E0, I_t[0], R_t[0], D_t[0]])
    out = [state]
    for i in range(T - 1):                          # inherently sequential (RK4 recurrence)
        state = rk4(state, beta[i], sigma[i], gamma[i], mu[i], dt)
        out.append(state)
    return torch.stack(out), beta, sigma, gamma, mu
# ----- train ----
EPOCHS = 3000
t0 = time.time()
history = []
for ep in range(EPOCHS):
    states, beta, sigma, gamma, mu = rollout()
    I_sim, R_sim, D_sim = states[:, 2], states[:, 3], states[:, 4]
    data_loss = wI*torch.mean((I_sim-I_t)**2) + wR*torch.mean((R_sim-R_t)**2) + wD*torch.mean((D_sim-D_t)**2)
    smooth = torch.mean(torch.diff(beta)**2) + torch.mean(torch.diff(gamma)**2) + torch.mean(torch.diff(mu)**2)
    loss = data_loss + 5.0*smooth
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(net.parameters())+[E0_raw], 5.0)
    opt.step(); sched.step()
    history.append(loss.item())
    if ep % 200 == 0 or ep == EPOCHS-1:
        print(f"epoch {ep:5d}/{EPOCHS}  loss {loss.item():.4e}  data {data_loss.item():.4e}  "
              f"smooth {smooth.item():.4e}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s")
# ------ eval ----
with torch.no_grad():
    states, beta, sigma, gamma, mu = rollout()
S_sim, E_sim, I_sim, R_sim, D_sim = [x.cpu().numpy() for x in states.unbind(dim=1)]
beta, sigma, gamma, mu = beta.cpu().numpy(), sigma.cpu().numpy(), gamma.cpu().numpy(), mu.cpu().numpy()
S_n, I_n, R_n, D_n = S_t.cpu().numpy(), I_t.cpu().numpy(), R_t.cpu().numpy(), D_t.cpu().numpy()
rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
print("RMSE  S:", rmse(S_sim, S_n), " I:", rmse(I_sim, I_n),
      " R:", rmse(R_sim, R_n), " D:", rmse(D_sim, D_n))

fig, ax = plt.subplots(2, 3, figsize=(18, 9))
for a, name, sim, true in zip(ax.flat, "SIRD", [S_sim, I_sim, R_sim, D_sim], [S_n, I_n, R_n, D_n]):
    a.plot(t_np, true, "k", lw=2, label="actual")
    a.plot(t_np, sim, "r--", lw=1.5, label="PINN-RK4")
    a.set_title(name); a.legend()
ax.flat[4].plot(t_np, E_sim, color="orange"); ax.flat[4].set_title("E (latent)")
ax.flat[5].plot(t_np, beta, label="beta"); ax.flat[5].plot(t_np, sigma, label="sigma")
ax.flat[5].plot(t_np, gamma, label="gamma"); ax.flat[5].plot(t_np, mu, label="mu")
ax.flat[5].legend(); ax.flat[5].set_title("learned params")
plt.tight_layout(); plt.savefig(f"{outdir}/pinn_seird_results.png", dpi=150)
plt.figure(); plt.plot(history); plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("training loss")
plt.tight_layout(); plt.savefig(f"{outdir}/training_loss.png", dpi=150)
pd.DataFrame({"t": t_np, "S_true": S_n, "I_true": I_n, "R_true": R_n, "D_true": D_n,
              "S_sim": S_sim, "E_sim": E_sim, "I_sim": I_sim, "R_sim": R_sim, "D_sim": D_sim,
              "beta": beta, "sigma": sigma, "gamma": gamma, "mu": mu}
             ).to_csv(f"{outdir}/pinn_seird_output.csv", index=False)
print("saved outputs to", outdir)