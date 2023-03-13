# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import math
import time
import itertools
import functools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
import seaborn as sns
# -

TENSORDTYPE = torch.float64
DEVICE = 'cpu'
LOGROOTDIR = './outputs'


# All problem-specific functions accpet tensor arguments.
#
# Take $f(t,x,a)$ as an example. We assume its inputs $t, x$ and $a$ are tensors with at least 1 dimension. All dimensions except the last are regarded as batch dimensions and are equivalent in function bodies. The outputs follow this rule too. This means even the time input $t$ and the value $f(t,x,a)$ are both scalar in their definition, we force them to be tensors in shape (1,) too.
#
# We also adopt the sequence-first convention, which is standard in seq2seq training. Most Tensors have shape (T, M, D), where
#
# - T : time axis
# - M : batch axis
# - D : original dim of this quantity

# # Helper Function

def re_cumsum(t, dim):
    r'''torch.cumsum in reverse direction'''
    return t + torch.sum(t, dim, keepdim=True) - torch.cumsum(t, dim)


def format_uncertainty(value, error, sig_fig=2):
    if error <= 0 or error >= value: return f"{value:.2G} ± {error:.2G}"
    digits = -math.floor(math.log10(error)) + sig_fig - 1
    if digits < sig_fig: return f"{value:.2G} ± {error:.2G}"
    return "{0:.{2}f}({1:.0f})".format(value, error*10**digits, digits)


def time_dir():
    return time.strftime("%y%m%d-%H%M", time.localtime())


# # Problem

# Solve the fully coupled FBSDE
#   \begin{equation*}
#   \left\{
#   \begin{aligned}
#     X_t &= x_0 + \int_0^tb(s,X_s,Y_s,Z_s)\,ds + \int_0^t\sigma(s,X_s,Y_s,Z_s)\,dW_s,\\
#     Y_t &= g(X_T) + \int_t^Tf(s,X_s,Y_s,Z_s)\,ds - \int_t^TZ_s\,dW_s.
#   \end{aligned}
#   \right.
#   \end{equation*}
#   Here, $X, Y, Z$ value in $\mathbb{R}^n, \mathbb{R}^m,\mathbb{R}^{m\times d}$.

# # FBSDE

class FBSDE_LongSin(object):
    
    def __init__(self, n=4):
        self.H = 50
        self.T = 1.0
        self.n = n
        self.m = 1
        self.d = self.n
        
        self.r = 0.
        self.sigma_0 = 0.4
        
        self.x0 = .5*np.pi*torch.ones(self.n).to(device=DEVICE, dtype=TENSORDTYPE)
        
    @property
    def dt(self):
        return self.T/self.H
        
    def b(self, t, x, y, z):
        return 0.*x
    
    def sigma(self, t, x, y, z):
        return self.sigma_0 * y.unsqueeze(-1) * torch.eye(self.d).to(device=DEVICE, dtype=TENSORDTYPE)
    
    def f(self, t, x, y, z):
        return -self.r*y + .5*torch.exp(-3*self.r*(self.dt*self.H-t))*self.sigma_0**2*(torch.sum(torch.sin(x), dim=-1, keepdim=True))**3
    
    def g(self, x):
        return torch.sum(torch.sin(x), dim=-1, keepdim=True)
    
    def get_Y(self, t, x):
        return torch.exp(-self.r*(self.dt*self.H-t))*torch.sum(torch.sin(x), dim=-1, keepdim=True)
    
    def get_Z(self, t, x):
        return self.sigma_0*(torch.exp(-2*self.r*(self.dt*self.H-t))*torch.sum(torch.sin(x), dim=-1, keepdim=True)*torch.cos(x)).unsqueeze(-2)


# # Network

# +
class YNet_FC3L(torch.nn.Module):
    
    def __init__(self, n, m, *, hidden_size):
        super().__init__()
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m, bias=True)
        )
        
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def forward(self, t, x):
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z)
    
    
class ZNet_FC3L(torch.nn.Module):
    
    def __init__(self, n, m, d, *, hidden_size):
        super().__init__()
        self.m = m
        self.d = d
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m*d, bias=True),
        )
        
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def forward(self, t, x):
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z).view(*x.shape[:-1], self.m, self.d)


# -

# # Solver

class FBSDE_BMLSolver(object):

    def __init__(self, fbsde):
        self.hidden_size = fbsde.n + 10
        self.batch_size = 512
        
        self.fbsde = fbsde
        self.ynet = YNet_FC3L(self.fbsde.n, self.fbsde.m, hidden_size=self.hidden_size)
        self.znet = ZNet_FC3L(self.fbsde.n, self.fbsde.m, self.fbsde.d, hidden_size=self.hidden_size)
        
        self.track_X_grad = False
        self.y_lr = 5e-3
        self.z_lr = 5e-3
        
    def set_parameter(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        elif hasattr(self.fbsde, name):
            setattr(self.fbsde, name, value)
        else:
            raise ValueError(f"{name} is not a proper parameter")
        
    def get_optimizer(self):
        return torch.optim.Adam([
            {'params': self.ynet.parameters(), 'lr': self.y_lr,},
            {'params': self.znet.parameters(), 'lr': self.z_lr,}
        ])
        
    def obtain_XYZ(self, t=None, dW=None):
        if t is None:
            t = torch.tensor([self.fbsde.dt*i for i in range(1+self.fbsde.H)], dtype=TENSORDTYPE, device=DEVICE).reshape(-1,1,1).expand(-1, self.batch_size, 1)
        
        if dW is None:
            dW = torch.normal(0., np.sqrt(self.fbsde.dt), size=(self.fbsde.H, self.batch_size, self.fbsde.d), dtype=TENSORDTYPE, device=DEVICE)
        
        X = torch.empty(1+self.fbsde.H, dW.shape[1], self.fbsde.n, dtype=TENSORDTYPE, device=DEVICE)
        Y = torch.empty(1+self.fbsde.H, dW.shape[1], self.fbsde.m, dtype=TENSORDTYPE, device=DEVICE)
        Z = torch.empty(1+self.fbsde.H, dW.shape[1], self.fbsde.m, self.fbsde.d, dtype=TENSORDTYPE, device=DEVICE)
        
        X[0] = self.fbsde.x0
        Y[0] = self.ynet(t[0], X[0])
        Z[0] = self.znet(t[0], X[0])
        for i in range(self.fbsde.H):
            if self.track_X_grad is True:
                X[i+1] = X[i] + self.fbsde.dt * self.fbsde.b(t[i], X[i], Y[i], Z[i]) + (self.fbsde.sigma(t[i], X[i], Y[i], Z[i]) @ dW[i].unsqueeze(-1)).squeeze(-1)
            else:
                with torch.no_grad():
                    X[i+1] = X[i] + self.fbsde.dt * self.fbsde.b(t[i], X[i], Y[i], Z[i]) + (self.fbsde.sigma(t[i], X[i], Y[i], Z[i]) @ dW[i].unsqueeze(-1)).squeeze(-1)
            Y[i+1] = self.ynet(t[i+1], X[i+1])
            Z[i+1] = self.znet(t[i+1], X[i+1])
        
        return t, X, Y, Z, dW
    
    def calc_loss(self, dirac=True, *, txyzw=None):
        if txyzw is None:
            t, X, Y, Z, dW = self.obtain_XYZ()
        else:
            t, X, Y, Z, dW = txyzw
            
        if dirac is True:
            error = Y[0] - (self.fbsde.g(X[-1]) + self.fbsde.dt * torch.sum(self.fbsde.f(t[:-1], X[:-1], Y[:-1], Z[:-1]), dim=0) - torch.sum(Z[:-1] @ dW.unsqueeze(-1), dim=0).squeeze(-1))
            return torch.sum(error*error/dW.shape[1])
        else:
            error = Y[:-1] - (self.fbsde.g(X[-1:]) + self.fbsde.dt * re_cumsum(self.fbsde.f(t[:-1], X[:-1], Y[:-1], Z[:-1]), dim=0) - re_cumsum(Z[:-1] @ dW.unsqueeze(-1), dim=0).squeeze(-1))
            if dirac is False:
                return (error*error).mean()
            elif isinstance(dirac, float):
               # weight = torch.exp(-dirac*t[:-1]/self.fbsde.dt)*(1-np.exp(-dirac))/(1-np.exp(-dirac*t.shape[0]))
                return torch.sum(error/ dW.shape[1] * error * self._get_weight(dirac, t.shape[0]-1).view(*([-1] + [1]*(len(t.shape)-1))))
            else:
                raise ValueError(f"Unknown dirac value={dirac}")
         
    @functools.lru_cache(maxsize=None)
    def _get_weight(self, gamma, N):
        return (1-np.exp(-gamma))/(1-np.exp(-gamma*N))*torch.exp(-gamma*torch.arange(N)).to(dtype=TENSORDTYPE, device=DEVICE)


# # Benchmark of Func calc_loss

# +
test_solver = FBSDE_BMLSolver(FBSDE_LongSin(n=4))
with torch.no_grad():
    t, X, Y, Z, dW = test_solver.obtain_XYZ()

terminal_error = test_solver.fbsde.g(X[-1]).squeeze() - X[-1].sin().sum(dim=-1)
running_error = test_solver.fbsde.f(t[:-1], X[:-1], Y[:-1], Z[:-1]) - (-test_solver.fbsde.r*Y[:-1]+.5*np.exp(-3*test_solver.fbsde.r*(test_solver.fbsde.dt*test_solver.fbsde.H-t[:-1]))*test_solver.fbsde.sigma_0**2*(X[:-1].sin().sum(dim=-1, keepdim=True))**3)
martingale_error = (Z[:-1] @ dW.unsqueeze(-1)).squeeze() - torch.sum(Z[:-1].squeeze(-2)*dW,dim=-1)

assert terminal_error.abs().max() < 1e-15
assert running_error.abs().max() < 1e-15
assert martingale_error.abs().max() < 1e-15
# -

# # Loss of True Solutions

# +
optimal_solver = FBSDE_BMLSolver(FBSDE_LongSin(n=4))
optimal_solver.ynet = optimal_solver.fbsde.get_Y
optimal_solver.znet = optimal_solver.fbsde.get_Z

optimal_solver.set_parameter('sigma_0', 0.3)
optimal_solver.set_parameter('r', .1)
optimal_solver.set_parameter('H', 200)

dirac_loss, lambd_loss, gamma_loss = [], [], []
for _ in range(10):
    t, X, Y, Z, dW = optimal_solver.obtain_XYZ()
    dirac_loss.append(optimal_solver.calc_loss(dirac=True, txyzw=(t, X, Y, Z, dW)).item())
    lambd_loss.append(optimal_solver.calc_loss(dirac=False, txyzw=(t, X, Y, Z, dW)).item())
    gamma_loss.append(optimal_solver.calc_loss(dirac=0.05, txyzw=(t, X, Y, Z, dW)).item())

print("δ-BML: ", format_uncertainty(np.mean(dirac_loss), np.std(dirac_loss)))
print("μ-BML: ", format_uncertainty(np.mean(lambd_loss), np.std(lambd_loss)))
print("γ-BML: ", format_uncertainty(np.mean(gamma_loss), np.std(gamma_loss)))


# -

# # Train

# ## Search Hyperparameters

def solve_LongSin(n, *, dirac, repeat=10, **solver_kws):
    para_logs, loss_logs = [], [[] for _ in range(repeat)]
    for epi in range(repeat):
        _solver = FBSDE_BMLSolver(FBSDE_LongSin(n=n))
        
        for k in solver_kws:
            _solver.set_parameter(k, solver_kws[k])
        
        optimizer = _solver.get_optimizer()
        
        _solver.ynet.train()
        _solver.znet.train()
        optimizer.zero_grad()
        for step in tqdm.trange(2000):
            loss = _solver.calc_loss(dirac=dirac)
            
            loss_logs[epi].append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        _solver.ynet.eval()
        _solver.znet.eval()
        _solver.batch_size = 512
        with torch.no_grad():
            t, X, Y, Z, dW = _solver.obtain_XYZ()
        error = (Y - _solver.fbsde.get_Y(t, X)).abs().squeeze(-1).detach().cpu()
        
        para_logs.append({
            'error_Y0': error.mean(dim=-1)[0].item(),
            'error_mean': error.mean().item(),
            'error_std': error.std().item(),
        })
        
    para_logs_agg = {}
    for k in para_logs[0].keys():
        arr = [p[k] for p in para_logs]
        para_logs_agg[k] = format_uncertainty(np.mean(arr), np.std(arr))
        
    return para_logs_agg, loss_logs


# +
search_mesh = {
    'dirac': [False], #, True],
    'y_lr': [5e-3], #, 5e-4, 5e-5],
    'z_lr': [5e-3],
    'batch_size': [256], #, 1024],
    
    'r': [0., 1.],
    'sigma_0': [0.2, 0.4],
}

res = []
for args in itertools.product(*search_mesh.values()):
    args = dict(zip(search_mesh.keys(), args))
    if abs(np.log10(args['y_lr']/args['z_lr'])) > 1.9:
        continue

    para_logs, loss_logs = solve_LongSin(n=4, repeat=10, **args)
    
    res.append({
        'args': args,
        'para_logs': para_logs,
        'loss_logs': loss_logs,
    })
# -

args_df = pd.DataFrame([{**r['args'], **r['para_logs']} for r in res])
args_df

log_dir = os.path.join(LOGROOTDIR, time_dir())
os.makedirs(log_dir, exist_ok=True)
args_df.to_csv(os.path.join(log_dir, "args.csv"), index=False)
print(f"args.csv saved to {log_dir}")

loss_logs_m = np.asarray([r['loss_logs'] for r in res])
np.save(os.path.join(log_dir, "loss_logs_m.npy"), loss_logs)
print(f"loss_logs_m.npy saved to {log_dir}")

# +
r_figs = 3
c_figs = math.ceil(len(res)/r_figs)
fig, axes = plt.subplots(c_figs, r_figs, figsize=(4.2*r_figs, 3.6*c_figs))
if c_figs == 1:
    axes = axes.reshape(1, -1)

for i, j in itertools.product(range(c_figs), range(r_figs)):
    if  i*r_figs + j >= len(res):
        break
    loss_arr = res[i*r_figs + j]['loss_logs']
    loss_data = pd.DataFrame([
        {'grad step': step, 'loss': loss_arr[exp_i][step], 'Exp No.': exp_i,} for exp_i, step in itertools.product(range(len(loss_arr)), range(len(loss_arr[0])))
    ])
    sns.lineplot(data=loss_data, x='grad step', y='loss', ax=axes[i,j], errorbar=('ci', 68))
    axes[i,j].set_yscale('log')
