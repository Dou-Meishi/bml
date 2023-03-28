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

# +
# local files
import bml.config

# change config before loading other modules
TENSORDTYPE = bml.config.TENSORDTYPE = torch.float32
DEVICE = bml.config.DEVICE = "cpu"

from bml.utils import *
from bml.fbsde_rescalc import FBSDE_LongSin_ResCalc
from bml.calc import ResSolver
from bml.models import YZNet_FC3L
# -

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

# # Problem

# Solve the fully coupled FBSDE
# \begin{equation*}
# \left\{
# \begin{aligned}
#   X_t &= x_0 + \int_0^tb(s,X_s,Y_s,Z_s)\,ds + \int_0^t\sigma(s,X_s,Y_s,Z_s)\,dW_s,\\
#   Y_t &= g(X_T) + \int_t^Tf(s,X_s,Y_s,Z_s)\,ds - \int_t^TZ_s\,dW_s.
# \end{aligned}
# \right.
# \end{equation*}
# Here, $X, Y, Z$ value in $\mathbb{R}^n, \mathbb{R}^m,\mathbb{R}^{m\times d}$.

# # Benchmark of SDE Simulations

# +
test_sde = FBSDE_LongSin_ResCalc(n=4, T=1.0, N=50, M=1024, r=0., sigma_0=0.4)

with torch.no_grad():
    dW = corrected_sample_dW(test_sde.h, test_sde.n, test_sde.N, test_sde.M,
                  dtype=TENSORDTYPE, device=DEVICE)
    t, X, Y, Z, dW = test_sde.calc_XYZ(test_sde.true_v, test_sde.true_u, dW)
    
terminal_error = test_sde.g(X[-1]).squeeze() - X[-1].sin().sum(dim=-1)*10/test_sde.d
running_error = test_sde.f(t[:-1], X[:-1], Y[:-1], Z[:-1]) - (-test_sde.r*Y[:-1]+.5*torch.exp(-3*test_sde.r*(test_sde.T-t[:-1]))*test_sde.sigma_0**2*(X[:-1].sin().sum(dim=-1, keepdim=True)*10/test_sde.d)**3)
martingale_error = (Z[:-1] @ dW.unsqueeze(-1)).squeeze() - torch.sum(Z[:-1].squeeze(-2)*dW,dim=-1)

assert terminal_error.abs().max() < 1e-6
assert running_error.abs().max() < 1e-6
assert martingale_error.abs().max() < 1e-6
# -

# # Loss of True Solutions

# +
test_sde = FBSDE_LongSin_ResCalc(n=4, T=1.0, N=50, M=1024, r=0., sigma_0=0.4)

dirac_loss, lambd_loss, gamma_loss = [], [], []
for _ in tqdm.trange(10):
    dW = corrected_sample_dW(test_sde.h, test_sde.n, test_sde.N, test_sde.M, dtype=TENSORDTYPE, device=DEVICE)

    dirac_loss.append(test_sde.calc_Res(test_sde.calc_MC(*test_sde.calc_XYZ(
        test_sde.true_v, test_sde.true_u, dW)), dirac=True).item())
    lambd_loss.append(test_sde.calc_Res(test_sde.calc_MC(*test_sde.calc_XYZ(
        test_sde.true_v, test_sde.true_u, dW)), dirac=False).item())
    gamma_loss.append(test_sde.calc_Res(test_sde.calc_MC(*test_sde.calc_XYZ(
        test_sde.true_v, test_sde.true_u, dW)), dirac=0.05).item())

print("δ-BML: ", format_uncertainty(np.mean(dirac_loss), np.std(dirac_loss)))
print("λ-BML: ", format_uncertainty(np.mean(lambd_loss), np.std(lambd_loss)))
print("γ-BML: ", format_uncertainty(np.mean(gamma_loss), np.std(gamma_loss)))

# -

# # Reformulate

# We follow the common methodology in machine learning to solve this problem. In particular, we define three parts necessary for learning as follows.
#
# 1. *Data*. This is defined as the sample path of Brownian motion, which is implemented in `bml.calc.sample_dW`.
#
# 2. *Model*. This is defined as the functions for predicting $Y$ and $Z$, which is implemented as `ynet` and `znet` of `bml.models.YZNet_FC3L`.
#
# 3. *Loss*. This is defined as a residuel loss w.r.t. to the FBSDE, which is implemented in `bml.calc.ResCalcMixin.calc_Res`.
#
# In addition, we define the following metric to evalute the performance of a model
#
# 1. *Y0*. This is the predict value of $Y_0$.
#
# 2. *Err Y0*. This is the relative error of the predicted $Y_0$.
#
# 3. *Aver X*. This is the distance between the predict $X$ and true $X$:
#    $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} | X^\theta_j -X_j)|^2\,\mu(jh). $$
#
# 4. *Aver Y*. This is the distance between the predict $Y$ and true $Y$:
#    $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} |v^\theta(ih, X^\theta_j) - v(ih, X_j)|^2\,\mu(jh). $$
#    
# 5. *Aver Z*. This is the distance between the predict $Z$ and true $Z$:
#    $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} |u^\theta(ih, X^\theta_j) - u(ih, X_j)|^2\,\mu(jh). $$
#    
# 6. *dist*. This is the objective function $\operatorname{dist}_\mu((Y^\theta,Z^\theta),(Y,Z))$:
#    $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} \Bigl\{ |v^\theta(ih, X^\theta_j) - v(ih, X_j)|^2 + h \sum_{i=j}^{N-1} |u^\theta(ih, X^\theta_i) - u(ih, X_i)|^2 \Bigr\}\mu(jh). $$

class Solver_LongSin(ResSolver):
    
    def __init__(self, sde, model, *, lr, dirac, quad_rule, correction):
        super().__init__(sde, model, lr=lr, dirac=dirac, quad_rule=quad_rule, correction=correction)

    def calc_metric_y0(self):
        t0 = self.sde.t[0:1, 0:1, 0:1]
        x0 = self.sde.x0.view(1, 1, -1)
        pred_y0 = self.model.ynet(t0, x0).flatten()[0].item()
        true_y0 = self.sde.true_y0.flatten()[0].item()
        return pred_y0, abs(pred_y0/true_y0 -1.)*100
    
    def calc_metric_averXYZ_and_dist(self):
        r'''Compare the difference between predict (X, Y, Z) and true (X, Y, Z)
        then average along the time dimension using sde.dirac'''
        data = self.sample_dW(self.sde.h, self.sde.n, self.sde.N, self.sde.M, 
                                dtype=TENSORDTYPE, device=DEVICE)
        t, pred_X, pred_Y, pred_Z, dW = self.sde.calc_XYZ(
            self.model.ynet, self.model.znet, data,
        )
        t, true_X, true_Y, true_Z, dW = self.sde.calc_XYZ(
            self.sde.true_v, self.sde.true_u, data,
        )
        
        # flat mtraix Z to vector
        pred_Z = pred_Z.reshape(1 + sde.N, sde.M, -1)
        true_Z = pred_Z.reshape(1 + sde.N, sde.M, -1)

        weight = self.sde.get_weight_mu(self.dirac)

        averX = torch.sum((pred_X - true_X).square()[:-1]
            * weight, dim=[0, 2]).mean().item()
        averY = torch.sum((pred_Y - true_Y).square()[:-1]
            * weight, dim=[0, 2]).mean().item()
        averZ = torch.sum((pred_Z - true_Z).square()[:-1]
            * weight, dim=[0, 2]).mean().item()
        
        dist = torch.sum(
            weight * ((pred_Y - true_Y).square()[:-1]
                + re_cumsum((pred_Z - true_Z).square()[:-1]*self.sde.h, dim=0)),
            dim=[0, 2]).mean().item()

        return averX, averY, averZ, dist

    def solve(self, max_steps, pbar=None):
        if pbar is None:
            pbar = tqdm.trange(max_steps)
        
        tab_logs, fig_logs = [], []

        optimizer = self.configure_optimizers()
        self.model.train()
        optimizer.zero_grad()

        for step in range(max_steps):
            loss = self.calc_loss()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.model.eval()
            with torch.no_grad():
                pred_y0, relative_error_y0 = self.calc_metric_y0()

            fig_logs.append({
                'step': step,
                'Y0': pred_y0,
                'val loss': relative_error_y0,
                'loss': loss.item(),
            })

        self.model.eval()
        with torch.no_grad():
            pred_y0, relative_error_y0 = self.calc_metric_y0()
            averX, averY, averZ, dist = self.calc_metric_averXYZ_and_dist()

        tab_logs.append({
            'Y0': pred_y0,
            'Err Y0': relative_error_y0,
            'averX': averX,
            'averY': averY,
            'averZ': averZ,
            'dist': dist,
            'val loss': relative_error_y0,
            'loss': fig_logs[-1]['loss'],
        })

        return tab_logs, fig_logs


# # Train

log_dir = os.path.join(LOGROOTDIR, time_dir())
os.makedirs(log_dir, exist_ok=False)

params = {
    'sde': {
        'n': 4,
        'T': 1.0,
        'N': 50,
        'M': 1024,
        'r': 0.0,
        'sigma_0': 0.4,
    },
    'model': {
        'hidden_size': 32,
    },
    'solver': {
        'lr': 1e-3,
        'dirac': 0.05,
        'quad_rule': 'trapezoidal',
        'correction': True,
    },
    'trainer': {
        'max_epoches': 1,
        'steps_per_epoch': 2000,
        'lr_decay_per_epoch': 0.9,
        
        # these lr are used at the first serveral epoches
        'warm_up_lr': [],
    },
}


# +
sde = FBSDE_LongSin_ResCalc(**params['sde'])
params['model']['n'] = sde.n
params['model']['m'] = sde.m
params['model']['d'] = sde.d

model = YZNet_FC3L(**params['model']).to(dtype=TENSORDTYPE, device=DEVICE)
solver = Solver_LongSin(sde, model, **params['solver'])

# add the usual lr to the last
params['trainer']['warm_up_lr'].append(solver.lr)

# create progress bar of total max_steps
max_epoches = params['trainer']['max_epoches']
pbar = tqdm.tqdm(total=params['trainer']['steps_per_epoch'], 
                 desc=f"Epoch: 1, Val Loss: 0.0")

tab_logs, fig_logs = [], []
for epoch in range(params['trainer']['max_epoches']):
    if epoch > 0:  # reset the progress bar at the start of each epoch
        pbar.reset(total=params['trainer']['steps_per_epoch'])
        pbar.set_description(
            f"Epoch: {epoch + 1}/{max_epoches}, Val Loss: {val_loss:.4f}")

    # select lr
    if epoch < len(params['trainer']['warm_up_lr']):
        solver.lr = params['trainer']['warm_up_lr'][epoch]
    else:
        solver.lr = params['trainer']['warm_up_lr'][-1]

    tab_log, fig_log = solver.solve(params['trainer']['steps_per_epoch'], pbar)
    
    val_loss = tab_log[-1]['val loss']
    solver.lr *= params['trainer']['lr_decay_per_epoch']
    
    # add a column to record the current number of epoches
    tab_logs += add_column_to_record(tab_log, 'epoch', [epoch] * len(tab_log))
    fig_logs += add_column_to_record(fig_log, 'epoch', [epoch] * len(fig_log))

# update the final val loss and close
pbar.set_description(f"Epoch: {epoch + 1}/{max_epoches}, Val Loss: {val_loss:.4f}")
pbar.close()

# +
tab_logs = pd.DataFrame(tab_logs)
fig_logs = pd.DataFrame(fig_logs)

tab_logs.to_csv(os.path.join(log_dir, "tab_logs.csv"), index=False)
fig_logs.to_csv(os.path.join(log_dir, "fig_logs.csv"), index=False)

print(tab_logs)

print(f"Results saved to {log_dir}")

# # Plotting

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Plot the running mean of the loss
ax1.plot(running_mean(fig_logs.loss.values, 5))
ax1.set_yscale('log')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training loss')

# Plot the predicted Y0 and true Y0
ax2.plot(fig_logs.Y0, label='Predicted Y0')
ax2.plot([sde.true_y0.item()]*len(fig_logs.Y0), label='True Y0')
ax2.set_xlabel('Step')
ax2.set_ylabel('Y0')
ax2.set_title('Y0 prediction')
ax2.legend()

# Plot the relative error of Y0
ax3.plot(.01*fig_logs['val loss'])
ax3.set_yscale('log')
ax3.set_xlabel('Step')
ax3.set_ylabel('Error')
ax3.set_title('Relative error of Y0 prediction')

fig.savefig(os.path.join(log_dir, 'fig.pdf'))

plt.show()
