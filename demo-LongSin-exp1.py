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
    dW = sample_dW(test_sde.h, test_sde.n, test_sde.N, test_sde.M,
                  dtype=TENSORDTYPE, device=DEVICE)
    t, X, Y, Z, dW = test_sde.calc_XYZ(test_sde.true_v, test_sde.true_u, dW)
    
terminal_error = test_sde.g(X[-1]).squeeze() - X[-1].sin().sum(dim=-1)*10/test_sde.d
running_error = test_sde.f(t[:-1], X[:-1], Y[:-1], Z[:-1]) - (-test_sde.r*Y[:-1]+.5*torch.exp(-3*test_sde.r*(test_sde.T-t[:-1]))*test_sde.sigma_0**2*(X[:-1].sin().sum(dim=-1, keepdim=True)*10/test_sde.d)**3)
martingale_error = (Z[:-1] @ dW.unsqueeze(-1)).squeeze() - torch.sum(Z[:-1].squeeze(-2)*dW,dim=-1)

assert terminal_error.abs().max() < 1e-12
assert running_error.abs().max() < 1e-12
assert martingale_error.abs().max() < 1e-12
# -

# # Loss of True Solutions

# +
test_sde = FBSDE_LongSin_ResCalc(n=4, T=1.0, N=50, M=1024, r=0., sigma_0=0.4)

dirac_loss, lambd_loss, gamma_loss = [], [], []
for _ in tqdm.trange(10):
    dW = sample_dW(test_sde.h, test_sde.n, test_sde.N, test_sde.M, dtype=TENSORDTYPE, device=DEVICE)

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
# 3. *Aver Y*. This is the distance between the predict $Y$ and true $Y$:
#    $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} |v^\theta(ih, X^\theta_i) - v(ih, X^\theta_i)|^2\,\mu(jh). $$

class Solver_LongSin(ResSolver):
    
    def __init__(self, sde, model, *, lr, dirac, quad_rule, correction):
        super().__init__(sde, model, lr=lr, dirac=dirac, quad_rule=quad_rule, correction=correction)

    def calc_metric_y0(self):
        t0 = self.sde.t[0:1, 0:1, 0:1]
        x0 = self.sde.x0.view(1, 1, -1)
        pred_y0 = self.model.ynet(t0, x0).flatten()[0].item()
        true_y0 = self.sde.true_y0.flatten()[0].item()
        return pred_y0, abs(pred_y0/true_y0 -1.)*100
    
    def calc_metric_averY(self):
        r'''Compare the difference between pred_Y and true_Y
        then average along the time dimension using sde.dirac
        '''
        data = self.sample_dW(self.sde.h, self.sde.n, self.sde.N, self.sde.M, 
                                dtype=TENSORDTYPE, device=DEVICE)
        t, X, Y, Z, dW = self.sde.calc_XYZ(self.model.ynet, self.model.znet, data)
        true_Y = self.sde.true_v(t, X)
        weight = self.sde.get_weight_mu(self.dirac)
        averY = ((Y - true_Y).abs().square()[:-1] * weight).sum(dim=0)
        averY = torch.sum(averY*averY, dim=-1)
        return averY.mean().item()

    def solve(self, pbar):
        tab_logs, fig_logs = [], []

        optimizer = self.configure_optimizers()
        self.model.train()
        optimizer.zero_grad()

        for step in pbar:
            loss = self.calc_loss()

            fig_logs.append({
                'step': step,
                'loss': loss.item(),
            })

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.model.eval()
        with torch.no_grad():
            pred_y0, relative_error_y0 = self.calc_metric_y0()
            averY = self.calc_metric_averY()

        tab_logs.append({
            'Y0': pred_y0,
            'Err Y0': relative_error_y0,
            'averY': averY,
            'loss': fig_logs[-1]['loss'],
        })

        return tab_logs, fig_logs


# # Train

# +
params = {
    'sde': {
        'n': 4,
        'T': 1.0,
        'N': 50,
        'M': 4096,
        'r': 0.0,
        'sigma_0': 0.4,
    },
    'model': {
        'hidden_size': 32,
    },
    'solver': {
        'lr': 5e-3,
        'dirac': False,
        'quad_rule': 'trapezoidal',
        'correction': True,
    },
    'trainer': {
        'max_epoches': 9,
        'steps_per_epoch': 200,
        'lr_decay_per_epoch': 0.6,
    },
}

sde = FBSDE_LongSin_ResCalc(**params['sde'])
params['model']['n'] = sde.n
params['model']['m'] = sde.m
params['model']['d'] = sde.d

model = YZNet_FC3L(**params['model']).to(dtype=TENSORDTYPE, device=DEVICE)
solver = Solver_LongSin(sde, model, **params['solver'])

tab_logs, fig_logs = [], []
for epoch in range(params['trainer']['max_epoches']):
    print(f"epoch: {epoch}")
    tab_log, fig_log = solver.solve(tqdm.trange(params['trainer']['steps_per_epoch']))
    solver.lr *= params['trainer']['lr_decay_per_epoch']
    
    # add a column to record the current number of epoches
    tab_logs += add_column_to_record(tab_log, 'epoch', [epoch] * len(tab_log))
    fig_logs += add_column_to_record(fig_log, 'epoch', [epoch] * len(fig_log))
# -

print(pd.DataFrame(tab_logs))

# +
fig_logs = pd.DataFrame(fig_logs)
plt.plot(fig_logs.loss)
plt.yscale('log')

plt.show()

