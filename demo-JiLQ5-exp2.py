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
from bml.trainer import Trainer
from bml.fbsde_rescalc import FBSDE_JiLQ5_ResCalc
from bml.calc import ResSolver
from bml.models import YZNet_FC3L, YZNet_FC2L
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

class Solver_JiLQ5(ResSolver):
    
    def __init__(self, sde, model, *, lr, dirac, quad_rule, correction):
        super().__init__(sde, model, lr=lr, dirac=dirac, quad_rule=quad_rule, correction=correction)

    def calc_metric_cost(self):
        data = self.sample_dW(self.sde.h, self.sde.d, self.sde.N, self.sde.M, 
                                dtype=TENSORDTYPE, device=DEVICE)
        t, pred_X, pred_Y, pred_Z, dW = self.sde.calc_XYZ(
            self.model.ynet, self.model.znet, data,
        )
        u = (pred_Y + pred_Z.squeeze(-1)) / 2
        cost = self.sde.calc_cost(t, pred_X, u)
        return {'Cost': cost.mean().item()}


class Trainer_JiLQ5(Trainer):
    
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)


# # Train

log_dir = os.path.join(LOGROOTDIR, time_dir())
os.makedirs(log_dir, exist_ok=False)

params = {
    'sde': {
        'n': 100,
        'T': .1,
        'N': 25,
        'M': 64,
    },
    'model': {
        'hidden_size': 16,
    },
    'solver': {
        'lr': 2e-3,
        'dirac': 0.05,
        'quad_rule': 'trapezoidal',
        'correction': True,
    },
    'trainer': {
        'max_epoches': 1,
        'steps_per_epoch': 1999,
        'lr_decay_per_epoch': 0.9,
        
        # these lr are used at the first serveral epoches
        'warm_up_lr': [],
    },
}


# +
sde = FBSDE_JiLQ5_ResCalc(**params['sde'])
params['model']['n'] = sde.n
params['model']['m'] = sde.m
params['model']['d'] = sde.d

model = YZNet_FC2L(**params['model']).to(dtype=TENSORDTYPE, device=DEVICE)
solver = Solver_JiLQ5(sde, model, **params['solver'])
trainer = Trainer(**params['trainer'])

tab_logs, fig_logs = trainer.train(solver)
# -

# # Save and Show Results

# +
tab_logs = pd.DataFrame(tab_logs)
fig_logs = pd.DataFrame(fig_logs)

tab_logs.to_csv(os.path.join(log_dir, "tab_logs.csv"), index=False)
fig_logs.to_csv(os.path.join(log_dir, "fig_logs.csv"), index=False)

print(tab_logs)

print(f"Results saved to {log_dir}")

# +
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Plot the running mean of the loss
ax1.plot(running_mean(fig_logs.loss.values, 5))
ax1.set_yscale('log')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training loss')

# Plot the predicted Y0 and true Y0
ax2.plot(fig_logs.Y0, label='Predicted Y0')
ax2.plot(fig_logs['True Y0'], label='True Y0')
ax2.set_xlabel('Step')
ax2.set_ylabel('Y0')
ax2.set_title('Y0 prediction')
ax2.legend()

# Plot the relative error of Y0
ax3.plot(.01*fig_logs['Err Y0'])
ax3.set_yscale('log')
ax3.set_xlabel('Step')
ax3.set_ylabel('Error')
ax3.set_title('Relative error of Y0 prediction')

fig.savefig(os.path.join(log_dir, 'fig.pdf'))

plt.show()
