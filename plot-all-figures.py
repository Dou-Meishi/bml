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
import yaml

# +
# local files
import bml.config

# change config before loading other modules
TENSORDTYPE = bml.config.TENSORDTYPE = torch.float32
DEVICE = bml.config.DEVICE = "cpu"

from bml.utils import *
from bml.fbsde_rescalc import *
# -

# # Specific data

# +
yaml_str = """
FuSinCos EXP1:
  - dirac: True
    log_dir: outputs/230329-0234
    commit ID: 835a4d0
    label: "δ-BML"
  - dirac: False
    log_dir: outputs/230329-0237
    commit ID: 38f37fe
    label: "λ-BML"
  - dirac: 0.05
    log_dir: outputs/230329-0242
    commit ID: 48a7a6a
    label: "γ-BML"
LongSin EXP1:
  - dirac: True
    log_dir: outputs/230329-0017
    commit ID: d94da2c
    label: "δ-BML"
  - dirac: False
    log_dir: outputs/230329-0003
    commit ID: a21e2e9
    label: "λ-BML"
  - dirac: 0.05
    log_dir: outputs/230329-0011
    commit ID: 752f8be
    label: "γ-BML"
JiLQ EXP1:
  - dirac: True
    log_dir: outputs/230329-0049
    commit ID: 0faf7ae
    label: "δ-BML"
  - dirac: False
    log_dir: outputs/230329-0046
    commit ID: e4625c8
    label: "λ-BML"
  - dirac: 0.05
    log_dir: outputs/230328-2036
    commit ID: f8606f6
    label: "γ-BML"
JiLQ EXP2:
  - dirac: True
    log_dir: outputs/230328-2059
    commit ID: 1728a62
    label: "δ-BML"
  - dirac: False
    log_dir: outputs/230328-2051
    commit ID: 32c0e77
    label: "λ-BML"
  - dirac: 0.05
    log_dir: outputs/230329-0114
    commit ID: 41771e9
    label: "γ-BML"
"""

data_paths = yaml.safe_load(yaml_str)

# -

{r['label']: r['log_dir'] for r in data_paths['FuSinCos EXP1']}


# # Plot

def plot_results(fig_logs, sde):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    line_styles = {
        "δ-BML": "-",
        "λ-BML": "--",
        "γ-BML": "-.",
    }

    # Plot the running mean of the loss
    for name, fig_data in fig_logs.items():
        linestyle = line_styles.get(name, "-")
        ax1.plot(running_mean(fig_data.loss.values, 5), linewidth=1.8, linestyle=linestyle, label=name)
    ax1.set_yscale('log')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training loss')
    ax1.legend()

    # Plot the predicted Y0 and true Y0
    for name, fig_data in fig_logs.items():
        linestyle = line_styles.get(name, "-")
        ax2.plot(fig_data.Y0.values, linewidth=1.8, linestyle=linestyle, label=f"Predicted Y0 by {name}")
    ax2.plot([sde.true_y0[0].item()]*max(
        len(fig_data.Y0.values) for fig_data in fig_logs.values()), 
        linewidth=1.8, linestyle=':', label='True Y0')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Y0')
    ax2.set_title('Y0 prediction')
    ax2.legend()

    # Plot the relative error of Y0
    for name, fig_data in fig_logs.items():
        linestyle = line_styles.get(name, "-")
        ax3.plot(running_mean(0.01*fig_data['val loss'].values, 5), linewidth=1.8, linestyle=linestyle, label=f"{name}")
    ax3.set_yscale('log')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Error')
    ax3.set_title('Relative error of Y0 prediction')
    ax3.legend()

    fig.subplots_adjust(wspace=0.4, bottom=0.2)
    fig.text(0.5, 0.05, 'SDE models comparison', ha='center', va='center')

    # Invert direction of ticks
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='both', direction='in', width=1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['top'].set_linewidth(1.0)

    return fig


# ## FuSinCos EXP1

# +
sde = FBSDE_FuSinCos_ResCalc(n=1)

fig_logs = {
    r['label']: pd.read_csv(os.path.join(r['log_dir'], 'fig_logs.csv')) 
    for r in data_paths['FuSinCos EXP1']
}

fig = plot_results(fig_logs, sde)
# -

# ## LongSin

# +
sde = FBSDE_LongSin_ResCalc(n=4, T=1.0, r=0., sigma_0=0.4)

fig_logs = {
    r['label']: pd.read_csv(os.path.join(r['log_dir'], 'fig_logs.csv')) 
    for r in data_paths['LongSin EXP1']
}

fig = plot_results(fig_logs, sde)
# -

# ## JiLQ EXP1

# +
sde = FBSDE_JiLQ5_ResCalc(n=5, T=0.1)

fig_logs = {
    r['label']: pd.read_csv(os.path.join(r['log_dir'], 'fig_logs.csv')) 
    for r in data_paths['JiLQ EXP1']
}

fig = plot_results(fig_logs, sde)
# -

# ## JiLQ EXP2

# +
sde = FBSDE_JiLQ5_ResCalc(n=100, T=0.1)

fig_logs = {
    r['label']: pd.read_csv(os.path.join(r['log_dir'], 'fig_logs.csv')) 
    for r in data_paths['JiLQ EXP2']
}

fig = plot_results(fig_logs, sde)
