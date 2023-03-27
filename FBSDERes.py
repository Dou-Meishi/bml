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
import scipy
import pandas as pd
import torch
import tqdm
import seaborn as sns

# +
# local files
import bml.config

TENSORDTYPE = bml.config.TENSORDTYPE = torch.float64
DEVICE = bml.config.DEVICE = "cpu"

from bml.utils import *
from bml.fbsde_rescalc import FBSDE_LongSin_ResCalc
# -

# # Notation

# | Symbol | Definition |
# | ------ | ---------- |
# | $\mathbb{E}$  |    Mathematical Expectation | 
# | $\overline{\mathbb{E}}_r$  |    Empirical Expectation w.r.t. to index $r$ |
# | $h$ | Time step |
# | $n$ | Dimension of $X$ |
# | $N$ | Number of time intervals |
# | $M$ | Number of samples |

# # Problem Statement

# ## Convergence w.r.t. Time Step

# For the following FBSDE
# $$\left\{
# \begin{aligned}
# X_t &= x_0 + \int_0^tb(s,X_s,Y_s,Z_s)\,ds + \int_0^t\sigma(s,X_s,Y_s,Z_s)\,dW_s,\\
# Y_t &= g(X_T) + \int_t^Tf(s,X_s,Y_s,Z_s)\,ds - \int_t^TZ_s\,dW_s,
# \end{aligned}
# \right.$$
# consider the residual loss (of the triple $(X, Y, Z)$)
# $$ \mathit{Res}:= \operatorname{\mathbb{E}} \int_0^T \biggl|
# Y_t - \Bigl( g(X_T) + \int_t^Tf(s,X_s,Y_s,Z_s)\,ds - \int_t^TZ_s\,dW_s \Bigr)
# \biggr|^2\,\mu(dt).$$
#
# We want to study this loss. Specifically, we want to answer the following questions.
#
# 1. How to represent these process $X$, $Y$ and $Z$.
#
#    1. *Choice I.* Directly represent via sequence-to-sequence models, i.e., $\{X_t,0\leq t\leq T\}$ is the process transformed from the Brownian motion $\{W_t,0\leq t\leq T\}$, and $\{Y_t,0\leq t\leq T\}$ and $\{Z_t,0\leq t\leq T\}$ are two process transformed from the Brownian mition $W$ and the obtained forward process $X$. In this case, another residual loss based on the forward SDE should be introduced to regularize the forward process. 
#
#    2. *Choice II.* Represent $X$ with a sequence-to-sequence model while $Y$ and $Z$ are represented by functions of $X$:
#       $$ Y_t=v(t, X_t),\qquad Z_t=u(t,X_t) .$$
#       In this case, an additional residual loss based on the forward SDE should also be introduced to regularize the forward process.
#
#    3. *Choice III.* Introduce the feedback functions $v$ and $u$ at first. Substitute them into the forward SDE to eliminate $Y$ and $Z$:
#       $$ X_t = x_0 + \int_0^tb(s,X_s,v(s, X_s), u(s, X_s))\,ds  + \int_0^t\sigma(s,X_s,v(s, X_s), u(s, X_s))\,dW_s. $$
#       Then apply Euler-Maruyama method to calculate $X$. Finally, $Y$ and $Z$ are obtained by substituting $X$ into $v$ and $u$. This can be viewed as a tricky implementation of *Choice II* because by applying the Euler-Maruyama method we do not need to regularized $X$ manually.
#       
#    4. *Choice IV.* This is a tricky implementation of the *Choice I* by combining *Choice II* and *Choice III*. It works like *Choice III* but $Y$ and $Z$ are obtained from $X$ by two sequence-to-sequence models.
#       
# 2. Suppose the true solution of the BSDE is known, i.e., $Y_t=u(t, X_t)$ and $Z_t=v(t, X_t)$.
#
#    We want to design a numerical scheme such that $\mathit{Res}$ would converge to zero as the time step approaches zero.

# # Loss of True Solution with Choice III

# Let's consider the computation of the forward process from the following difference equation. To enhance clarity, we will use a superscript ${}^\theta$ to indicate all approximated processes. Also, for the sake of notation simplicity, we will use subscript ${}_i$ to denote the time point $ih$ and drop the dependency of $\omega_r$.
#
# The difference equation is as follows:
# $$ X^\theta_{i+1} = X^\theta_i + b(ih, X^\theta_i, v^\theta(X^\theta_i), u^\theta(X^\theta_i))\,h + \sigma(ih,  X^\theta_i, v^\theta(X^\theta_i), u^\theta(X^\theta_i))[W_{i+1} - W_i].$$
#
# Then we could calculate the (discrete) residual loss as
# $$ \mathit{Res}(\theta;h,M) = \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} |\mathit{MC}(\theta;h,M;r,j)|^2\,\mu(jh),$$
# where 
# $$ \mathit{MC}(\theta;h,M;r,j):= v(X^\theta_j) - \biggl( g(X^\theta_N) + h\sum_{i=j}^{N-1} f(X^\theta_i, v^\theta(X^\theta_i), u^\theta(X^\theta_i)) - \sum_{i=j}^{N-1} u^\theta(X^\theta_i)[W_{i+1} - W_{i}] \biggr). $$

# Now we assume that we are lucky enough such that $v^\theta$ and $u^\theta$ exactly match the true form $v$ and $u$. Denote by $\theta_*$ the corresponding hyperparameter. We want to study the behavior of $\mathit{Res}(\theta_*;h,M)$ as $h\to 0$ and $M\to\infty$. Note that if these integrals can be obtained without discretization errors, then $\mathit{Res}(\theta_*)$ is strictly zero. Thus, all we want to do is to verify the limit equation
# $$ \lim_{\substack{h\to0\\M\to\infty}}\mathit{Res}(\theta_*;h,M) = \mathit{Res}(\theta_*).$$

# +
sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=512)
dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
mc = sde.calc_MC(t, X, Y, Z, dW)

plt.plot(mc.squeeze(-1).var(dim=-1).cpu().numpy())

# +
dirac_loss, lambd_loss, gamma_loss = [], [], []
for _ in tqdm.trange(10):
    dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)

    dirac_loss.append(sde.calc_Res(sde.calc_MC(*sde.calc_XYZ(
        sde.true_v, sde.true_u, dW)), dirac=True).item())
    lambd_loss.append(sde.calc_Res(sde.calc_MC(*sde.calc_XYZ(
        sde.true_v, sde.true_u, dW)), dirac=False).item())
    gamma_loss.append(sde.calc_Res(sde.calc_MC(*sde.calc_XYZ(
        sde.true_v, sde.true_u, dW)), dirac=0.05).item())

print("δ-BML: ", format_uncertainty(np.mean(dirac_loss), np.std(dirac_loss)))
print("μ-BML: ", format_uncertainty(np.mean(lambd_loss), np.std(lambd_loss)))
print("γ-BML: ", format_uncertainty(np.mean(gamma_loss), np.std(gamma_loss)))
# -

# ## As Time Step Changes

# +
n_runs = 10
scale_values = [1, 2, 4, 8]

for scale in tqdm.tqdm(scale_values):
    variances = []
    for i in range(n_runs):
        sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50*scale, M=512)
        dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
        t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)

        empirical_var = sde.calc_MC(t, X, Y, Z, dW).squeeze(-1).var(dim=-1)
        assert len(empirical_var.shape) == 1

        variances.append(empirical_var.cpu().numpy())
    variances = np.array(variances)
    mean_var = np.mean(variances, axis=0)
    std_var = np.std(variances, axis=0)

    plt.plot(mean_var[::-1], label=f"scale={scale}")
    plt.fill_between(range(len(mean_var)), (mean_var - std_var)[::-1], (mean_var + std_var)[::-1], alpha=0.2)

plt.legend()
plt.title("As time step become finer")
plt.xlabel("number of time steps")
plt.ylabel("Variance")
plt.show()
# -

# ## As Sample Size Changes

# +
n_runs = 10
scale_values = [1, 2, 4, 8]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, scale in enumerate(scale_values):
    variances = []
    for j in range(n_runs):
        sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=128*scale)
        dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
        t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)

        empirical_var = sde.calc_MC(t, X, Y, Z, dW).squeeze(-1).var(dim=-1)
        variances.append(empirical_var.cpu().numpy())
    variances = np.array(variances)
    mean_var = np.mean(variances, axis=0)
    std_var = np.std(variances, axis=0)

    row = i // 2
    col = i % 2
    axs[row, col].plot(mean_var, label=f"scale={scale}")
    axs[row, col].fill_between(range(len(mean_var)), mean_var - std_var, mean_var + std_var, alpha=0.2)
    axs[row, col].legend()
    axs[row, col].set_xlabel("number of samples in dW")
    axs[row, col].set_ylabel("Variance")
    axs[row, col].set_title(f"scale={scale}")

plt.suptitle("As sample size increases")
plt.tight_layout()
plt.show()


# -

# ## A try on high order integrators

# We may use high order quadrature rules when approximating the time integral. More specifically, we want to approximate the time integral $\int_t^Tf^\theta(s,X^\theta_s,v(s, X^\theta_s), u(s, X^\theta_s))\,ds$ with data 
# $$f_i^\theta := f(X^\theta_i, v^\theta(X^\theta_i), u^\theta(X^\theta_i)), \quad i=0, 1, 2, \ldots, j$$.
#
# For the *rectangle rule*, we have
# $$ \int_t^Tf^\theta(s,X^\theta_s,v(s, X^\theta_s), u(s, X^\theta_s))\,ds \approx \sum_{i=j}^{N-1}f_i^\theta\,h.$$
# For the *trapezoidal rule*, we have
# $$ \int_t^Tf^\theta(s,X^\theta_s,v(s, X^\theta_s), u(s, X^\theta_s))\,ds \approx \sum_{i=j}^{N-1}\frac{f_i^\theta+f^\theta_{i+1}}{2}\,h.$$
# For the *Simpson's rule*, we have (assume $j$ is even)
# $$ \begin{aligned}
# \int_t^Tf^\theta(s,X^\theta_s,v(s, X^\theta_s), u(s, X^\theta_s))\,ds &\approx \sum_{m=0}^{j/2-1}\frac{h}{3}\,\bigl(f_{2m}+4f_{2m+1}+f_{2m+2}\bigr) \\
# &= \frac{h}{3}\biggl(\sum_{i=0}^{j-1}f_j + \sum_{i=1}^{j}f_i + 2\sum_{m=0}^{j/2-1}f_{2m+1}\biggr)
# \end{aligned}$$

# +
def calc_MC_with_quadrature(self, t, X, Y, Z, dW, *, rule='rectangle'):
    f = self.f(t, X, Y, Z)
    if rule == 'rectangle':
        return Y[:-1] - (self.g(X[-1:]) + self.h * re_cumsum(
            f[:-1], dim=0
        ) - re_cumsum(Z[:-1] @ dW.unsqueeze(-1), dim=0).squeeze(-1))
    elif rule == 'trapezoidal':
        return Y[:-1] - (self.g(X[-1:]) + self.h * re_cumsum(
            (f[:-1]+f[1:])/2, dim=0
        ) - re_cumsum(Z[:-1] @ dW.unsqueeze(-1), dim=0).squeeze(-1))
    else:
        raise ValueError(f"Unrecognized rule {rule}")
        
FBSDE_LongSin_ResCalc.calc_MC_with_quadrature = calc_MC_with_quadrature

# +
for rule in ['rectangle', 'trapezoidal']:
    sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=1024)
    dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
    t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
    empirical_var = sde.calc_MC_with_quadrature(
        t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"rule={rule}")

    plt.legend()

plt.title("when different integrators are applied")
# -

# ## Sample dW with Specific Constraints

# **Problem.** How to sample these $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ such that
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0_{n},\quad \overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI_{n},$$
# where $0_n$ is the zero vector in $\mathbb{R}^n$ and $I_n$ is the $n\times n$ identity matrix.

# Because n normal r.v. can be viewed as a n-dimensional normal r.v., the above problem is equivalent to generate $M$ samples of $nN$ i.i.d. real valued normal variables $\{\epsilon_u(\omega_r)\}_{\substack{0\leq u\leq nN-1 \\ 0\leq r\leq M-1}}$ such that
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_u(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r\epsilon_u(\omega_r)\epsilon_v(\omega_r)=h\delta_{uv}.$$
# This is again equivalent to forcing $\{\epsilon_u, 0\leq u\leq nN\}$ to be an orthogonal vector basis of $\mathbb{R}^M$, where we manually add the $\epsilon_{nN} = \mathbb{1}_m$, which is the all-$1$ vector. **Hence, we should have $M \geq nN+1$ to achieve this.**

# +
# Create empty arrays to store the results
sampled_var = []
corrected_var = []
sigma_values = [0.4, 0.1]

# Repeat the procedure 10 times for each value of sigma
for sigma in sigma_values:
    sampled_var_sigma = []
    corrected_var_sigma = []
    for i in tqdm.trange(10):
        # Sample dW and calculate empirical variance for the sampled dW
        sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=256, sigma_0=sigma)
        dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
        t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
        empirical_var = sde.calc_MC_with_quadrature(
            t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
        sampled_var_sigma.append(empirical_var.cpu().numpy())
        
        # Sample corrected dW and calculate empirical variance for the corrected dW
        sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=256, sigma_0=sigma)
        dW = corrected_sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
        t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
        empirical_var = sde.calc_MC_with_quadrature(
            t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
        corrected_var_sigma.append(empirical_var.cpu().numpy())
    
    # Append the results for this value of sigma to the overall arrays
    sampled_var.append(sampled_var_sigma)
    corrected_var.append(corrected_var_sigma)

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=1)
sampled_std = np.std(sampled_var, axis=1)
corrected_mean = np.mean(corrected_var, axis=1)
corrected_std = np.std(corrected_var, axis=1)

# Create the figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, ax in enumerate(axs):
    sigma = sigma_values[i]
    sampled_line = ax.plot(sampled_mean[i], label='sampled')
    ax.fill_between(range(len(sampled_mean[i])), sampled_mean[i] - sampled_std[i], sampled_mean[i] + sampled_std[i], alpha=0.2)
    corrected_line = ax.plot(corrected_mean[i], label='corrected')
    ax.fill_between(range(len(corrected_mean[i])), corrected_mean[i] - corrected_std[i], corrected_mean[i] + corrected_std[i], alpha=0.2)
    ax.set_title(f'$\\sigma_0$ = {sigma}')
    ax.legend()

plt.show()

# +
# Create empty arrays to store the results
sampled_var = []
corrected_var = []
sigma_values = [0.4, 0.1]
M_scales = [1, 2, 4, 8]

# Iterate through M_scales
for scale in M_scales:
    M = 256 * scale
    sampled_var_M = []
    corrected_var_M = []

    for sigma in sigma_values:
        sampled_var_sigma = []
        corrected_var_sigma = []

        for i in tqdm.trange(10):
            # Sample dW and calculate empirical variance for the sampled dW
            sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=M, sigma_0=sigma)
            dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
            t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
            empirical_var = sde.calc_MC_with_quadrature(
                t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
            sampled_var_sigma.append(empirical_var.cpu().numpy())

            # Sample corrected dW and calculate empirical variance for the corrected dW
            sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=M, sigma_0=sigma)
            dW = corrected_sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
            t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
            empirical_var = sde.calc_MC_with_quadrature(
                t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
            corrected_var_sigma.append(empirical_var.cpu().numpy())
        
        sampled_var_M.append(sampled_var_sigma)
        corrected_var_M.append(corrected_var_sigma)

    sampled_var.append(sampled_var_M)
    corrected_var.append(corrected_var_M)

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=2)
sampled_std = np.std(sampled_var, axis=2)
corrected_mean = np.mean(corrected_var, axis=2)
corrected_std = np.std(corrected_var, axis=2)

# Create the figure with multiple rows and columns of subplots
fig, axs = plt.subplots(len(M_scales), len(sigma_values), figsize=(8, 3 * len(M_scales)))

for row, M_scale in enumerate(M_scales):
    for col, sigma in enumerate(sigma_values):
        ax = axs[row, col]
        sampled_line = ax.plot(sampled_mean[row, col], label='sampled')
        ax.fill_between(range(len(sampled_mean[row, col])), sampled_mean[row, col] - sampled_std[row, col], sampled_mean[row, col] + sampled_std[row, col], alpha=0.2)
        corrected_line = ax.plot(corrected_mean[row, col], label='corrected')
        ax.fill_between(range(len(corrected_mean[row, col])), corrected_mean[row, col] - corrected_std[row, col], corrected_mean[row, col] + corrected_std[row, col], alpha=0.2)
        ax.set_title(f'M = {256 * M_scale}, $\\sigma_0$ = {sigma}')
        ax.legend()

plt.tight_layout()
plt.show()
# -

# ## Conclusion of Choice III

# In conclusion, we have examined the convergence behavior of the residual loss of the true solution with respect to the time step size and the number of Monte Carlo samples for a general type of fully coupled FBSDEs. Our observations reveal that the residual loss of FBSDEs behaves similarly to their SDE counterparts. Our findings are as follows:
#
# 1. The discrete residual loss of the true solution to the FBSDE is non-zero, but it approaches zero as the time step used for discretization approaches zero.
#
# 2. Monte Carlo methods can estimate the residual loss using sample paths of the Brownian motion. Larger sample sizes lead to smaller variances of the residual loss, while the mean value remains unaffected.
#
# 3. The "manually forcing orthogonality" technique used in SDEs can also be applied to FBSDEs.

# ## A very interesting trend

# It has been observed that when the sample size approaches zero while applying the forcing orthogonality trick, the residual loss mean converges to zero even if the time step is not. Additionally, the variance is very small.
#
# Without formal mathematical proof, it is hypothesized that this occurs because the residual loss is uniformly distributed in all directions. When dW is directly sampled, the loss is measured in random directions and lengths. However, with the forcing orthogonality trick, the directions are orthogonal and fixed in length. When the sample size is large, all directions are covered, making the trick unnecessary.

# +
# Create empty arrays to store the results
sampled_var = []
corrected_var = []
sigma_values = [0.4, 0.1]
M_scales = [1, 2, 4, 8]

# Iterate through M_scales
for scale in M_scales:
    M = 256 // scale
    sampled_var_M = []
    corrected_var_M = []

    for sigma in sigma_values:
        sampled_var_sigma = []
        corrected_var_sigma = []

        for i in tqdm.trange(10):
            # Sample dW and calculate empirical variance for the sampled dW
            sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=M, sigma_0=sigma)
            dW = sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
            t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
            empirical_var = sde.calc_MC_with_quadrature(
                t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
            sampled_var_sigma.append(empirical_var.cpu().numpy())

            # Sample corrected dW and calculate empirical variance for the corrected dW
            sde = FBSDE_LongSin_ResCalc(n=4, T=1., N=50, M=M, sigma_0=sigma)
            dW = corrected_sample_dW(sde.h, sde.n, sde.N, sde.M, dtype=TENSORDTYPE, device=DEVICE)
            t, X, Y, Z, dW = sde.calc_XYZ(sde.true_v, sde.true_u, dW)
            empirical_var = sde.calc_MC_with_quadrature(
                t, X, Y, Z, dW, rule=rule).squeeze(-1).var(dim=-1)
            corrected_var_sigma.append(empirical_var.cpu().numpy())
        
        sampled_var_M.append(sampled_var_sigma)
        corrected_var_M.append(corrected_var_sigma)

    sampled_var.append(sampled_var_M)
    corrected_var.append(corrected_var_M)

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=2)
sampled_std = np.std(sampled_var, axis=2)
corrected_mean = np.mean(corrected_var, axis=2)
corrected_std = np.std(corrected_var, axis=2)

# Create the figure with multiple rows and columns of subplots
fig, axs = plt.subplots(len(M_scales), len(sigma_values), figsize=(8, 3 * len(M_scales)))

for row, M_scale in enumerate(M_scales):
    for col, sigma in enumerate(sigma_values):
        ax = axs[row, col]
        sampled_line = ax.plot(sampled_mean[row, col], label='sampled')
        ax.fill_between(range(len(sampled_mean[row, col])), sampled_mean[row, col] - sampled_std[row, col], sampled_mean[row, col] + sampled_std[row, col], alpha=0.2)
        corrected_line = ax.plot(corrected_mean[row, col], label='corrected')
        ax.fill_between(range(len(corrected_mean[row, col])), corrected_mean[row, col] - corrected_std[row, col], corrected_mean[row, col] + corrected_std[row, col], alpha=0.2)
        ax.set_title(f'M = {256 // M_scale}, $\\sigma_0$ = {sigma}')
        ax.legend()

plt.tight_layout()
plt.show()
