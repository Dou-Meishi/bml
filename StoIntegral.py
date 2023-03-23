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

# local files
from yznets import *
# -

TENSORDTYPE = torch.float64
DEVICE = "cpu"


# # Notation

# | Symbol | Definition |
# | ------ | ---------- |
# | $\mathbb{E}$  |    Mathematical Expectation | 
# | $\overline{\mathbb{E}}_r$  |    Empirical Expectation w.r.t. to index $r$ |

# # Convergence w.r.t. Time Step

# For the following SDE
# $$ X_t = x_0 + \int_0^tb(s,X_s)\,ds + \int_0^t\sigma(s,X_s)\,dW_s,$$
# consider the residue loss (of the stochastic process $X$)
# $$ \mathit{Res}:= \operatorname{\mathbb{E}}\int_0^T\biggl|X_t - \Bigl(x_0 + \int_0^tb(s,X_s)\,ds + \int_0^t\sigma(s,X_s)\,dW_s\Bigr)\biggr|^2\,\mu(dt).$$
# We want to study this loss. Specifically, we want to answer the following questions.
#
# 1. How to compute $X_t$.
#
#    1. *Choice I.* Directly represent via neural networks (`Seq2Seq` model).
#    
#    2. *Choice II.* Compute by difference equation starting at $X_0=x_0$.
#    
# 2. Suppose the true solution $X_t=X(t,\omega)$ is known. 
#
#    We want to design a numerical scheme for computing $\mathit{Res}$ (with aforementioned scheme for computing $X_t$) such that $\mathit{Res}$ would converge to zero. 

# # Choice I with Naive Euler-Maruyama Discretization

# Let us assume that we choose to fit the process $\{X_t,0\leq t\leq T\}$ directly from $\{W_t,0\leq T\}$, i.e., $X_t\approx X^\theta(t,\omega)$. Then we substitue this into the definition of $\mathit{Res}$ and obtain
# $$ \mathit{Res}(\theta):= \operatorname{\mathbb{E}}\int_0^T\biggl|X^\theta(t,\omega) - \Bigl(x_0 + \int_0^tb(s,X^\theta(s,\omega))\,ds + \int_0^t\sigma(s,X^\theta(s,\omega))\,dW_s\Bigr)\biggr|^2\,\mu(dt).$$

# Hence, we need only some appropriate integrator for calculating these integrals. For example, we may apply the simple rectangle rule for the time integral and Euler-Maruyama rule for the stochastic integral:
# $$ \mathit{Res}(\theta; h, M):= \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1}\biggl|X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}b(ih,X^\theta(ih,\omega_r))h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr)\biggr|^2\,\mu(jh).$$

# For clarity, let us consider the temporal difference error at a single time point and sample event
# $$ \mathit{TD}(\theta; h, M; r,j):= X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}b(ih,X^\theta(ih,\omega_r))h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr).$$
# With this notation, we have 
# $$ \mathit{Res}(\theta;h,M) = \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} \bigl|\mathit{TD}(\theta;h,M;r,j)\bigr|^2\,\mu(jh).$$

# Now assume that we are lucky enough such that we happen to find a representation which  coincides with the true solution, i.e., $X^{\theta_*}(t,\omega)=ae^{\gamma t}\|W(t,\omega)\|^2$. We want to study the behavior of $\mathit{Res}(\theta_*;h,M)$ as $h\to 0$ and $M\to\infty$.

# ## Example WSquareSDE

# **Example.** (`WSqureSDE`)  We can choose $x_0=0$ and
# $$ \begin{cases}
# b(t,x) = \gamma x + nae^{\gamma t},\\
# \sigma(t,x) = 2ae^{\gamma t}W_t.
# \end{cases}$$
# Then the true solution is $X_t=ae^{\gamma t}\|W_t\|^2$.
#
# When $a=1/n,~\gamma=0$, this SDE reduces to (denote by $n$ the dimension of $W$)
# $$ \frac{1}{n}\|W_t\|^2 = t + \frac{2}{n}\int_0^t\langle W_s,dW_s\rangle.$$
#
# Let us study the statistics of $\mathit{TD}(\theta_*;h,M;r,j)$ with $a=1/n,~\gamma=0$. In this case,
# $$ \mathit{TD}(\theta_*;h,M;r,j) = \frac{1}{n}\|W(jh,\omega_r)\|^2 - jh - \frac{2}{n}\sum_{i=0}^{j-1}\langle  W(ih,\omega_r), W((i+1)h,\omega_r) - W(ih,\omega_r)\rangle .$$

# **Analytic Solution**. It is not very hard to show that if $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ satisfies the statistics of $N$ i.i.d. $n$-dimensional standard normal distribution, which containts but not limit to
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI,$$
# where $\epsilon_i(\omega_r):=W((i+1)h,\omega_r) - W(ih,\omega_r)$, then
# $$ \overline{\operatorname{\mathbb{E}}}_r \mathit{TD}(\theta_*;h,M;r,j) = 0, \quad \overline{\operatorname{\mathbb{E}}}_r |\mathit{TD}(\theta_*;h,M;r,j)|^2 = 2jh^2/n.$$
#
# *Proof.* Rewrite $\mathit{TD}$ with $\{\epsilon_i\}_{i=0}^{N-1}$,
# $$\mathit{TD}(\theta_*;h,M;r,j) = \frac{1}{n}\Bigl\|\sum_{i=0}^{j-1}\epsilon_i\Bigr\|^2 - jh - \frac{2}{n}\sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle.$$
# With the given hypothesess, it can be shown that
# $$ \overline{\operatorname{\mathbb{E}}}_r \Bigl\|\sum_{i=0}^{j-1}\epsilon_i\Bigr\|^2 = jnh,\quad \overline{\operatorname{\mathbb{E}}}_r \Bigl\|\sum_{i=0}^{j-1}\epsilon_i\Bigr\|^4 = (3jn-2n+2)jnh^2.$$
# $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle = 0,\quad \overline{\operatorname{\mathbb{E}}}_r \Bigl|\sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle\Bigr|^2 = \frac{1}{2}j(j-1)(nh)^2.$$
# $$ \overline{\operatorname{\mathbb{E}}}_r \left[\Bigl\|\sum_{i'=0}^{j-1}\epsilon_{i'}\Bigr\|^2 \sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle \right] = j(j-1)(nh)^2.$$
# Desired properties hold when these equations hold.
#
# *Q.E.D.*

def calc_TD_WSquareSDE(dW, h):
    W = torch.cat([torch.zeros_like(dW[0:1]), dW]).cumsum(dim=0)
    return torch.sum(W*W,dim=-1)/W.shape[-1] - torch.arange(W.shape[0], device=DEVICE).view(-1, 1)*h - torch.cat([torch.zeros_like(dW[0:1, :, 0]), torch.sum(W[:-1]*dW, dim=-1)]).cumsum(dim=0)/W.shape[-1]*2


def sample_dW(h, n, N, M):
    r'''Return dW[i,r,*] of M independent sample paths of
    n dimensional standard Brownian Motion with time step h 
    and final time T=hN.'''
    return torch.randn(N, M, n, dtype=TENSORDTYPE, device=DEVICE)*np.sqrt(h)


# +
# sample M samples of n dim from normal dist with std h
h = 0.01
n = 10
N = 100      # 1.0/0.01
M = 1024

dW = sample_dW(h, n, N, M)

# +
empirical_var = calc_TD_WSquareSDE(dW, h).var(dim=-1)
predict_var = 2*h*h/n*torch.arange(N+1, device=DEVICE)

plt.plot(empirical_var.cpu().numpy())
plt.plot(predict_var.cpu().numpy())


# -

# ## Generalized WSquareSDE

class WSquareSDE(object):
    
    def __init__(self, n, *, a=None, gamma=None):
        self.n = n
        self.a = a if a is not None else 1/n
        self.gamma = gamma if gamma is not None else 0.

    def calc_TD(self, dW, h):
        r'''dW is the increment of standard n-dimensional
        Brownian motion with shape [j, r, *] and time step h'''
        def shift_cumsum(t):
            return torch.cat([torch.zeros_like(t[:1]), t.cumsum(dim=0)])

        weight = torch.exp(self.gamma*h*torch.arange(
            dW.shape[0]+1, device=DEVICE)).view(-1, 1, 1)
        W = shift_cumsum(dW)
        X = torch.sum(W*W, dim=-1, keepdim=True)*self.a*weight

        b = self.n*self.a*weight + self.gamma*X
        timeint = shift_cumsum(b[:-1])*h
        stoint = shift_cumsum(torch.sum(
            weight[:-1]*W[:-1]*dW, dim=-1, keepdim=True)
                             )*self.a*2
        return (X - timeint - stoint).squeeze(-1)


# +
for scale in [1, 2, 4, 8]:
    _dW = sample_dW(h/scale, 100, N*scale, 1024)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=.99, a=0.1
                              ).calc_TD(_dW, h/scale).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"scale={scale}")

    plt.legend()

plt.title("As time step become finer")

# +
for scale in [1, 2, 4, 8]:
    _dW = sample_dW(h, 100, N, 128*scale)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=.99, a=0.1
                              ).calc_TD(_dW, h).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"scale={scale}")

    plt.legend()

plt.title("As sample size increase")
# -

# ## Replotting the final figure with mean and standard deviation

# In this section, we will replot the final figure showing the convergence behavior of the residue loss for different time step sizes and sample sizes, this time including the mean and standard deviation for each line.
#
# To achieve this, we will modify the plotting code to repeat the test of multiple times and stored the mean and standard deviation for each curve in a pandas dataframe. Finally, we use `sns.lineplot` to replot the figure with mean and standard deviation shown.

# +
n_runs = 10
scale_values = [1, 2, 4, 8]

for scale in tqdm.tqdm(scale_values):
    variances = []
    for i in range(n_runs):
        _dW = sample_dW(h/scale, 100, N*scale, 1024)
        empirical_var = WSquareSDE(_dW.shape[-1], gamma=.99, a=0.1).calc_TD(_dW, h/scale).var(dim=-1)
        variances.append(empirical_var.cpu().numpy())
    variances = np.array(variances)
    mean_var = np.mean(variances, axis=0)
    std_var = np.std(variances, axis=0)

    plt.plot(mean_var, label=f"scale={scale}")
    plt.fill_between(range(len(mean_var)), mean_var - std_var, mean_var + std_var, alpha=0.2)

plt.legend()
plt.title("As time step become finer")
plt.xlabel("number of time steps")
plt.ylabel("Variance")
plt.show()


# +
n_runs = 10
scale_values = [1, 2, 4, 8]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, scale in enumerate(scale_values):
    variances = []
    for j in range(n_runs):
        _dW = sample_dW(h, 100, N, 128 * scale)
        empirical_var = WSquareSDE(_dW.shape[-1], gamma=.99, a=0.1).calc_TD(_dW, h).var(dim=-1)
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

# ## A try on higher order integrators

# We may use high order quadrature rules when approximating the time integral. For example, if we adopt the "trapezoidal rule", then
# $$ \mathit{TD}(\theta; h, M; r,j):= X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}\frac{b_i + b_{i+1}}{2}h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr).$$
# Or the "Simpson's rule":
# $$ \mathit{TD}(\theta; h, M; r,j):= X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-2}\frac{b_i + 4b_{i+1} + b_{i+2}}{3}h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr).$$

# ## Conclusion of Choice I

# In conclusion, we have studied the convergence behavior of a residue loss for a specific stochastic differential equation (SDE) with respect to the time step size and the number of Monte Carlo samples. 
#
# We have explored two approaches for computing the solution to the SDE, and we have focused on the first approach, which involves fitting the process directly via neural networks. We have analyzed the error of the resulting numerical scheme, given the true solution for a specific SDE. In particular, we have provided an analytic expression for the temporal difference error at a single time point and sample event, and we have shown that **it converges to zero as the time step size goes to zero. Interestingly, the sample size seems not affect the accuracy (in the mean case) but do affect the stability of returned results.**
#
# Finally, we have tested the proposed scheme on a generalized version of the specific SDE, and we have observed that the numerical scheme's error behaves as expected.

# ## Sample dW with Specific Constraints

# **Problem.** How to sample these $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$?
#
# Note that $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ contains $nNM$ free variables, whereas the condition
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI$$
# consists of $nN+nN(nN+1)/2$ constraints.
#
# Hence, there should be $M \geq nN/2+3/2$.

# Because n normal r.v. can be viewed as a n-dimensional normal r.v., the above problem is equivalent to generate $M$ samples of $nN$ i.i.d. real valued normal variables $\{\epsilon_u(\omega_r)\}_{\substack{0\leq u\leq nN-1 \\ 0\leq r\leq M-1}}$ such that
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_u(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r\epsilon_u(\omega_r)\epsilon_v(\omega_r)=h\delta_{uv}.$$

def corrected_randn(m, n):
    r'''Generate m samples of n normal r.v.s such that the returned matrix r satisfies
    r.mean(dim=0)=0 and r.T.cov()=torch.eye(n)'''
    r = torch.randn(m, n, dtype=TENSORDTYPE, device=DEVICE)
    r -= r.mean(dim=0)
    cov = r.T @ r / m
#     L = torch.linalg.cholesky(cov.inverse())
#     return (L @ r.T).T
    eigvals, eigvecs = torch.linalg.eigh(cov)
    D_half_inv = torch.diag(1 / torch.sqrt(eigvals))
    return r @ eigvecs @ D_half_inv


# +
# for directly sampled noise
dW1 = torch.randn(M, n*N, dtype=TENSORDTYPE, device=DEVICE)
print(dW1.mean(dim=0).abs().max())
print(((dW1.T @ dW1 / M) - torch.eye(n*N, dtype=TENSORDTYPE, device=DEVICE)).abs().max())

# for corrected noise
dW2 = corrected_randn(M, n*N)
print(dW2.mean(dim=0).abs().max())
print(((dW2.T @ dW2 / M) - torch.eye(n*N, dtype=TENSORDTYPE, device=DEVICE)).abs().max())

# -

dW = dW1.view(N, M, n)*np.sqrt(h)
empirical_var = WSquareSDE(dW.shape[-1], gamma=1.99, a=0.1).calc_TD(dW, h).var(dim=-1)
plt.plot(empirical_var.cpu().numpy())

dW = dW2.view(N, M, n)*np.sqrt(h)
empirical_var = WSquareSDE(dW.shape[-1], gamma=1.99, a=0.1).calc_TD(dW, h).var(dim=-1)
plt.plot(empirical_var.cpu().numpy())
