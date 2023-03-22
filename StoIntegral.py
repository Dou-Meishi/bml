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

# local files
from yznets import *


# -

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
#    We want to design a numerical scheme for computing $\mathit{Res}$ (with aforementioned scheme for computing $X_t$) such that $\mathit{Res}$ would converge to zero. Interestingly, we guess a naive discretization scheme cannot achieve this.

# **Example.** (`WSqureSDE`)  We can choose $x_0=0$ and
# $$ \begin{cases}
# b(t,x) = \gamma x + nae^{\gamma t},\\
# \sigma(t,x) = 2ae^{\gamma t}W_t.
# \end{cases}$$
# Then the true solution is $X_t=ae^{\gamma t}\|W_t\|^2$.
#
# When $a=1/n,~\gamma=0$, this reduces to (denote by $n$ the dimension of $W$)
# $$ \frac{1}{n}\|W_t\|^2 = t + \frac{2}{n}\int_0^t\langle W_s,dW_s\rangle.$$

# ## Choice I with Naive Euler-Maruyama Discretization

# Let us assume that we choose to fit the process $\{X_t,0\leq t\leq T\}$ directly from $\{W_t,0\leq T\}$, i.e., $X_t\approx X^\theta(t,\omega)$. Then we substitue this into the definition of $\mathit{Res}$ and obtain
# $$ \mathit{Res}(\theta):= \operatorname{\mathbb{E}}\int_0^T\biggl|X^\theta(t,\omega) - \Bigl(x_0 + \int_0^tb(s,X^\theta(s,\omega))\,ds + \int_0^t\sigma(s,X^\theta(s,\omega))\,dW_s\Bigr)\biggr|^2\,\mu(dt).$$

# Hence, we need only some appropriate integrator for calculating these integrals. For example, we may apply the simple rectangle rule for the time integral and Euler-Maruyama rule for the stochastic integral:
# $$ \mathit{Res}(\theta; h, M):= \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1}\biggl|X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}b(ih,X^\theta(ih,\omega_r))h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr)\biggr|^2\,\mu(jh).$$

# Now assume that we are lucky enough such that we happen to find a representation which  coincides with the true solution, i.e., $X^{\theta_*}(t,\omega)=ae^{\gamma t}\|W(t,\omega)\|^2$. We want to study the behavior of $\mathit{Res}(\theta_*;h,M)$ as $h\to 0$ and $M\to\infty$.

# For clarity, let us consider the temporal difference error at a single time point and sample event
# $$ \mathit{TD}(\theta; h, M; r,j):= X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}b(ih,X^\theta(ih,\omega_r))h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr).$$
# With this notation, we have 
# $$ \mathit{Res}(\theta;h,M) = \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} \bigl|\mathit{TD}(\theta;h,M;r,j)\bigr|^2\,\mu(jh).$$

# Let us study the statistics of $\mathit{TD}(\theta_*;h,M;r,j)$ with $a=1/n,~\gamma=0$. In this case,
# $$ \mathit{TD}(\theta_*;h,M;r,j) = \frac{1}{n}\|W(jh,\omega_r)\|^2 - jh - \frac{2}{n}\sum_{i=0}^{j-1}\langle  W(ih,\omega_r), W((i+1)h,\omega_r) - W(ih,\omega_r)\rangle .$$

# It is not very hard to show that if $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ satisfies the statistics of $N$ i.i.d. $n$-dimensional standard normal distribution, which containts but not limit to
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

# **Problem.** How to sample these $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$?
#
# Note that $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ contains $nNM$ free variables, whereas the condition
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI$$
# consists of $nN+n^2N^2$ constraints.
#
# Hence, there should be $M \geq nN+1$.

def sample_dW(h, n, N, M):
    r'''Return dW[i,r,*] of M independent sample paths of n dimensional standard Brownian Motion with time step h and final time T=hN.'''
    return torch.randn(N, M, n)*h


# +
# sample M samples of n dim from normal dist with std h
h = 0.01
n = 5
N = 100      # 1.0/0.01
M = 1024

epsilon = sample_dW(h, n, N, M)
# -

# check mean
epsilon.mean(dim=1).abs().max()

# check cov
max([(epsilon[i].T.cov(correction=0)-h*h*torch.eye(n)).abs().max() for i in range(N)])

# check independence
max([(epsilon[i].unsqueeze(-1) @ epsilon[j].unsqueeze(1)).mean(dim=0).abs().max() for i, j in itertools.combinations(range(N), 2)])


def calc_TD_WSquareSDE(dW, h):
    W = torch.cat([torch.zeros_like(dW[0:1]), dW]).cumsum(dim=0)
    return torch.sum(W*W,dim=-1)/W.shape[-1] - h*torch.arange(W.shape[0]).view(-1, 1) - 2/W.shape[-1]*torch.cat([torch.zeros_like(dW[0:1, :, 0]), torch.sum(W[:-1]*dW, dim=-1)]).cumsum(dim=0)


td = calc_TD_WSquareSDE(epsilon, h)

# check mean
td.mean(dim=-1)
