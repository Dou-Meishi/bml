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
# 1. How to represent the solution $X_t$.
#
#    1. *Choice I.* Directly represent via neural networks (`Seq2Seq` model).
#    
#    2. *Choice II.* Compute by difference equation starting at $X_0=x_0$.
#    
# 2. Suppose the true solution $X_t=X(t,\omega)$ is known. 
#
#    We want to design a numerical scheme for computing $\mathit{Res}$ (with aforementioned scheme for computing $X_t$) such that $\mathit{Res}$ would converge to zero. Note that if these integrals could be obtained without discretization error then this $\mathit{Res}$ is strictly equal to zero.

# # Loss of True Solution with Choice I

# Let us assume that we choose to fit the process $\{X_t,0\leq t\leq T\}$ directly from $\{W_t,0\leq T\}$, i.e., $X_t\approx X^\theta(t,\omega)$. Then we substitue this into the definition of $\mathit{Res}$ and obtain
# $$ \mathit{Res}(\theta):= \operatorname{\mathbb{E}}\int_0^T\biggl|X^\theta(t,\omega) - \Bigl(x_0 + \int_0^tb(s,X^\theta(s,\omega))\,ds + \int_0^t\sigma(s,X^\theta(s,\omega))\,dW_s\Bigr)\biggr|^2\,\mu(dt).$$

# Hence, we need only some appropriate integrator for calculating these integrals. For example, we may apply the simple rectangle rule for the time integral and Euler-Maruyama rule for the stochastic integral:
# $$ \mathit{Res}(\theta; h, M):= \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1}\biggl|X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}b(ih,X^\theta(ih,\omega_r))h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr)\biggr|^2\,\mu(jh).$$

# For clarity, let us consider the temporal difference error at a single time point and sample event
# $$ \mathit{MC}(\theta; h, M; r,j):= X^\theta(jh,\omega_r) - \Bigl(x_0 + \sum_{i=0}^{j-1}b(ih,X^\theta(ih,\omega_r))h + \sum_{i=0}^{j-1}\sigma(ih,X^\theta(ih,\omega_r))\,(W_{(i+1)h}-W_{ih})\Bigr).$$
# With this notation, we have 
# $$ \mathit{Res}(\theta;h,M) = \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} \bigl|\mathit{MC}(\theta;h,M;r,j)\bigr|^2\,\mu(jh).$$

# Now assume that we are lucky enough such that we happen to find a representation which  coincides with the true solution, i.e., $X^{\theta_*}(t,\omega)=ae^{\gamma t}\|W(t,\omega)\|^2$. We want to study the behavior of $\mathit{Res}(\theta_*;h,M)$ as $h\to 0$ and $M\to\infty$. Note that if these integrals can be obtained without discretization errors, then $\mathit{Res}(\theta_*)$ is strictly zero. Thus, all we want to do is to verify the limit equation
# $$ \lim_{\substack{h\to0\\M\to\infty}}\mathit{Res}(\theta_*;h,M) = \mathit{Res}(\theta_*).$$

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
# Let us study the statistics of $\mathit{MC}(\theta_*;h,M;r,j)$ with $a=1/n,~\gamma=0$. In this case,
# $$ \mathit{MC}(\theta_*;h,M;r,j) = \frac{1}{n}\|W(jh,\omega_r)\|^2 - jh - \frac{2}{n}\sum_{i=0}^{j-1}\langle  W(ih,\omega_r), W((i+1)h,\omega_r) - W(ih,\omega_r)\rangle .$$

# **Analytic Solution**. It is not very hard to show that if $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ satisfies the statistics of $N$ i.i.d. $n$-dimensional standard normal distribution, which containts but not limit to
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI,$$
# where $\epsilon_i(\omega_r):=W((i+1)h,\omega_r) - W(ih,\omega_r)$, then
# $$ \overline{\operatorname{\mathbb{E}}}_r \mathit{MC}(\theta_*;h,M;r,j) = 0, \quad \overline{\operatorname{\mathbb{E}}}_r |\mathit{MC}(\theta_*;h,M;r,j)|^2 = 2jh^2/n.$$
#
# *Proof.* Rewrite $\mathit{MC}$ with $\{\epsilon_i\}_{i=0}^{N-1}$,
# $$\mathit{MC}(\theta_*;h,M;r,j) = \frac{1}{n}\Bigl\|\sum_{i=0}^{j-1}\epsilon_i\Bigr\|^2 - jh - \frac{2}{n}\sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle.$$
# With the given hypothesess, it can be shown that
# $$ \overline{\operatorname{\mathbb{E}}}_r \Bigl\|\sum_{i=0}^{j-1}\epsilon_i\Bigr\|^2 = jnh,\quad \overline{\operatorname{\mathbb{E}}}_r \Bigl\|\sum_{i=0}^{j-1}\epsilon_i\Bigr\|^4 = (3jn-2n+2)jnh^2.$$
# $$ \overline{\operatorname{\mathbb{E}}}_r \sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle = 0,\quad \overline{\operatorname{\mathbb{E}}}_r \Bigl|\sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle\Bigr|^2 = \frac{1}{2}j(j-1)(nh)^2.$$
# $$ \overline{\operatorname{\mathbb{E}}}_r \left[\Bigl\|\sum_{i'=0}^{j-1}\epsilon_{i'}\Bigr\|^2 \sum_{i=0}^{j-1}\sum_{k=0}^{i-1}\langle \epsilon_k,\epsilon_i\rangle \right] = j(j-1)(nh)^2.$$
# Desired properties hold when these equations hold.
#
# *Q.E.D.*

# **Remark.** Based on these analytic formulae, we can make some insightful predictions as follows.
#
# 1. As $h\to0$ (even if $N$ increased such that the total time $T=hN$ remains constant), ***the discrete residue loss $\mathit{Res}(\theta_*;h,M)$ indeed goes to zero in the same rate of*** $h$. This is because the $\overline{\operatorname{\mathbb{E}}}_r |\mathit{MC}(\theta_*;h,M;r,j)|^2 \propto jh^2$.
#    
#    1. If $\mu$ is set to Dirac measure at the final time, then
#       $$ \mathit{Res}(\theta_*;h,M) = \overline{\operatorname{\mathbb{E}}}_r |\mathit{MC}(\theta_*;h,M;r,N)|^2 \propto Th.$$
#       
#    2. If $\mu$ is set to simple $dt$ then
#       $$ \mathit{Res}(\theta_*;h,M) = \overline{\operatorname{\mathbb{E}}}_r h \sum_{j=0}^{N-1} |\mathit{MC}(\theta_*;h,M;r,j)|^2 \propto T^2h.$$
#       
#    3. If $\mu$ is set to decaying measure then (below the $\gamma\neq 0$)
#       $$ \mathit{Res}(\theta_*;h,M) = C(\gamma, h) \overline{\operatorname{\mathbb{E}}}_r h \sum_{j=0}^{N-1} e^{\gamma hj} |\mathit{MC}(\theta_*;h,M;r,j)|^2 \propto e^{\gamma T}h.$$
#       
# 2. The ***sample size $M$ does not directly affect the residue loss but through conditions like*** $\overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0, ~\overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI$. With large sample size $M$, these conditions are more likely to hold and then so the previous formulae.

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
    _dW = sample_dW(h/scale, n, N*scale, M)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1
                              ).calc_TD(_dW, h/scale).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"scale={scale}")

    plt.legend()

plt.title("As time step become finer")

# +
for scale in [1, 2, 4, 8]:
    _dW = sample_dW(h, n, N, 128*scale)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1
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
        _dW = sample_dW(h/scale, n, N*scale, M)
        empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1).calc_TD(_dW, h/scale).var(dim=-1)
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
        _dW = sample_dW(h, n, N, 128 * scale)
        empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1).calc_TD(_dW, h).var(dim=-1)
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

# ## Conclusion of Choice I

# In conclusion, we have studied the convergence behavior of a residue loss of the true solution with respect to the time step size and the number of Monte Carlo samples. 
#
# We have explored two approaches for representing the true solution, and we have focused on the first approach, which involves computing the true solution directly from the given sample of the Brownian motion. 
#
# We have analyzed the error of the resulting numerical scheme, given the true solution for a specific SDE. In particular, we have provided an analytic expression for the temporal difference error at a single time point and sample event, and we have shown that **it converges to zero as the time step size goes to zero. Interestingly, the sample size seems not affect the accuracy (in the mean case) but do affect the stability of returned results.**
#
# Finally, we have tested the proposed scheme on a generalized version of the specific SDE, and we have observed that the numerical scheme's error behaves as expected.

# ## A try on higher order integrators

# We may use high order quadrature rules when approximating the time integral. More specifically, we want to approximate the time integral $\int_0^tb(s,X^\theta_s)\,ds$ with data 
# $$b_i^\theta(\omega_r) := b(ih, X^\theta(ih,\omega_r)), \quad i=0, 1, 2, \ldots, j$$.
#
# For the *rectangle rule*, we have
# $$ \int_0^tb(s,X^\theta_s)\,ds \approx \sum_{i=0}^{j-1}b_i^\theta\,h.$$
# For the *trapezoidal rule*, we have
# $$ \int_0^tb(s,X^\theta_s)\,ds \approx \sum_{i=0}^{j-1}\frac{b_i^\theta+b^\theta_{i+1}}{2}\,h.$$
# For the *Simpson's rule*, we have (assume $j$ is even)
# $$ \begin{aligned}
# \int_0^tb(s,X^\theta_s)\,ds &\approx \sum_{m=0}^{j/2-1}\frac{h}{3}\,\bigl(b_{2m}+bf_{2m+1}+b_{2m+2}\bigr) \\
# &= \frac{h}{3}\biggl(\sum_{i=0}^{j-1}b_j + \sum_{i=1}^{j}b_i + 2\sum_{m=0}^{j/2-1}b_{2m+1}\biggr)
# \end{aligned}$$

# +
def quadrature(b, h, rule):
    def shift_cumsum(t, offset=1):
        return torch.cat([torch.zeros_like(t[:offset]), t.cumsum(dim=0)])

    if rule == 'rectangle':
        return h * shift_cumsum(b[:-1], offset=1)
    elif rule == 'trapezoidal':
        return h * shift_cumsum((b[:-1] + b[1:]) / 2, offset=1)
    else:
        raise ValueError(f"Unrecognized rule {rule}")


def calc_TD_with_quadrature(self, dW, h, rule='rectangle'):
    def shift_cumsum(t):
        return torch.cat([torch.zeros_like(t[:1]), t.cumsum(dim=0)])

    weight = torch.exp(self.gamma*h*torch.arange(
        dW.shape[0]+1, device=DEVICE)).view(-1, 1, 1)
    W = shift_cumsum(dW)
    X = torch.sum(W*W, dim=-1, keepdim=True)*self.a*weight

    b = self.n*self.a*weight + self.gamma*X
    timeint = quadrature(b, h, rule)
    stoint = shift_cumsum(torch.sum(
        weight[:-1]*W[:-1]*dW, dim=-1, keepdim=True)
                         )*self.a*2
    return (X - timeint - stoint).squeeze(-1)

WSquareSDE.calc_TD_with_quadrature = calc_TD_with_quadrature

# +
for rule in ['rectangle', 'trapezoidal']:
    _dW = sample_dW(h, n, N, M)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1
                              ).calc_TD_with_quadrature(_dW, h, rule=rule).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"rule={rule}")

    plt.legend()

plt.title("when different integrators are applied")


# -

# Hence, the error mainly comes from the discretization error of stochastic integrals, and ***increase the accuracy of the approximation to time integrals cannot significantly improves the performance***.
#
# **Hypothesis**. If we could force these conditions to be satisfied then we may use small batch size to achieve the comparable performance of large batch size. 

# ## Sample dW with Specific Constraints

# **Problem.** How to sample these $\{\epsilon_i(\omega_r)\}_{\substack{0\leq i\leq N-1\\ 0\leq r \leq M-1}}$ such that
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_i(\omega_r)=0_{n},\quad \overline{\operatorname{\mathbb{E}}}_r[\epsilon_i(\omega_r)][\epsilon_k(\omega_r)]^\intercal=\delta_{ik}hI_{n},$$
# where $0_n$ is the zero vector in $\mathbb{R}^n$ and $I_n$ is the $n\times n$ identity matrix.

# Because n normal r.v. can be viewed as a n-dimensional normal r.v., the above problem is equivalent to generate $M$ samples of $nN$ i.i.d. real valued normal variables $\{\epsilon_u(\omega_r)\}_{\substack{0\leq u\leq nN-1 \\ 0\leq r\leq M-1}}$ such that
# $$ \overline{\operatorname{\mathbb{E}}}_r  \epsilon_u(\omega_r)=0,\quad \overline{\operatorname{\mathbb{E}}}_r\epsilon_u(\omega_r)\epsilon_v(\omega_r)=h\delta_{uv}.$$
# This is again equivalent to forcing $\{\epsilon_u, 0\leq u\leq nN\}$ to be an orthogonal vector basis of $\mathbb{R}^M$, where we manually add the $\epsilon_{nN} = \mathbb{1}_m$, which is the all-$1$ vector. **Hence, we should have $M \geq nN+1$ to achieve this.**
#
# *TODO.* Maybe we could analytically find a $(nN+1)\times nN$ matrix satisfying the above conditions.

def force_orthogonal(A):
    r'''For m by n matrix $A=(a_1,a_2,\ldots,a_n)$, return another matrix
    $B=(b_1,b_2,\ldots,b_n)$ such that $\langle 1, b_i\rangle = 0$ and
    $B^T B = I_{n}$.
    
    Require $A^T A$ is full rank and A does not contain all-1 vector.'''
    # ensure $\langle 1, b_i \rangle = 0$
    B = A - A.mean(dim=0)
    # Transform B such that B^T B = I
    cov = B.T @ B / B.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    if (eigvals.abs() < 1e-10).any():
        raise ValueError(f"Input matrix in shape {A.shape} is not full rank")
    D_half_inv = torch.diag(1/ torch.sqrt(eigvals))
    return B @ eigvecs @ D_half_inv


# +
# for directly sampled noise
noise1 = torch.randn(M, n*N, dtype=TENSORDTYPE, device=DEVICE)
print(noise1.mean(dim=0).abs().max())
print(((noise1.T @ noise1 / M) - torch.eye(
    n*N, dtype=TENSORDTYPE, device=DEVICE)).abs().max())

# for corrected noise
noise2 = force_orthogonal(noise1)
print(noise2.mean(dim=0).abs().max())
print(((noise2.T @ noise2 / M) - torch.eye(
    n*N, dtype=TENSORDTYPE, device=DEVICE)).abs().max())


# -

def corrected_sample_dW(h, n, N, M):
    return force_orthogonal(
        torch.randn(M, n*N, dtype=TENSORDTYPE, device=DEVICE)
    ).view(M, N, n).transpose(0, 1)*np.sqrt(h)


# +
# Create empty arrays to store the results
sampled_var = []
corrected_var = []

# Repeat the procedure 10 times for each value of gamma
for gamma in [1.99, -1.99]:
    sampled_var_gamma = []
    corrected_var_gamma = []
    for i in tqdm.trange(10):
        # Sample dW and calculate empirical variance for the sampled dW
        dW = sample_dW(h, n, N, M)
        empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_TD(dW, h).var(dim=-1)
        sampled_var_gamma.append(empirical_var.cpu().numpy())
        
        # Sample corrected dW and calculate empirical variance for the corrected dW
        dW = corrected_sample_dW(h, n, N, M)
        empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_TD(dW, h).var(dim=-1)
        corrected_var_gamma.append(empirical_var.cpu().numpy())
    
    # Append the results for this value of gamma to the overall arrays
    sampled_var.append(sampled_var_gamma)
    corrected_var.append(corrected_var_gamma)

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=1)
sampled_std = np.std(sampled_var, axis=1)
corrected_mean = np.mean(corrected_var, axis=1)
corrected_std = np.std(corrected_var, axis=1)

# Create the figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, ax in enumerate(axs):
    gamma = [1.99, -1.99][i]
    sampled_line = ax.plot(sampled_mean[i], label='sampled')
    ax.fill_between(range(len(sampled_mean[i])), sampled_mean[i] - sampled_std[i], sampled_mean[i] + sampled_std[i], alpha=0.2)
    corrected_line = ax.plot(corrected_mean[i], label='corrected')
    ax.fill_between(range(len(corrected_mean[i])), corrected_mean[i] - corrected_std[i], corrected_mean[i] + corrected_std[i], alpha=0.2)
    ax.set_title(f'Gamma = {gamma}')
    ax.legend()

plt.show()

# +
scales = [1, 2, 4, 8]
gamma_values = [1.99, -1.99]

# Create empty arrays to store the results
sampled_var = np.zeros((len(scales), len(gamma_values), 10, N+1))
corrected_var = np.zeros((len(scales), len(gamma_values), 10, N+1))

# Repeat the procedure for different values of M and gamma
for scale_idx, scale in enumerate(scales):
    for gamma_idx, gamma in enumerate(gamma_values):
        for i in tqdm.trange(10):
            # Sample dW and calculate empirical variance for the sampled dW
            dW = sample_dW(h, n, N, M*scale)
            empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_TD(dW, h).var(dim=-1)
            sampled_var[scale_idx, gamma_idx, i] = empirical_var.cpu().numpy()
            
            # Sample corrected dW and calculate empirical variance for the corrected dW
            dW = corrected_sample_dW(h, n, N, M*scale)
            empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_TD(dW, h).var(dim=-1)
            corrected_var[scale_idx, gamma_idx, i] = empirical_var.cpu().numpy()

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=2)
sampled_std = np.std(sampled_var, axis=2)
corrected_mean = np.mean(corrected_var, axis=2)
corrected_std = np.std(corrected_var, axis=2)

# Create the figure with subplots
fig, axs = plt.subplots(len(scales), len(gamma_values), figsize=(8, 12))

for (scale_idx, gamma_idx), ax in np.ndenumerate(axs):
    gamma = gamma_values[gamma_idx]
    scale = scales[scale_idx]
    sampled_line = ax.plot(sampled_mean[scale_idx, gamma_idx], label='sampled')
    ax.fill_between(range(len(sampled_mean[scale_idx, gamma_idx])), sampled_mean[scale_idx, gamma_idx] - sampled_std[scale_idx, gamma_idx], sampled_mean[scale_idx, gamma_idx] + sampled_std[scale_idx, gamma_idx], alpha=0.2)
    corrected_line = ax.plot(corrected_mean[scale_idx, gamma_idx], label='corrected')
    ax.fill_between(range(len(corrected_mean[scale_idx, gamma_idx])), corrected_mean[scale_idx, gamma_idx] - corrected_std[scale_idx, gamma_idx], corrected_mean[scale_idx, gamma_idx] + corrected_std[scale_idx, gamma_idx], alpha=0.2)
    ax.set_title(f'Gamma = {gamma}, scale = {scale}')
    ax.legend()

plt.tight_layout()
plt.show()


# -

# *Discuss.* The above figure shows two phenomena:
#
# 1. As the sample size increases, the difference between the "sampled" and "corrected" methods becomes smaller.
#
# 2. The "corrected" method produces smaller errors than the naive "sampled" method.
#
# The first phenomenon is expected because when the sample size is very large, the dW sampled directly from normal distributions are regular enough to satisfy the orthogonal conditions, rendering manual correction unnecessary.
#
# It is interesting to note, however, that the "corrected" method produces smaller errors than the naive "sampled" method, even when the sample size is very small. In fact, the "corrected" method outperforms the naive method by a factor of 8, despite having only a fraction of the sample size.
#
# Overall, the figure highlights the importance of the "corrected" method in producing more accurate results, particularly with smaller sample sizes.
#
# *TODO.* Try different nonlinear SDEs

# # Loss of True Solution with Choice II

# Let's say that we choose to compute $X$ from the difference equation (for the sake of clarity, denote by $\hat{X}$ the processed produced this way)
# $$\hat{X}((i+1)h,\omega_r) =\hat{X}(ih, \omega_r) + b(ih, \hat{X}(ih, \omega_r))\,h + \sigma(ih, \hat{X}(ih, \omega_r))\,[W((i+1)h,\omega_r) - W(ih, \omega_r)].$$
#
# Note that for this process, its discrete residue loss is automatically zero. This is because for arbitary chosen $h$ and $M$, we have
# $$ \mathit{MC}(\hat{X};h,M;r,j)\equiv 0,\quad \forall r, j$$
# by definition. In other words, $\hat{X}$ is the solution to the following optimization problem
# $$ \min_{X^\theta}\quad \mathit{Res}(X^\theta; h, M).$$

# We want to study the error (or say distance) between $\hat{X}$ and the true solution $X$. Introduce
# $$ \operatorname{dist}_\mu(X^\theta,X):= \operatorname{\mathbb{E}} \int_0^T |X^\theta_t - X_t|^2\,\mu(dt),$$
# $$ \operatorname{dist}_\mu(\theta^*; h, M) := \overline{\operatorname{\mathbb{E}}}_r \sum_{j=0}^{N-1} |\hat{X}(jh, \omega_r) - X(jh, \omega_r)|^2\,\mu(jh)$$
# and 
# $$ \mathit{VE}(\theta^*; h, M; r, j) := \hat{X}(jh, \omega_r) - X(jh, \omega_r).$$

# +
def calc_VE_WSquareSDE(self, dW, h):
    def shift_cumsum(t):
        return torch.cat([torch.zeros_like(t[:1]), t.cumsum(dim=0)])

    weight = torch.exp(self.gamma*h*torch.arange(
        dW.shape[0]+1, device=DEVICE)).view(-1, 1, 1)
    W = shift_cumsum(dW)
    X = torch.zeros(W.shape[0], W.shape[1], 1, dtype=TENSORDTYPE, device=DEVICE)

    X[0] = 0.
    for i in range(dW.shape[0]):
        X[i+1] = X[i] + (
            self.gamma*X[i] + self.n*self.a*weight[i]
        ) * h + 2*weight[i]*self.a*torch.sum(W[i]*dW[i], dim=-1, keepdim=True)
        
    return (X - torch.sum(W*W, dim=-1, keepdim=True)*self.a*weight).squeeze(-1)

WSquareSDE.calc_VE = calc_VE_WSquareSDE

# +
for scale in [1, 2, 4, 8]:
    _dW = sample_dW(h/scale, n, N*scale, M)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1
                              ).calc_VE(_dW, h/scale).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"scale={scale}")

    plt.legend()

plt.title("As time step become finer")

# +
for scale in [1, 2, 4, 8]:
    _dW = sample_dW(h, n, N, 128*scale)
    empirical_var = WSquareSDE(_dW.shape[-1], gamma=1.99, a=0.1
                              ).calc_VE(_dW, h).var(dim=-1)
    plt.plot(empirical_var.cpu().numpy(), label=f"scale={scale}")

    plt.legend()

plt.title("As sample size increase")

# +
# Create empty arrays to store the results
sampled_var = []
corrected_var = []

# Repeat the procedure 10 times for each value of gamma
for gamma in [1.99, -1.99]:
    sampled_var_gamma = []
    corrected_var_gamma = []
    for i in tqdm.trange(10):
        # Sample dW and calculate empirical variance for the sampled dW
        dW = sample_dW(h, n, N, M)
        empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_VE(dW, h).var(dim=-1)
        sampled_var_gamma.append(empirical_var.cpu().numpy())
        
        # Sample corrected dW and calculate empirical variance for the corrected dW
        dW = corrected_sample_dW(h, n, N, M)
        empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_VE(dW, h).var(dim=-1)
        corrected_var_gamma.append(empirical_var.cpu().numpy())
    
    # Append the results for this value of gamma to the overall arrays
    sampled_var.append(sampled_var_gamma)
    corrected_var.append(corrected_var_gamma)

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=1)
sampled_std = np.std(sampled_var, axis=1)
corrected_mean = np.mean(corrected_var, axis=1)
corrected_std = np.std(corrected_var, axis=1)

# Create the figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, ax in enumerate(axs):
    gamma = [1.99, -1.99][i]
    sampled_line = ax.plot(sampled_mean[i], label='sampled')
    ax.fill_between(range(len(sampled_mean[i])), sampled_mean[i] - sampled_std[i], sampled_mean[i] + sampled_std[i], alpha=0.2)
    corrected_line = ax.plot(corrected_mean[i], label='corrected')
    ax.fill_between(range(len(corrected_mean[i])), corrected_mean[i] - corrected_std[i], corrected_mean[i] + corrected_std[i], alpha=0.2)
    ax.set_title(f'Gamma = {gamma}')
    ax.legend()

plt.show()

# +
scales = [1, 2, 4, 8]
gamma_values = [1.99, -1.99]

# Create empty arrays to store the results
sampled_var = np.zeros((len(scales), len(gamma_values), 10, N+1))
corrected_var = np.zeros((len(scales), len(gamma_values), 10, N+1))

# Repeat the procedure for different values of M and gamma
for scale_idx, scale in enumerate(scales):
    for gamma_idx, gamma in enumerate(gamma_values):
        for i in tqdm.trange(10):
            # Sample dW and calculate empirical variance for the sampled dW
            dW = sample_dW(h, n, N, M*scale)
            empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_VE(dW, h).var(dim=-1)
            sampled_var[scale_idx, gamma_idx, i] = empirical_var.cpu().numpy()
            
            # Sample corrected dW and calculate empirical variance for the corrected dW
            dW = corrected_sample_dW(h, n, N, M*scale)
            empirical_var = WSquareSDE(dW.shape[-1], gamma=gamma, a=0.1).calc_VE(dW, h).var(dim=-1)
            corrected_var[scale_idx, gamma_idx, i] = empirical_var.cpu().numpy()

# Calculate the mean and standard deviation of each line
sampled_mean = np.mean(sampled_var, axis=2)
sampled_std = np.std(sampled_var, axis=2)
corrected_mean = np.mean(corrected_var, axis=2)
corrected_std = np.std(corrected_var, axis=2)

# Create the figure with subplots
fig, axs = plt.subplots(len(scales), len(gamma_values), figsize=(8, 12))

for (scale_idx, gamma_idx), ax in np.ndenumerate(axs):
    gamma = gamma_values[gamma_idx]
    scale = scales[scale_idx]
    sampled_line = ax.plot(sampled_mean[scale_idx, gamma_idx], label='sampled')
    ax.fill_between(range(len(sampled_mean[scale_idx, gamma_idx])), sampled_mean[scale_idx, gamma_idx] - sampled_std[scale_idx, gamma_idx], sampled_mean[scale_idx, gamma_idx] + sampled_std[scale_idx, gamma_idx], alpha=0.2)
    corrected_line = ax.plot(corrected_mean[scale_idx, gamma_idx], label='corrected')
    ax.fill_between(range(len(corrected_mean[scale_idx, gamma_idx])), corrected_mean[scale_idx, gamma_idx] - corrected_std[scale_idx, gamma_idx], corrected_mean[scale_idx, gamma_idx] + corrected_std[scale_idx, gamma_idx], alpha=0.2)
    ax.set_title(f'Gamma = {gamma}, scale = {scale}')
    ax.legend()

plt.tight_layout()
plt.show()
# -

# # Conclusion of the Residue Loss

# In this work, we investigated the residual loss of a stochastic process with respect to the general type of SDE. Although it is easily observed from the definition that this residual loss is zero if the considered process is the solution to the SDE, this property becomes more nuanced when applied to practical numerical methods. The issue arises because the original definition requires the calculation of several integrals, which involves approximations throughout the computation. We examined the discrete version of this residual loss and have summarized our conclusions as follows:
#
# 1. The discrete residual loss of the true solution to the SDE is not zero. However, as the time step used for discretization approaches zero, the loss also approaches zero. In our toy example, this convergence rate is linear.
#
# 2. The process that reduces the discrete residual loss to zero is the process generated by the simple Euler-Maruyama scheme. Although this process deviates from the true solution, the difference will vanish if the time step approaches zero.
#
# 3. In the tests mentioned above, both of these errors (i.e., the discrete loss of the true solution and the difference between the Euler-Maruyama process and the true solution) can be estimated using sample paths of the Brownian motion via Monte Carlo methods. As anticipated, larger sample sizes result in smaller variances, but the mean value of these errors is not affected. However, if we replace these sample paths with specifically generated samples, a significant decrease in errors is indeed observed. The reason behind this occurrence remains unaddressed in this study and presents a promising avenue for future research.
