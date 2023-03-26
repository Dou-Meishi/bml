import math
import time
import warnings

import torch


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


def sample_dW(h, n, N, M, *args, **kws):
    r'''Return dW[i,r,*] of M independent sample paths of
    n dimensional standard Brownian Motion with time step h 
    and final time T=hN.'''
    return torch.randn(N, M, n, *args, **kws)*math.sqrt(h)


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


def generate_orth_vecs(m, n, *args, **kws):
    r'''return n orthogonal $\mathbb{R}^m$ vectors as
    [e1, e2, ..., en] and ensure ei is orthogonal to
    the all-1 vector in $\mathbb{R}^m$.

    Require m >= n+1
    '''
    assert m >= n + 1, "Require m >= n + 1"
    if n > 10**3:
        warnings.warn(f"You're trying to manipulating very large matrix {m} by {n}")

    A = torch.randn(m, n+1, *args, **kws)
    A[:, 0] = 1.
    return torch.linalg.qr(A)[0][:, 1:]


def corrected_sample_dW(h, n, N, M, *args, **kws):
    return generate_orth_vecs(
        M, n*N, *args, **kws
    ).view(M, N, n).transpose(0, 1)*math.sqrt(h*M)
