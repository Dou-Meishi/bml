import torch
import numpy as np

from .config import TENSORDTYPE, DEVICE


class FBSDE_LongSin(object):
    
    def __init__(self, n, *, T=None, N=None, M=None, r=0., sigma_0=0.4):
        self.n = n
        self.m = 1
        self.d = self.n
        
        self.T, self.N, self.M = T, N, M

        self.r = r
        self.sigma_0 = sigma_0

        self.x0 = .5*np.pi*torch.ones(self.n).to(device=DEVICE, dtype=TENSORDTYPE)
        
    def b(self, t, x, y, z):
        return 0.*x
    
    def sigma(self, t, x, y, z):
        return self.sigma_0 * y.unsqueeze(-1) * torch.eye(self.d).to(device=DEVICE, dtype=TENSORDTYPE)
    
    def f(self, t, x, y, z):
        return -self.r*y + .5*torch.exp(-3*self.r*(self.T-t))*self.sigma_0**2*(10/self.n*torch.sum(torch.sin(x), dim=-1, keepdim=True))**3
    
    def g(self, x):
        return 10/self.n*torch.sum(torch.sin(x), dim=-1, keepdim=True)
    
    def true_v(self, t, x):
        return 10/self.n*torch.exp(-self.r*(self.T-t))*torch.sum(torch.sin(x), dim=-1, keepdim=True)

    def true_u(self, t, x):
        return self.sigma_0*100/self.n**2*(torch.exp(-2*self.r*(self.T-t))*torch.sum(torch.sin(x), dim=-1, keepdim=True)*torch.cos(x)).unsqueeze(-2)

