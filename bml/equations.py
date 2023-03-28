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

        self.x0 = .5*np.pi*torch.ones(self.n, dtype=TENSORDTYPE, device=DEVICE)
        
    def b(self, t, x, y, z):
        return 0.*x
    
    def sigma(self, t, x, y, z):
        return self.sigma_0 * y.unsqueeze(-1) * torch.eye(
            self.d, dtype=TENSORDTYPE, device=DEVICE)
    
    def f(self, t, x, y, z):
        return -self.r*y + .5*torch.exp(-3*self.r*(self.T-t))*self.sigma_0**2*(10/self.n*torch.sum(torch.sin(x), dim=-1, keepdim=True))**3
    
    def g(self, x):
        return 10/self.n*torch.sum(torch.sin(x), dim=-1, keepdim=True)
    
    def true_v(self, t, x):
        return 10/self.n*torch.exp(-self.r*(self.T-t))*torch.sum(torch.sin(x), dim=-1, keepdim=True)

    def true_u(self, t, x):
        return self.sigma_0*100/self.n**2*(torch.exp(-2*self.r*(self.T-t))*torch.sum(torch.sin(x), dim=-1, keepdim=True)*torch.cos(x)).unsqueeze(-2)

    @property
    def true_y0(self):
        t0 = torch.zeros(1, dtype=TENSORDTYPE, device=DEVICE)
        x0 = self.x0
        return self.true_v(t0, x0)

class FBSDE_FuSinCos(object):
    
    def __init__(self, n=1, *, T=None, N=None, M=None):
        assert n==1, "FuSinCos allow only n=1"

        self.n = 1
        self.m = 1
        self.d = 1

        self.T, self.N, self.M = T, N, M

        self.x0 = torch.ones(self.n, device=DEVICE, dtype=TENSORDTYPE)
        
    def b(self, t, x, y, z):
        return -.25*torch.sin(2*(t+x))*(y*y+z.squeeze(-1))

    def sigma(self, t, x, y, z):
        return (.5*torch.cos(t+x)*(y*torch.sin(t+x)+z.squeeze(-1)+1)).unsqueeze(-1)
    
    def f(self, t, x, y, z):
        return y*z.squeeze(-1)-torch.cos(t+x)

    def g(self, x):
        return torch.sin(self.T+x)
    
    def true_v(self, t, x):
        return torch.sin(t+x)
    
    def true_u(self, t, x):
        return torch.cos(t+x).unsqueeze(-1)

    @property
    def true_y0(self):
        t0 = torch.zeros(1, dtype=TENSORDTYPE, device=DEVICE)
        x0 = self.x0
        return self.true_v(t0, x0)


class FBSDE_JiLQ5(object):
    
    def __init__(self, n=5, *, T=None, N=None, M=None):
        self.n = n
        self.m = self.n
        self.d = 1

        self.T, self.N, self.M = T, N, M
        
        self.x0 = 1. * torch.ones(self.n, device=DEVICE, dtype=TENSORDTYPE)
        
    def b(self, t, x, y, z):
        return -.25*x + .5*y + .5*z.squeeze(-1)
    
    def sigma(self, t, x, y, z):
        return (.2*x + .5*y + .5*z.squeeze(-1)).unsqueeze(-1)
    
    def f(self, t, x, y, z):
        return -.5*x - .25*y + .2*z.squeeze(-1)
    
    def g(self, x):
        return -x

    @property
    def true_y0(self):
        return -0.9568 * self.x0

    def calc_cost(self, t, x, u):
        x2 = torch.sum(x*x, dim=-1, keepdim=True)
        u2 = torch.sum(u*u, dim=-1, keepdim=True)
        cost = .5*x2[-1:] + torch.sum(.25*x2[:-1] + u2[:-1], dim=0, keepdim=True)*self.h
        return cost