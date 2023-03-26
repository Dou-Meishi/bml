# # Network

import torch

# ## Simple FC Nets

# +
class YNet_FC2L(torch.nn.Module):
    
    def __init__(self, n, m, *, hidden_size):
        super().__init__()
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m, bias=True)
        )
        
    def forward(self, t, x):
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z)
    
    
class ZNet_FC2L(torch.nn.Module):
    
    def __init__(self, n, m, d, *, hidden_size):
        super().__init__()
        self.m = m
        self.d = d
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m*d, bias=True)
        )
        
    def forward(self, t, x):
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z).view(*x.shape[:-1], self.m, self.d)


# +
class YNet_FC3L(torch.nn.Module):
    
    def __init__(self, n, m, *, hidden_size):
        super().__init__()
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m, bias=True)
        )
        
    def forward(self, t, x):
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z)
    
    
class ZNet_FC3L(torch.nn.Module):
    
    def __init__(self, n, m, d, *, hidden_size):
        super().__init__()
        self.m = m
        self.d = d
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m*d, bias=True),
        )
        

    def forward(self, t, x):
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z).view(*x.shape[:-1], self.m, self.d)
# -

# ## FC Nets with Fixed $y_0$

class YNet_FC2L_Fixt0(torch.nn.Module):
    
    def __init__(self, n, m, *, hidden_size):
        super().__init__()
        
        self.register_parameter(
            'y0',
            torch.nn.Parameter(torch.rand(m).uniform_(1, 2))
        )

        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m, bias=True)
        )
        
    def forward(self, t, x):
        chi = (abs(t)>1e-15).float().to(device=x.device, dtype=x.dtype)
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z)*chi + self.y0


class YNet_FC3L_Fixt0(torch.nn.Module):
    
    def __init__(self, n, m, *, hidden_size):
        super().__init__()
        
        self.register_parameter(
            'y0',
            torch.nn.Parameter(torch.rand(m).uniform_(1, 2))
        )
        
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(1+n, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, m, bias=True)
        )
        
    def forward(self, t, x):
        chi = (abs(t)>1e-15).float().to(device=x.device, dtype=x.dtype)
        z = torch.cat([t, x], dim=-1)
        return self.fcnet(z)*chi + self.y0

