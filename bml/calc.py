import tqdm
import torch
import numpy as np

from .config import TENSORDTYPE, DEVICE
from .utils import re_cumsum, sample_dW, corrected_sample_dW


class ResCalcMixin(object):
    
    @property
    def h(self):
        return self.T / self.N
    
    @property
    def t(self):
        return self.h * torch.arange(
            1 + self.N, dtype=TENSORDTYPE, device=DEVICE
        ).view(-1,1,1).expand(-1, self.M, 1)

    def get_weight_mu(self, dirac):
        r'''return a weight vector w.r.t. time dimension.

        - if self.dirac is True, only the first weight is 1 and all others are zero.
        - if self.dirac is False, return the uniform weight.
        - if self.dirac is a real number, return the decay weight with rate exp(-self.dirac).
        '''
        if dirac is True:
            weight = torch.cat((torch.tensor([1.0]), torch.zeros(self.N-1)), dim=0)
        elif dirac is False:
            weight = torch.ones(self.N) / float(self.N)
        else:
            gamma = float(dirac)
            weight = (1-np.exp(-gamma))/(1-np.exp(-gamma*self.N))*torch.exp(
                -gamma*torch.arange(self.N))
        return weight.view(-1,1,1).expand(-1, self.M, 1).to(
            dtype=TENSORDTYPE, device=DEVICE)

    def calc_Res(self, mc, dirac):
        r'''Average MC error to obain Res.'''
        return torch.sum(mc * mc * self.get_weight_mu(dirac)) / mc.shape[1]

    def calc_MC(self, t, X, Y, Z, dW, *, rule='rectangle'):
        r'''Evaluate the MC error of given processes.'''
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

    def calc_XYZ(self, v, u, dW):
        r'''Calculate $(X, Y, Z)$ through Choice III'''
        t = self.t
        X = torch.empty(1+self.N, dW.shape[1], self.n, 
                        dtype=TENSORDTYPE, device=DEVICE)
        Y = torch.empty(1+self.N, dW.shape[1], self.m, 
                        dtype=TENSORDTYPE, device=DEVICE)
        Z = torch.empty(1+self.N, dW.shape[1], self.m, self.d,
                        dtype=TENSORDTYPE, device=DEVICE)

        X[0] = self.x0
        Y[0] = v(t[0], X[0])
        Z[0] = u(t[0], X[0])
        for i in range(self.N):
            with torch.no_grad():
                X[i+1] = X[i] + self.h * self.b(t[i], X[i], Y[i], Z[i]) + (
                    self.sigma(t[i], X[i], Y[i], Z[i]) @ dW[i].unsqueeze(-1)
                ).squeeze(-1)
            Y[i+1] = v(t[i+1], X[i+1])
            Z[i+1] = u(t[i+1], X[i+1])
        
        return t, X, Y, Z, dW


class ResSolver(object):

    def __init__(self, sde, model, *, lr, dirac, quad_rule, correction):
        super().__init__()
        self.sde = sde
        self.model = model
        self.lr = lr
        self.dirac = dirac
        self.quad_rule = quad_rule
        self.correction = correction

    @property
    def sample_dW(self):
        if self.correction is True:
            return corrected_sample_dW
        else:
            return sample_dW

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def calc_loss(self):
        data = self.sample_dW(self.sde.h, self.sde.d, self.sde.N, self.sde.M,
                              dtype=TENSORDTYPE, device=DEVICE)
        data = self.sde.calc_XYZ(self.model.ynet, self.model.znet, data)
        data = self.sde.calc_MC(*data, rule=self.quad_rule)
        loss = self.sde.calc_Res(data, dirac=self.dirac)
        return loss

    def calc_metric_y0(self):
        t0 = self.sde.t[0:1, 0:1, 0:1]
        x0 = self.sde.x0.view(1, 1, -1)
        # compare their first element 
        pred_y0 = self.model.ynet(t0, x0).flatten()[0].item()
        true_y0 = self.sde.true_y0.flatten()[0].item()
        return {
            'Y0': pred_y0,
            'Err Y0': abs(pred_y0/true_y0 -1.) * 100,
            'True Y0': true_y0,
        }

    def calc_metric_averXYZ_and_dist(self):
        r'''Compare the difference between predict (X, Y, Z) and true (X, Y, Z)
        then average along the time dimension using sde.dirac'''
        data = self.sample_dW(self.sde.h, self.sde.n, self.sde.N, self.sde.M, 
                                dtype=TENSORDTYPE, device=DEVICE)
        t, pred_X, pred_Y, pred_Z, dW = self.sde.calc_XYZ(
            self.model.ynet, self.model.znet, data,
        )
        t, true_X, true_Y, true_Z, dW = self.sde.calc_XYZ(
            self.sde.true_v, self.sde.true_u, data,
        )
        
        # flat mtraix Z to vector
        pred_Z = pred_Z.reshape(1 + self.sde.N, self.sde.M, -1)
        true_Z = pred_Z.reshape(1 + self.sde.N, self.sde.M, -1)

        weight = self.sde.get_weight_mu(self.dirac)

        averX = torch.sum((pred_X - true_X).square()[:-1]
            * weight, dim=[0, 2]).mean().item()
        averY = torch.sum((pred_Y - true_Y).square()[:-1]
            * weight, dim=[0, 2]).mean().item()
        averZ = torch.sum((pred_Z - true_Z).square()[:-1]
            * weight, dim=[0, 2]).mean().item()

        dist = torch.sum(
            weight * ((pred_Y - true_Y).square()[:-1]
                + re_cumsum((pred_Z - true_Z).square()[:-1]*self.sde.h, dim=0)),
            dim=[0, 2]).mean().item()

        return {
            'averX': averX,
            'averY': averY,
            'averZ': averZ,
            'dist': dist,
        }

    def solve(self, max_steps, pbar=None):
        if pbar is None:
            pbar = tqdm.trange(max_steps)

        tab_logs = []    # summary metrics at the end of the training
        fig_logs = []    # running metrics during training

        optimizer = self.configure_optimizers()
        self.model.train()
        optimizer.zero_grad()

        for step in range(max_steps):
            loss = self.calc_loss()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.model.eval()
            with torch.no_grad():
                metric_y0 = self.calc_metric_y0()

            fig_logs.append({
                'step': step,
                'loss': loss.item(),
                **metric_y0,
            })

        self.model.eval()
        with torch.no_grad():
            metric_y0 = self.calc_metric_y0()
            # specific validation loss
            val_loss = metric_y0['Err Y0']

        tab_logs.append({
            'loss': fig_logs[-1]['loss'],
            'val loss': val_loss,
            **metric_y0,
        })

        return tab_logs, fig_logs

