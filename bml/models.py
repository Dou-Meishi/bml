import torch
import numpy as np

from .yznets import *


class YZNet_FC2L(torch.nn.Module):

    def __init__(self, n, m, d, *, hidden_size):
        super().__init__()

        self.ynet = YNet_FC2L(n, m, hidden_size=hidden_size)
        self.znet = ZNet_FC2L(n, m, d, hidden_size=hidden_size)



class YZNet_FC3L(torch.nn.Module):

    def __init__(self, n, m, d, *, hidden_size):
        super().__init__()

        self.ynet = YNet_FC3L(n, m, hidden_size=hidden_size)
        self.znet = ZNet_FC3L(n, m, d, hidden_size=hidden_size)
