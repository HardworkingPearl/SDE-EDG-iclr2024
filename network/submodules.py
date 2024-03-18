import pdb
from abc import abstractmethod
import copy

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent

import network.init_func as init


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        
        self.num_layers = hparams['mlp_depth']
        if self.num_layers > 1:
            self.input = nn.Linear(n_inputs, hparams['mlp_width'])
            self.dropout = nn.Dropout(hparams['mlp_dropout'])
            self.hiddens = nn.ModuleList([
                nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
                for _ in range(hparams['mlp_depth']-2)])
            self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        else:
            self.input = nn.Linear(n_inputs, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        if self.num_layers > 1:
            x = self.dropout(x)
            x = F.relu(x)
            for hidden in self.hiddens:
                x = hidden(x)
                x = self.dropout(x)
                x = F.relu(x)
            x = self.output(x)
        return x
    

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    def __init__(self, n_inputs, n_outputs, hparams):
        super().__init__()
        self.sde_type = hparams["solver"]  # 'ito':"euler","milstein","srk" 'stratonovich':"midpoint","milstein","reversible_heun"
        self.brownian_size = n_outputs # hparams["brownian_size"] # n_outputs // 2 if n_outputs > 16 else n_outputs  # 8

        self.mu1 = MLP(n_inputs, n_outputs, hparams)
        self.mu2 = MLP(n_inputs, n_outputs, hparams)

        self.mu3 = MLP(n_outputs, n_outputs, hparams)
        self.mu4 = MLP(n_outputs, n_outputs, hparams)

        self.sigma1 = MLP(n_inputs, n_outputs, hparams)
        self.sigma2 = MLP(n_inputs, n_outputs, hparams)
        self.state_size = n_inputs

    # Drift
    def f(self, t, x):
        self.device = "cuda" if x.is_cuda else "cpu"
        t = t.expand(x.size(0), x.size(1)).to(self.device)
        x = self.mu1(x) + self.mu2(t)
        return x

    # Diffusion
    def g(self, t, x):
        self.device = "cuda" if x.is_cuda else "cpu"
        t = t.expand(x.size(0), x.size(1)).to(self.device)
        x = self.sigma1(x) + self.sigma2(t)
        return x
