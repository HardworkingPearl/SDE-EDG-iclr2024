import pdb
import torch
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import copy

from .submodules import *
from .cla_func import *
from .loss_func import *
from engine.utils import one_hot, random_pairs_of_minibatches, ParamDict
from engine.configs import Algorithms

@Algorithms.register('erm')
class ERM(torch.nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model_func, cla_func, hparams):
        super(ERM, self).__init__()
        self.featurizer = model_func
        self.classifier = cla_func
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None, return_z=False):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        T = len(minibatches)
        n = len(minibatches[0][0])

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        preds = self.classifier(all_z)
        loss = self.criterion(preds, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss,preds, all_y

    def predict(self, x, domain_idx, *args, **kwargs):
        return self.classifier(self.featurizer(x))
    


@Algorithms.register('sde')
class EDGSDE(torch.nn.Module):
    import torchsde
    torchsde = torchsde
    def __init__(self, model_func, cla_func, hparams): #input_shape, num_classes, num_domains, hparams):
        super(EDGSDE, self).__init__()
        hparams["clip"] = 0.05
        self.hparams = hparams
        self.model_func = model_func 
        self.cla_func = cla_func
        self.T = hparams['source_domains']
        self.l = hparams['intermediate_domains'] + hparams['target_domains']
        self.feature_dim = self.model_func.n_outputs
        self.data_size = hparams['data_size']
        self.num_classes = hparams['num_classes']
        self.euclidean_metric = hparams['euclidean_metric']
        self.path_weight = hparams['path_weight']
        self.interp_weight = hparams['interp_weight']
        self.dm_idx = 0
        self.criterion = nn.CrossEntropyLoss()
        self.path_regression_loss = torch.nn.MSELoss()
        self.n_batch = 16
        self.init_std = 0.05 # sine plot RMNIST
        self.invariant = True  # D(y|t) is invariant
        self.uni = hparams["uni"] # Uni-modal / Multi-modal 
        self.interp = hparams["interp"]
        self.path_yhat = None

        self._build()
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.init_mu)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def _build(self):
        self.init_mu = nn.Parameter(torch.empty((self.num_classes, 1, self.feature_dim)),requires_grad=True)
        self.sde = nn.ModuleList([SDE(self.feature_dim, self.feature_dim, self.hparams) for _ in range(self.num_classes)])
        
        self.optimizer = torch.optim.Adam(
            [{'params':self.model_func.parameters()},
             {'params':self.init_mu},
            {'params':self.sde.parameters(), 'weight_decay':self.hparams['weight_decay']}],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']*0.1
        )

    def sde_cal(self, k, z0, ts):
        return self.torchsde.sdeint(self.sde[k], z0, ts,  method=self.hparams['method'], dt=self.hparams['dt'])

    def cal_dist(self, x, y, use_euclidean=False):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return self.metric_dist(x, y, use_euclidean)

    
    def metric_dist(self, x, y, use_euclidean=False):
        if self.euclidean_metric or use_euclidean:
            return - torch.pow(x - y, 2).sum(2)  # [n, m]
        else:
            # dot product as -distance
            return torch.sum(x * y,dim=-1)


    def cal_loss(self, proto, z, y):
        yhat =  self.cal_dist(z, proto)  # [batchsize, n_class]
        loss = self.criterion(yhat, y)   # [batchsize, n_class]
        return loss, yhat
    
    def cal_loss_multi(self, path, z, y):
        eps=1e-7
        sim =  self.cal_dist(z, path)
        sim = sim*(sim < 10)
        same_class_mask = y.unsqueeze(1).expand(self.B, self.num_classes * self.n_batch)==self.path_yhat.unsqueeze(0).expand(self.B, self.num_classes * self.n_batch) 
        pos_cls = torch.sum( torch.exp(sim) * same_class_mask, -1)
        all_cls = torch.sum( torch.exp(sim), -1)
        loss = -torch.log(pos_cls+eps) + torch.log(all_cls+eps)
        yhat = torch.sum( torch.exp(sim).view(self.B, self.num_classes, self.n_batch), -1) 
        return loss.mean(), yhat

    def update(self, minibatches, unlabeled=None):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        if self.path_yhat == None:
            self.path_yhat = torch.arange(self.num_classes).unsqueeze(1).repeat(1, self.n_batch).view(-1).to(self.device)
        T = len(minibatches)
        self.B = len(minibatches[0][0])

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.model_func(all_x)
        ts = torch.linspace(0, 1, self.T+self.l+1)

        all_z = all_z.view(T, self.B, -1)
        all_y = all_y.view(T, self.B)
        loss = 0
        yhats = []
        sde_init = self.init_mu +self.init_std * torch.randn(self.num_classes, self.n_batch, self.feature_dim).to(self.device)
        if not self.interp:
            self.sde_path = torch.stack([self.sde_cal(i, sde_init[i], ts) for i in range(self.num_classes)],dim=1)[1:]
            self.proto = self.sde_path.mean(dim=2) 

            sde_path = self.sde_path
            proto = self.proto
            path_yhat = self.path_yhat
        else: 
            interp_ratio = np.random.beta(0.2,0.2)  
            interp_ratio = interp_ratio if (interp_ratio >= 0.001 and interp_ratio <= 0.999) else 0.001
            step_interp = 1 / (self.T + self.l) * interp_ratio
            ts_interp = torch.linspace(step_interp, self.T/(self.T+self.l)+step_interp, self.T+1)[:-1]
            ts_interp = ts_interp[1:]
            ts_interp_idx = []
            for i in range(len(ts_interp)):
                ts = torch.cat((ts[:2*(i+1)],ts_interp[[i]],ts[2*(i+1):]))
                ts_interp_idx.append(2*(i+1))
            ts_idx = []
            for i in range(len(ts)):
                if i not in ts_interp_idx and i != 0:
                    ts_idx.append(i)

            sde_whole_path = torch.stack([self.sde_cal(i, sde_init[i], ts) for i in range(self.num_classes)],dim=1)
            self.sde_path = sde_whole_path[ts_idx]
            self.proto = self.sde_path.mean(dim=2) 

            sde_path = self.sde_path
            proto = self.proto
            path_yhat = self.path_yhat
            sde_path_interp = sde_whole_path[ts_interp_idx]
            proto_interp = sde_path_interp.mean(dim=2)


        for i in range(len(minibatches)):
            # find the 1-NN sample and cal_dist
            zhat = sde_path[i].flatten(0,1)
            track_sim = self.cal_dist(zhat, all_z[i], True)  # - distance (similarity between domain i, i+1)
            not_same_class_mask = (path_yhat.unsqueeze(1).expand(self.num_classes * self.n_batch,self.B) != all_y[i].unsqueeze(0).expand(self.num_classes * self.n_batch,self.B)) * (-1e6)
            track_sim = track_sim + not_same_class_mask
            _, track_pair = torch.max(track_sim, dim=-1)
            sde_path_loss = self.path_regression_loss(zhat, all_z[i][track_pair])
            if self.uni:
                lss, yhat = self.cal_loss(proto[i], all_z[i],  all_y[i])
            else:
                lss, yhat = self.cal_loss_multi(zhat, all_z[i], all_y[i])
            loss += lss
            loss += sde_path_loss * self.path_weight
            yhats.append(yhat)

            ###### interpolation ##############
            if self.interp and i < len(minibatches) - 1:
                track_sim = self.cal_dist(all_z[i], all_z[i+1], True)
                not_same_class_mask = (all_y[i].unsqueeze(1).expand(self.B,self.B) != all_y[i+1].unsqueeze(0).expand(self.B,self.B)) * (-1e6)
                track_sim = track_sim + not_same_class_mask
                _, track_pair = torch.max(track_sim, dim=-1)
                zinterp = (1 - interp_ratio) * all_z[i] + interp_ratio * all_z[i+1][track_pair] # correspondence! 

                zhat = sde_path_interp[i].flatten(0,1)
                lss, _ = self.cal_loss(proto_interp[i], zinterp,  all_y[i])  # test it out?
                loss += self.interp_weight * lss

        loss/= T
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.hparams["clip"])
        self.optimizer.step()
        return loss, torch.cat(yhats), all_y.view(-1)

    def predict(self, x, domain_idx=None,*args, **kwargs):
        if not hasattr(self, 'proto') or self.dm_idx > domain_idx:
            self.device = "cuda" if x.is_cuda else "cpu"
            sde_init = self.init_mu +self.init_std * torch.randn(self.num_classes, self.n_batch, self.feature_dim).to(self.device)
            ts = torch.linspace(0, 1, self.T+self.l+1)
            if self.uni:
                self.proto = torch.stack([self.sde_cal(i, sde_init[i], ts) for i in range(self.num_classes)],dim=1)[1:].mean(dim=2)
            else: 
                self.sde_path = torch.stack([self.sde_cal(i, sde_init[i], ts) for i in range(self.num_classes)],dim=1)[1:]
        self.dm_idx = domain_idx
        z = self.model_func(x)
        B = len(x)
        if self.uni:
            yhat = self.cal_dist(z, self.proto[domain_idx])
        else:
            path = self.sde_path[domain_idx].flatten(0,1)
            sim = self.cal_dist(z, path)
            yhat = torch.sum( torch.exp(sim).view(B, self.num_classes, self.n_batch), -1)  # / all_cls
        return yhat
