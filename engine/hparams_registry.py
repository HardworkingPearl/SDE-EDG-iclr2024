# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/hparams_registry.py
# The specific hyper-parameters for each algorithm

import numpy as np
from engine.utils import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(cfg):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms.py / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['RMNIST']

    Digit_Data = ['RotatedGaussian','ToyCircle', 'ToySine', 'CoverType', 'PowerSupply']

    random_seed = cfg.seed
    dataset = cfg.data_name
    algorithm = cfg.algorithm

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert (name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Define global parameters
    # Note that domain_num is for test only
    _hparam('source_domains', cfg.source_domains, lambda r: cfg.source_domains)
    _hparam('intermediate_domains', cfg.intermediate_domains, lambda r: cfg.intermediate_domains)
    _hparam('target_domains', cfg.target_domains, lambda r: cfg.target_domains)
    _hparam('num_classes', cfg.num_classes, lambda r: cfg.num_classes)
    _hparam('feature_dim', cfg.feature_dim, lambda r: cfg.feature_dim)
    _hparam('euclidean_metric', cfg.euclidean_metric, lambda r: cfg.euclidean_metric)
    _hparam('data_size', cfg.data_size, lambda r: cfg.data_size)
    _hparam('uni', not cfg.multi, lambda r: not cfg.multi)
    _hparam('interp', cfg.interp, lambda r: cfg.interp)
    _hparam('mlp_depth', cfg.mlp_depth, lambda r: r.choice([2,3,4,5,6,7,8]))
    _hparam('mlp_width', cfg.mlp_width, lambda r: r.choice([32, 64, 128, 512]))
    _hparam('mlp_dropout', cfg.dropout, lambda r: r.choice([0, 0.05, 0.1, 0.2, 0.5]))
    _hparam('weight_decay', cfg.weight_decay, lambda r: r.choice([0, 1e-3,5e-4,5e-5]))

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    elif dataset in Digit_Data:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-5, -3.5)) 
    else:
        _hparam('lr', 1e-4, lambda r: 10 ** r.uniform(-5, -3.5))

    if algorithm in ['VAE', 'DIVA', 'LSSAE']:
        _hparam('zc_dim', cfg.zc_dim, lambda r: cfg.zc_dim)
        _hparam('stochastic', True, lambda r: True)
        # Params for DIVA only
        _hparam('zdy_dim', cfg.zw_dim, lambda r: cfg.zw_dim)
        # Params for LSSAE only
        _hparam('zw_dim', cfg.zw_dim, lambda r: cfg.zw_dim)
        _hparam('zv_dim', cfg.num_classes, lambda r: cfg.num_classes)
        _hparam('coeff_y', 80, lambda r: 80)
        _hparam('coeff_ts', 20, lambda r: 20)
    elif algorithm == "SDE":
        _hparam('path_weight', cfg.path_weight, lambda r: r.choice([100,20, 1, 0.5]))
        _hparam('interp_weight', cfg.interp_weight, lambda r: r.choice([0, 0.1, 0.2, 0.3]))
        _hparam('dt', 0.05, lambda r: r.choice([0.005, 0.01, 0.05, 0.1])) # 0.05
        _hparam('method', "euler", lambda r: r.choice(["euler", "reversible_heun"]))
        _hparam('solver', "ito", lambda r: r.choice(["ito"]))

    return hparams


def get_hparams(cfg):
    return {a: b for a, (b, c) in _hparams(cfg).items()}
