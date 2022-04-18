import random
import subprocess
import numpy as np
from omegaconf import DictConfig

def flatten_dictConf(cfg, prefix=""):
    flat_cfg = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            sub_dict = flatten_dictConf(v, prefix=k+".")
            flat_cfg.update(sub_dict)
        else:
            flat_cfg[prefix+k] = v
    return flat_cfg

def get_git_label():
    try:
        git_label = str(subprocess.check_output(["git", 'describe', '--always'])).strip()[2:-3]
    except:
        git_label = ''
    return git_label


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.random.set_seed(myseed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)
