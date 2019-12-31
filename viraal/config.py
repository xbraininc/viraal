from omegaconf import OmegaConf
from omegaconf import DictConfig
import functools
import uuid
import random
import numpy as np
import torch

def get_key(cfg, key):
    if key == '':
        return cfg
    else:
        keys = key.split('.')
        if keys[0] in cfg:
            return get_key(getattr(cfg, keys[0]), '.'.join(keys[1:]))
        else:
            return False

def merge_kwargs(kwargs1, kwargs2):
    k1 = kwargs1 if isinstance(kwargs1, DictConfig) else OmegaConf.create(kwargs1)
    k2 = kwargs2 if isinstance(kwargs2, DictConfig) else OmegaConf.create(kwargs2)
    merged = OmegaConf.merge(k1,k2)
    return merged.to_container(resolve=True)

def pass_conf(f, cfg, key):
    item = get_key(cfg, key)
    if item:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **merge_kwargs(item, kwargs))
        return wrapper
    else:
        return f

def call_if(condition):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if callable(condition) and condition():
                return func(*args,**kwargs)
            elif condition:
                return func(*args,**kwargs)
        return wrapper
    return decorator

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def register_interpolations():
    OmegaConf.register_resolver("seed", lambda : random.randint(0,1e6))
    OmegaConf.register_resolver("id", lambda : uuid.uuid4().hex)