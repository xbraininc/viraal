import logging
import ometrics
import torch
import functools
import numpy as np
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

def prepare_tensors(func):
    def wrapper(*args, **kwargs):
        for t in args:
            if isinstance(t, torch.Tensor):
                t.detach().cpu()
        for t in kwargs.values():
            if isinstance(t, torch.Tensor):
                t.detach().cpu()
        func(*args, **kwargs)
    return wrapper

class Average:
    def __init__(self, name):
        self.reset()
        self.name=name

    def reset(self):
        self.sum=0
        self.nb_samples=0
        self.ave=None
    
    def update(self, **kwargs):
        if self.name in kwargs:
            value = kwargs[self.name]
            self.nb_samples += np.prod(value.shape)
            self.sum += value.sum()
            self.ave = self.sum/self.nb_samples

    def get(self):
        return self.ave

    def __repr__(self):
        return f'{self.name}: {self.get()}'
    
class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.nb_samples = 0
        self.nb_correct = 0
        self.acc = None

    def update(self, logits=None, label=None, **kwargs):
        if logits is not None and label is not None:
            self.nb_samples += logits.size(0)
            pred = logits.argmax(dim=-1)
            self.nb_correct += (pred == label).sum()
            self.acc = float(self.nb_correct)/float(self.nb_samples)
    def get(self):
        return self.acc

    def __repr__(self):
        return f'acc: {self.get()}'

class Metrics:
    def __init__(self):
        self._metrics = ometrics.Metrics({
            'train': [Accuracy(), Average('ce_loss'), Average('vat_loss')],
            'val' : [Accuracy()],
            'test' : [Accuracy()]
        })
    @prepare_tensors
    def update(self, phase, **kwargs):
        for metric in self._metrics[phase]:
            metric.update(**kwargs)
    
    def reset(self):
        for phase in self._metrics:
            for metric in self._metrics[phase]:
                metric.reset()

    def upload(self):
        pass

    def log(self, phases=None):
        phases = phases or self._metrics.keys()
        for phase in phases:
            if any([metric.get() is not None for metric in self._metrics[phase]]):
                message = phase.upper()+'    '+' - '.join([f'{metric}' for metric in self._metrics[phase] if metric.get() is not None])
                logger.info(message)