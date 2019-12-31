import logging
import ometrics
import torch
import functools
import numpy as np
from hydra.utils import instantiate
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def prepare_tensors(func):
    def wrapper(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        for t in args:
            if isinstance(t, torch.Tensor):
                new_args.append(t.detach().cpu())
            else:
                new_args.append(t)
        for name, t in kwargs.items():
            if isinstance(t, torch.Tensor):
                new_kwargs[name] = t.detach().cpu()
            else:
                new_kwargs[name] = t
        func(*new_args, **new_kwargs)
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
        return f'{self.name}: {self.get():.2e}'

class Std:
    def __init__(self, name):
        self.reset()
        self.name=name

    def reset(self):
        self.values=[]

    def update(self, **kwargs):
        if self.name in kwargs:
            value = kwargs[self.name]
            self.values.append(value)

    def get(self):
        return np.std(self.values)

    def __repr__(self):
        return f'{self.name} std: {self.get():.2e}'
    
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
        return f'acc: {self.get()*100:.2f}%'

class Histogram:
    def __init__(self, name, writer):
        self.name=name
        self.writer=writer
        self.reset()

    def reset(self):
        self.values=[]

    def update(self, **kwargs):
        if self.name in kwargs:
            value = kwargs[self.name]
            self.values.append(value)

    def get(self):
        return None

    def tensorboard(self, iteration, phase):
        self.writer.add_histogram(phase+self.name, self.values, iteration)

    def __repr__(self):
        return f'{self.name}: {self.get():.2e}'
    

class Metrics:
    def __init__(self):
        self.writer = SummaryWriter('tensorboard')
        self._metrics = ometrics.Metrics({
            'labeled_train': [Accuracy(), Average('ce_loss'), Average('vat_loss')],
            'unlabeled_train': [Accuracy(), Average('vat_loss')],
            'train': [Accuracy(), Average('vat_loss'), Average('prop_labeled'), Std('prop_labeled')],
            'val' : [Accuracy()],
            'test' : [Accuracy()]
        })
        self.iteration=0
    @prepare_tensors
    def update(self, phase, **kwargs):
        for metric in self._metrics[phase]:
            metric.update(**kwargs)
    
    def reset(self):
        self.iteration+=1
        for phase in self._metrics:
            for metric in self._metrics[phase]:
                metric.reset()

    def upload(self):
        pass

    def tensorboard(self):
        for phase in self._metrics:
            for metric in self._metrics:
                if hasattr(metric, 'tensorboard'):
                    metric.tensorboard(phase, self.iteration)

    def log(self):
        for phase in self._metrics:
            any_metric_to_display = any([metric.get() is not None for metric in self._metrics[phase]])
            if any_metric_to_display:
                representations = [f'{metric}' for metric in self._metrics[phase] if metric.get() is not None]
                message = phase.upper()+'    '+' - '.join(representations)
                logger.info(message)