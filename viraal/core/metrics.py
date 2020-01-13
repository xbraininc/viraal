import logging
import ometrics
import torch
import wandb
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

class Entity():
    def __init__(self, beg, end, tag):
        self.beg = beg
        self.end = end
        self.tag = tag


def getEntity(tag_list):
    begin_label = "B-"
    inside_label = "I-"
    current_tag = "O"
    beg_current_entity = -1
    list_entity = {}
    for i in range(0, len(tag_list)):
        tag = tag_list[i]
        if begin_label == tag[:len(begin_label)]:
            if beg_current_entity != -1:
                list_entity[beg_current_entity] = (
                    Entity(beg_current_entity, i-1, current_tag))
            current_tag = tag[len(begin_label):]
            beg_current_entity = i
        if tag == "O":
            if beg_current_entity != -1:
                list_entity[beg_current_entity] = (
                    Entity(beg_current_entity, i-1, current_tag))
            current_tag = None
            beg_current_entity = -1
        if inside_label == tag[:len(inside_label)]:
            if tag[len(inside_label):] != current_tag:
                if beg_current_entity != -1:
                    list_entity[beg_current_entity] = (
                        Entity(beg_current_entity, i-1, current_tag))
                current_tag = tag[len(inside_label):]
                beg_current_entity = i
    return list_entity

def getPrecisionRecall(list_tag_pred, list_tag_real):
    assert len(list_tag_pred) == len(list_tag_real)
    entity_pred = getEntity(list_tag_pred)
    entity_real = getEntity(list_tag_real)
    nb_pred = len(entity_pred)
    nb_real = len(entity_real)
    nb_right = 0
    same_beg = set(entity_real.keys()).intersection(set(entity_pred.keys()))
    for i in same_beg:
        if entity_real[i].end == entity_pred[i].end and entity_real[i].tag == entity_pred[i].tag:
            nb_right += 1
    if nb_pred != 0:
        recall = nb_right/nb_pred
    else:
        recall = 0
    if nb_real == 0:
        return 0, recall
    return nb_right/nb_real, recall

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

    def get_name(self):
        return f'{self.name}_ave'

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

    def get_name(self):
        return f'{self.name}_std'

    def __repr__(self):
        return f'{self.name} std: {self.get():.2e}'
    
class Accuracy:
    def __init__(self, name=None):
        self.reset()
        self.name = name or ''

    def reset(self):
        self.nb_samples = 0
        self.nb_correct = 0
        self.acc = None

    def update(self, name='', logits=None, label=None, **kwargs):
        if logits is not None and label is not None and name==self.name:
            self.nb_samples += logits.size(0)
            pred = logits.argmax(dim=-1)
            self.nb_correct += (pred == label).sum()
            self.acc = float(self.nb_correct)/float(self.nb_samples)
    def get(self):
        return self.acc

    def get_name(self):
        return f'{self.name} acc'

    def __repr__(self):
        return f'{self.get_name()}: {self.get()*100:.2f}%'

class F1Conlleval:
    def __init__(self, name=None):
        self.reset()
        self.name = name or ''

    def reset(self):
        self.pred = []
        self.tags = []

    def update(self, name='', logits=None, tags=None, vocab=None, mask=None, **kwargs):
        if logits is not None and tags is not None and vocab is not None and mask is not None and name==self.name:
            pred = logits.argmax(dim=-1)
            self.pred.extend([vocab.get_token_from_index(idx.item(), namespace='tags') for idx in pred[mask]])
            self.tags.extend([vocab.get_token_from_index(idx.item(), namespace='tags') for idx in tags[mask]])

    def get(self):
        if self.pred and self.tags:
            p,r = getPrecisionRecall(self.pred, self.tags)
            return 2*p*r/(p+r) if p+r > 0 else 0
        else:
            return None

    def get_name(self):
        return f'{self.name} conlleval f1'

    def __repr__(self):
        return f'{self.get_name()}: {self.get()*100:.2f}%'

class Metrics:
    def __init__(self, wandb=True):
        # self.writer = SummaryWriter('tensorboard')
        self.wandb=wandb
        self._metrics = ometrics.Metrics({
            'labeled_train': [Accuracy(), Accuracy('int'), F1Conlleval('tag'), Average('ce_loss'), Average('vat_loss')],
            'unlabeled_train': [Accuracy(), Accuracy('int'), F1Conlleval('tag'), Average('vat_loss')],
            'train': [Accuracy(), Accuracy('int'), F1Conlleval('tag'), Average('vat_loss')],
            'val' : [Accuracy(), Accuracy('int'), F1Conlleval('tag')],
            'test' : [Accuracy(), Accuracy('int'), F1Conlleval('tag')]
        })

    @prepare_tensors
    def update(self, phase, **kwargs):
        for metric in self._metrics[phase]:
            metric.update(**kwargs)
    
    def reset(self):
        for phase in self._metrics:
            for metric in self._metrics[phase]:
                metric.reset()

    def upload(self, step):
        if self.wandb:
            to_upload={'epoch':step}
            for phase in self._metrics:
                for metric in self._metrics[phase]:
                    if metric.get() is not None:
                        full_name='_'.join([phase, metric.get_name()])
                        to_upload[full_name] = metric.get()
            wandb.log(to_upload, step=step)

    def log(self):
        for phase in self._metrics:
            any_metric_to_display = any([metric.get() is not None for metric in self._metrics[phase]])
            if any_metric_to_display:
                representations = [f'{metric}' for metric in self._metrics[phase] if metric.get() is not None]
                message = phase.upper()+'    '+' - '.join(representations)
                logger.info(message)