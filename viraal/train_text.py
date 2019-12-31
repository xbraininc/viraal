import glob
import logging
import os

import hydra
import ometrics
import torch
import numpy as np
from tqdm import tqdm
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from viraal.datasets.imdb import ImdbDatasetReader
from viraal.config import register_interpolations, set_seeds, get_key

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    if not os.path.isdir(directory): os.makedirs(directory)

def get_checkpoint(checkpoint_prefix, iteration):
    return {
        'model' : os.path.join(checkpoint_prefix, f'model_{iteration}.th'),
        'vocab' : os.path.join(checkpoint_prefix, 'model.vocab')
    }

def batch_to_device(batch,device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch_to_device(value, device)

def from_locals(variables, loc):
    kwargs = {}
    for v in variables:
        if v in loc:
            kwargs[v] = loc[v]
    return kwargs

def apply(dico, func):
    new_dico = {}
    for k,v in dico.items():
        new_dico[k] = func(v)
    return new_dico

class TrainText:
    def __init__(self, config):

        config = config if isinstance(config, DictConfig) else OmegaConf.create(config)

        self.unlabeler = instantiate(config.training.unlabeler) if get_key(config, 'training.unlabeler') else None
        self.dataset_reader = instantiate(config.dataset.reader)
        self.iterator = instantiate(config.training.iterator)

        logger.info("Reading train instances")
        self.train_instances = self.dataset_reader.read(config.dataset.splits.train)

        if self.unlabeler is not None: 
            logger.info("Unlabeling train instances")
            self.unlabeler(self.train_instances)
        
        if 'vat' not in config.losses:
            self.train_instances = list(filter(lambda instance: instance.fields['labeled'].metadata, self.train_instances))
        
        logger.info("Reading val instances")
        self.val_instances = self.dataset_reader.read(config.dataset.splits.val)

        self.vocab = Vocabulary.from_instances(self.train_instances)
        self.iterator.index_with(self.vocab)

        embedding_params = Params(OmegaConf.to_container(config.dataset.embedding))
        token_embedding = Embedding.from_params(self.vocab, embedding_params)
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        self.word_embeddings.to(config.misc.device)
        self.model = instantiate(config.model, input_size=self.word_embeddings.get_output_dim(), vocab_size=self.vocab.get_vocab_size('label'))
        self.model.to(config.misc.device)
        self.optimizer = instantiate(config.training.optimizer, self.model.parameters())
        
        self.losses = {}
        for loss in config.losses:
            self.losses[loss] = instantiate(config.losses[loss])
            self.losses[loss].to(config.misc.device)

        self.metrics = instantiate(config.training.metrics)

        self.iteration = 0
        self.saved_iterations = []
        self.config = config
        self.epochs = config.training.epochs
        self.checkpoint_max = config.training.checkpoint_max
        self.checkpoint_freq = config.training.checkpoint_freq
        self.device = config.misc.device

    def train_epoch(self):
        
        self.model.train()
        logger.info("Epoch %s", self.iteration)
        for batch in tqdm(self.iterator(self.train_instances, num_epochs=1, shuffle=True)):
            batch_to_device(batch, self.device)
            embeddings = self.word_embeddings(batch['sentence'])
            mask = get_text_field_mask(batch['sentence'])
            logits = self.model(embeddings=embeddings, mask=mask)
            label = batch['label']
            labeled = np.array(batch['labeled'], dtype=bool)

            if 'ce' in self.losses and any(labeled):
                ce_loss = self.losses['ce'](logits[labeled], label[labeled])
                ce_loss.mean().backward(retain_graph=True)
                self.metrics.update('labeled_train', ce_loss=ce_loss)
                
            if 'vat' in self.losses:
                model_forward = lambda embeddings : self.model(embeddings=embeddings, mask=mask)
                vat_loss = self.losses['vat'](logits, model_forward, embeddings, mask)
                vat_loss.mean().backward()
                self.metrics.update('train', vat_loss=vat_loss)

            tensors = from_locals(['logits', 'mask', 'label', 'vat_loss'],loc = locals())

            if any(labeled):
                labeled_tensors = apply(tensors, lambda x:x[labeled])
                self.metrics.update('labeled_train',**labeled_tensors)

            if any(~labeled):
                unlabeled_tensors = apply(tensors, lambda x:x[~labeled])
                self.metrics.update('unlabeled_train',**unlabeled_tensors)

            self.metrics.update('train', prop_labeled=sum(labeled)/len(labeled), **tensors)

            self.optimizer.step()
        
        self.iteration += 1
        self.metrics.log()
        # self.metrics.tensorboard()
        self.metrics.upload()
        self.metrics.reset()

    def ensure_max_checkpoints(self, new_checkpoint):
        self.saved_iterations.append(new_checkpoint)
        while len(self.saved_iterations) > self.checkpoint_max:
            os.remove(self.saved_iterations.pop(0))

    def save_best(self, checkpoint_prefix="model"):
        torch.save(self.model, os.path.join(checkpoint_prefix, 'best.th'))

    def save(self, checkpoint_prefix="model"):
        checkpoint = get_checkpoint(checkpoint_prefix, self.iteration)
        ensure_dir(checkpoint_prefix)
        self.ensure_max_checkpoints(checkpoint['model'])

        torch.save(self.model, checkpoint['model'])
        if not os.path.isdir(checkpoint['vocab']): 
            self.vocab.save_to_files(checkpoint['vocab'])
        
        return checkpoint

    def restore(self, checkpoint):
        self.model = torch.load(checkpoint['model'])
        self.vocab.from_files(checkpoint['vocab'])

    def train_loop(self):
        for epoch in range(self.epochs):
            try:
                self.train_epoch()
                if self.iteration % self.checkpoint_freq==0:
                    self.save()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.save()
                raise e
        self.save()

@hydra.main(config_path='config/train_text.yaml', strict=False)
def train_text(cfg):
    register_interpolations()
    cfg_yaml = cfg.pretty(resolve=True)
    logger.info("====CONFIG====\n%s", cfg_yaml)
    with open('conf.yaml', 'w+') as conf:
        conf.write(cfg_yaml)

    set_seeds(cfg.misc.seed)

    tr = TrainText(cfg)

    tr.train_loop()

if __name__ == "__main__":
    train_text()