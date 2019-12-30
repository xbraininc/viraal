import glob
import logging
import os

import hydra
import ometrics
import torch
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
            
            if 'ce' in self.losses and any(batch['labeled']):
                labeled = batch['labeled']
                ce_loss = self.losses['ce'](logits[labeled], batch['label'][labeled])
                ce_loss.mean().backward(retain_graph=True)
                self.metrics.update('train', ce_loss=ce_loss)
            
            if 'vat' in self.losses:
                model_forward = lambda embeddings : self.model(embeddings=embeddings, mask=mask)
                vat_loss = self.losses['vat'](logits, model_forward, embeddings, mask)
                vat_loss.mean().backward()
                self.metrics.update('train', vat_loss=vat_loss)

            self.metrics.update('train',
                                logits=logits, 
                                mask=mask, 
                                label=batch['label'], 
                                labeled=batch['labeled'])

            self.optimizer.step()
        
        self.iteration += 1
        self.metrics.log()
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
            self.train_epoch()
            if self.iteration % self.checkpoint_freq==0:
                self.save()
        self.save()

@hydra.main(config_path='config/train_text.yaml', strict=False)
def train_text(cfg):
    register_interpolations()
    logger.info("====CONFIG====\n%s", cfg.pretty(resolve=True))

    set_seeds(cfg.misc.seed)

    tr = TrainText(cfg)

    tr.train_loop()

if __name__ == "__main__":
    train_text()