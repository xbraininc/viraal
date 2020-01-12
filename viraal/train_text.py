import logging
import os
import time

import hydra
import numpy as np
import torch
import wandb
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from viraal.config import (flatten_dict, get_key, pass_conf,
                           register_interpolations, save_config, set_seeds)
from viraal.core.utils import (apply, batch_to_device, destroy_trainer, ensure_dir,
                          from_locals, instances_info)

logger = logging.getLogger(__name__)


class TrainText:
    def __init__(self, config, checkpoint_prefix='model'):

        """
        This function relies heavily on hydra.utils.instantiate which fetches 
        the object directly as specified by the config. Once can also put additional
        argments with it, they will be merged with what the config specified.
        """
        
        self.checkpoint_prefix=checkpoint_prefix
        self.c = config if isinstance(config, DictConfig) else OmegaConf.create(config)
        
        self.instantiate_dataset()
        self.instantiate_model()
        self.instantiate_losses()

        self.metrics = instantiate(config.training.metrics, wandb=self.c.misc.wandb)

        if config.misc.wandb: wandb.watch(self.model)

        self.iteration = 0
        self.saved_iterations = []

    def instantiate_losses(self):
        self.losses = {}
        for loss in self.c.losses:
            self.losses[loss] = instantiate(self.c.losses[loss])
            self.losses[loss].to(self.c.misc.device)
        
    def instantiate_dataset(self):
        self.unlabeler = (
            instantiate(self.c.training.unlabeler)
            if get_key(self.c, "training.unlabeler")
            else None
        )
        self.dataset_reader = instantiate(self.c.dataset.reader)
        self.iterator = instantiate(self.c.training.iterator)

        logger.info("Reading train instances")
        self.train_instances = self.dataset_reader.read(self.c.dataset.splits.train)

        if self.unlabeler is not None:
            logger.info("Unlabeling train instances")
            self.unlabeler(self.train_instances)

        logger.info("Reading val instances")
        self.val_instances = self.dataset_reader.read(self.c.dataset.splits.val)

        configured_vocab_from_inst = pass_conf(Vocabulary.from_instances, self.c, "dataset.vocab")
        self.vocab = configured_vocab_from_inst(self.train_instances)
        self.iterator.index_with(self.vocab)

    def instantiate_model(self):
        embedding_params = Params(OmegaConf.to_container(self.c.dataset.embedding))
        token_embedding = Embedding.from_params(self.vocab, embedding_params)
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        self.word_embeddings.to(self.c.misc.device)
        self.model = instantiate(
            self.c.model,
            input_size=self.word_embeddings.get_output_dim(),
            vocab_size=self.vocab.get_vocab_size("labels"),
        )
        self.model.to(self.c.misc.device)

        self.optimizer = instantiate(
            self.c.training.optimizer,
            list(self.model.parameters()) + list(self.word_embeddings.parameters()),
        )

    def ensure_max_checkpoints(self):
        while len(self.saved_iterations) > self.c.training.checkpoint_max:
            for p in self.saved_iterations.pop(0):
                os.remove(p)

    def save(self, checkpoint_prefix="model"):
        checkpoint = get_checkpoint(checkpoint_prefix, self.iteration)
        ensure_dir(checkpoint_prefix)
        self.saved_iterations.append([checkpoint["embedding"], checkpoint["encoder"]])
        self.ensure_max_checkpoints()

        torch.save(self.model.state_dict(), checkpoint["encoder"])
        torch.save(self.word_embeddings.state_dict(), checkpoint["embedding"])
        if not os.path.isdir(checkpoint["vocab"]):
            self.vocab.save_to_files(checkpoint["vocab"])

        return checkpoint

    def restore(self, checkpoint):
        self.word_embeddings.load_state_dict(torch.load(checkpoint["embedding"]))
        self.model.load_state_dict(torch.load(checkpoint["encoder"]))
        self.vocab.from_files(checkpoint["vocab"])
    
    def get_train_instances(self):
        instances = self.train_instances
        if "vat" not in self.c.losses:
            filter_func= lambda instance: instance.fields["labeled"].metadata
            instances = list(filter(filter_func, self.train_instances))
        
        return instances
    
    def get_unlabeled_instances(self):
        filter_func= lambda instance: not instance.fields["labeled"].metadata
        instances = list(filter(filter_func, self.train_instances))
        
        return instances

    def get_iterator(self):
        iterator = self.iterator(self.get_train_instances(), num_epochs=1, shuffle=True)
        if self.c.misc.tqdm:
            iterator = tqdm(iterator)
        return iterator

    def set_train_mode(self):
        self.model.train()
        self.word_embeddings.train()
    
    def set_eval_mode(self):
        self.model.eval()
        self.word_embeddings.eval()

    def train_epoch(self):

        self.set_train_mode()
        logger.info("Epoch %s", self.iteration)

        for batch in self.get_iterator():
            self.optimizer.zero_grad()
            batch_to_device(batch, self.c.misc.device)
            embeddings = self.word_embeddings(batch["sentence"])
            mask = get_text_field_mask(batch["sentence"])
            logits = self.model(embeddings=embeddings, mask=mask)
            label = batch["label"]
            labeled = np.array(batch["labeled"], dtype=bool)

            if "ce" in self.losses and any(labeled):
                ce_loss = self.losses["ce"](logits[labeled], label[labeled])
                ce_loss.mean().backward(retain_graph=True)
                self.metrics.update("labeled_train", ce_loss=ce_loss)

            if "vat" in self.losses:
                model_forward = lambda embeddings: self.model(
                    embeddings=embeddings, mask=mask
                )
                vat_loss = self.losses["vat"](logits, model_forward, embeddings, mask)
                vat_loss.mean().backward()
                self.metrics.update("train", vat_loss=vat_loss)
            
            if "vat_old" in self.losses:
                model_forward = lambda embeddings: self.model(
                    embeddings=embeddings, mask=mask
                )
                vat_loss = self.losses["vat_old"](model_forward, logits, embeddings)
                vat_loss.mean().backward()
                self.metrics.update("train", vat_loss=vat_loss)


            tensors = from_locals(["logits", "mask", "label", "vat_loss"], loc=locals())

            if any(labeled):
                labeled_tensors = apply(tensors, lambda x: x[labeled])
                self.metrics.update("labeled_train", **labeled_tensors)

            if any(~labeled):
                unlabeled_tensors = apply(tensors, lambda x: x[~labeled])
                self.metrics.update("unlabeled_train", **unlabeled_tensors)

            self.optimizer.step()

        self.iteration += 1
        self.metrics.log()
        self.metrics.upload(step=self.iteration)
        self.metrics.reset()

    def evaluate(self, phase, instances):
        self.set_eval_mode()

        iterator = self.iterator(instances, num_epochs=1, shuffle=True)
        if self.c.misc.tqdm:
            iterator = tqdm(iterator)
        for batch in iterator:
            batch_to_device(batch, self.c.misc.device)
            embeddings = self.word_embeddings(batch["sentence"])
            mask = get_text_field_mask(batch["sentence"])
            logits = self.model(embeddings=embeddings, mask=mask)
            label = batch["label"]

            self.metrics.update(phase, logits=logits, mask=mask, label=label)

        self.metrics.log()
        self.metrics.upload(step=self.iteration)
        self.metrics.reset()

    def train_loop(self):
        instances_info(phase="Training", instances=self.train_instances)
        instances_info(phase="Val", instances=self.val_instances)
        for epoch in range(self.c.training.epochs):
            try:
                self.train_epoch()
                if self.iteration % self.c.training.checkpoint_freq == 0:
                    self.save(checkpoint_prefix=self.checkpoint_prefix)
                if self.iteration % self.c.training.eval_freq == 0:
                    self.evaluate("val", self.val_instances)
            except KeyboardInterrupt:
                logger.error("Keyboard Interupt")
                break
            except Exception as e:
                self.save(checkpoint_prefix=self.checkpoint_prefix)
                raise e
        if self.iteration % self.c.training.checkpoint_freq != 0:
            self.save()
            self.evaluate("val", self.val_instances)
        if self.c.misc.test:
            self.test_instances = self.dataset_reader.read(self.c.dataset.splits.test)
            instances_info(phase="Test", instances=self.test_instances)
            self.evaluate("test", self.test_instances)

#This decorator makes it posisble to have easy command line arguments and receive a cfg object
@hydra.main(config_path="config/train_text.yaml", strict=False)
def train_text(cfg):
    register_interpolations() 
    #This is to replace ${seed:} and ${id:} in the config with a seed and an id

    cfg_yaml = cfg.pretty(resolve=True) #We write the config to disk
    logger.info("====CONFIG====\n%s", cfg_yaml)
    save_config(cfg_yaml)
    set_seeds(cfg.misc.seed)

    #If wandb is activated we initialize it
    if cfg.misc.wandb:
        pass_conf(wandb.init, cfg, 'wandb')(config=cfg.to_container(resolve=True))

    tr = TrainText(cfg)

    tr.train_loop()

    destroy_trainer(tr)

if __name__ == "__main__":
    try:
        train_text()
    except Exception:
        logger.exception("Fatal error")
