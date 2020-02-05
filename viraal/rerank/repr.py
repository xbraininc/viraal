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

import umap

from viraal.config import (flatten_dict, get_key, pass_conf,
                           register_interpolations, save_config, set_seeds)
from viraal.core.utils import (apply, batch_to_device, destroy_trainer, ensure_dir,
                          from_locals, instances_info)

from viraal.train.text import TrainText

logger = logging.getLogger(__name__)

class TrainRepr(TrainText):

    def evaluate_repr(self, instances):
        self.model.eval()

        internal_tensors = {}
        def hook(name, module, inp):
            internal_tensors[name] = inp[0].detach().cpu().numpy()
        self.model.output2label.register_forward_pre_hook(lambda module, inp: hook('presoftmax', module, inp))

        presoft_repr = {}

        iterator = self.iterator(instances, num_epochs=1, shuffle=False)
        if self.c.misc.tqdm:
            iterator = tqdm(iterator)
        for batch in iterator:
            batch_to_device(batch, self.c.misc.device)
            embeddings = self.word_embeddings(batch["sentence"])
            mask = get_text_field_mask(batch["sentence"])
            logits = self.model(embeddings=embeddings, mask=mask)
            idx = batch["idx"]

            for i, t in zip(idx, internal_tensors['presoftmax']):
                presoft_repr[i] = t
    
        return presoft_repr

    def train_loop(self):
        instances_info(phase="Training", instances=self.train_instances)
        instances_info(phase="Val", instances=self.val_instances)
        for epoch in range(self.c.training.epochs):
            try:
                if epoch == self.c.training.epochs-1:
                    logger.info('Evaluating representation before last epoch')
                    before_last_repr = self.evaluate_repr(self.get_unlabeled_instances())
                self.train_epoch()
                if epoch == self.c.training.epochs-1:
                    logger.info('Evaluating representation after last epoch')
                    last_repr = self.evaluate_repr(self.get_unlabeled_instances())
    
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
        
        return before_last_repr, last_repr

#This decorator makes it posisble to have easy command line arguments and receive a cfg object
@hydra.main(config_path="../config/rerank_repr.yaml", strict=False)
def train_text(cfg):
    register_interpolations()

    cfg_yaml = cfg.pretty(resolve=True)
    logger.info("====CONFIG====\n%s", cfg_yaml)
    save_config(cfg_yaml)
    set_seeds(cfg.misc.seed)

    if cfg.misc.wandb:
        pass_conf(wandb.init, cfg, 'wandb')(config=cfg.to_container(resolve=True))

    tr = TrainRepr(cfg)
    logger.info("Training model")
    bf_last, last = tr.train_loop()
    logger.info("Reranking training instances")
    rerank(tr, cfg, bf_last, last)
    logger.info("Resetting model weights")
    tr.instantiate_model()
    logger.info("Training model")
    tr.train_loop()

    destroy_trainer(tr)

def rerank(trainer, cfg, bf_last, last):
    to_select = int(cfg.rerank.part*len(trainer.train_instances))
    unlabeled = trainer.get_unlabeled_instances()
    unlabeled = {
        instance["idx"].metadata : instance 
        for instance in unlabeled
    }
    distance = {}
    for idx in bf_last:
        distance[idx] = np.linalg.norm(bf_last[idx]-last[idx])
    selected = sorted(distance, key=lambda idx: distance[idx])[-to_select:]
    for idx in selected:
        unlabeled[idx]['labeled'].metadata = True

if __name__ == "__main__":
    try:
        train_text()
    except Exception:
        logger.exception("Fatal error")
