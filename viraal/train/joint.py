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
                          from_locals, instances_info, setup_wandb)

from viraal.core import metrics

from viraal.train.text import TrainText


logger = logging.getLogger(__name__)

def get_checkpoint(checkpoint_prefix, iteration):
    return {
        "model": os.path.join(checkpoint_prefix, f"model_{iteration}.th"),
        "vocab": os.path.join(checkpoint_prefix, "model.vocab"),
    }

class TrainJoint(TrainText):

    def instantiate_metrics(self):
        m = {
            'labeled_train': [metrics.Accuracy('int'), metrics.F1Conlleval('tag'), metrics.Average('ce_loss'), metrics.Average('vat_loss')],
            'unlabeled_train': [metrics.Accuracy('int'), metrics.F1Conlleval('tag'), metrics.Average('vat_loss')],
            'train': [metrics.Accuracy('int'), metrics.F1Conlleval('tag'), metrics.Average('vat_loss')],
            'val' : [metrics.Accuracy('int'), metrics.F1Conlleval('tag')],
            'test' : [metrics.Accuracy('int'), metrics.F1Conlleval('tag')]
        }
        self.metrics = instantiate(self.c.training.metrics, m,wandb=self.c.misc.wandb)

    def instantiate_model(self):
        
        self.instantiate_word_embedding()

        self.model = instantiate(
            self.c.model,
            input_size=self.word_embeddings.get_output_dim(),
            int_vocab_size=self.vocab.get_vocab_size("labels"),
            tag_vocab_size=self.vocab.get_vocab_size("tags")
        )
        #We assign the word embeddings to the model in order for it to be saved correctly
        self.model.word_embeddings = self.word_embeddings
        self.model.to(self.c.misc.device)

        self.optimizer = instantiate(
            self.c.training.optimizer,
            self.model.parameters(),
        )

    def get_train_instances(self):
        instances = self.train_instances
        if "vat_joint" not in self.c.losses:
            filter_func= lambda instance: instance.fields["labeled"].metadata
            instances = list(filter(filter_func, self.train_instances))
        
        return instances

    def train_epoch(self):

        self.model.train()
        logger.info("Epoch %s", self.iteration)

        for batch in self.get_iterator():
            self.optimizer.zero_grad()
            batch_to_device(batch, self.c.misc.device)
            embeddings = self.word_embeddings(batch["sentence"])
            mask = get_text_field_mask(batch["sentence"]).bool()
            int_logits, tag_logits = self.model(embeddings=embeddings, mask=mask)
            label, tags = batch["label"], batch["tags"]
            labeled = np.array(batch["labeled"], dtype=bool)

            if "ce" in self.losses and any(labeled):
                ce_loss = self.losses["ce"](int_logits[labeled], label[labeled])
                ce_loss = ce_loss.mean() + self.losses["ce"](tag_logits[labeled][mask[labeled]], tags[labeled][mask[labeled]]).mean()
                ce_loss.backward(retain_graph=True)
                self.metrics.update("labeled_train", ce_loss=ce_loss)

            if "vat_joint" in self.losses:
                model_forward = lambda embeddings: self.model(
                    embeddings=embeddings, mask=mask
                )
                vat_int_loss, vat_tag_loss = self.losses["vat_joint"]((int_logits, tag_logits), model_forward, embeddings, mask)
                vat_int_loss = vat_int_loss.mean()
                vat_tag_loss = vat_tag_loss.mean()
                vat_loss = vat_int_loss+vat_tag_loss
                vat_loss.backward()
                self.metrics.update("train", vat_loss=vat_loss)
            
            tensors = from_locals(["int_logits", "tag_logits", "tags", "mask", "label"], loc=locals())

            if any(labeled):
                labeled_tensors = apply(tensors, lambda x: x[labeled])
                self.metrics.update("labeled_train", name="int", logits=labeled_tensors['int_logits'], **labeled_tensors)
                self.metrics.update("labeled_train", name="tag", logits=labeled_tensors['tag_logits'], vocab=self.vocab, **labeled_tensors)

            if any(~labeled):
                unlabeled_tensors = apply(tensors, lambda x: x[~labeled])
                self.metrics.update("unlabeled_train", name="int", logits=unlabeled_tensors['int_logits'], **unlabeled_tensors)
                self.metrics.update("unlabeled_train", name="tag", logits=unlabeled_tensors['tag_logits'], vocab=self.vocab, **unlabeled_tensors)

            self.optimizer.step()

        self.iteration += 1
        self.metrics.log()
        self.metrics.upload(step=self.iteration)
        self.metrics.reset()

    def evaluate(self, phase, instances):
        self.model.eval()

        iterator = self.iterator(instances, num_epochs=1, shuffle=True)
        if self.c.misc.tqdm:
            iterator = tqdm(iterator)
        for batch in iterator:
            batch_to_device(batch, self.c.misc.device)
            embeddings = self.word_embeddings(batch["sentence"])
            mask = get_text_field_mask(batch["sentence"]).bool()
            int_logits, tag_logits = self.model(embeddings=embeddings, mask=mask)
            label, tags = batch["label"], batch["tags"]

            self.metrics.update(phase, name="int", logits=int_logits, label=label)
            self.metrics.update(phase, name="tag", logits=tag_logits, vocab=self.vocab, tags=tags, mask=mask)

        self.metrics.log()
        self.metrics.upload(step=self.iteration)
        self.metrics.reset()

#This decorator makes it posisble to have easy command line arguments and receive a cfg object
@hydra.main(config_path="../config/train_joint.yaml", strict=False)
def train_text(cfg):
    register_interpolations() 
    #This is to replace ${seed:} and ${id:} in the config with a seed and an id

    cfg_yaml = cfg.pretty(resolve=True) #We write the config to disk
    logger.info("====CONFIG====\n%s", cfg_yaml)
    save_config(cfg_yaml)
    set_seeds(cfg.misc.seed)

    setup_wandb(cfg)
    tr = TrainJoint(cfg)

    tr.train_loop()

    destroy_trainer(tr)

if __name__ == "__main__":
    try:
        train_text()
    except Exception:
        logger.exception("Fatal error")
