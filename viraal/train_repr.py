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
import pandas as pd

import umap

from viraal.config import (flatten_dict, get_key, pass_conf,
                           register_interpolations, save_config, set_seeds)
from viraal.core.utils import (apply, batch_to_device, destroy_trainer, ensure_dir,
                          from_locals, instances_info)

from viraal.train_text import TrainText

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
        dfs = []
        for epoch in tqdm(range(self.c.training.epochs)):
            try:
                self.train_epoch()
                if epoch > 0: epoch_repr_prev = epoch_repr
                epoch_repr = self.evaluate_repr(self.get_unlabeled_instances())
                epoch_repr_arr = sorted(epoch_repr.items(), key=lambda x : x[0])
                epoch_repr_idx = np.stack([e[0] for e in epoch_repr_arr])
                epoch_repr_arr = np.stack([e[1] for e in epoch_repr_arr])
                if epoch % 10 == 0:
                    reducer = umap.UMAP().fit(epoch_repr_arr)
                epoch_repr_2D = reducer.transform(epoch_repr_arr)

                if epoch > 0:
                    to_select = int(self.c.rerank.part*len(self.train_instances))
                    distance = {
                            idx : np.linalg.norm(epoch_repr[idx]-epoch_repr_prev[idx])
                            for idx in epoch_repr
                        }
                    selected = sorted(distance, key=lambda idx: distance[idx])[-to_select:]
                
                df = pd.DataFrame()
                df['x'] = epoch_repr_2D[:,0]
                df['y'] = epoch_repr_2D[:,1]
                df['epoch'] = epoch
                df['idx'] = epoch_repr_idx
                df['selected'] = 0
                if epoch > 0:
                    for i in selected:
                        df.loc[df.idx==i, 'selected'] = 1
                dfs.append(df)
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

        import plotly
        import plotly.express as px

        fig = px.scatter(pd.concat(dfs), x='x', y='y', animation_frame='epoch', animation_group='idx', color='selected')

        ensure_dir('figures')
        plotly.offline.plot(fig, 'figures/evol.html')

#This decorator makes it posisble to have easy command line arguments and receive a cfg object
@hydra.main(config_path="config/rerank_repr.yaml", strict=False)
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
    tr.train_loop()

    destroy_trainer(tr)

if __name__ == "__main__":
    try:
        train_text()
    except Exception:
        logger.exception("Fatal error")
