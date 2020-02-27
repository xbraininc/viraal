import logging
import os
import random

import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import wandb
from allennlp.nn.util import get_text_field_mask
from torch.distributions import Categorical
from sklearn.preprocessing import OneHotEncoder

from omegaconf import OmegaConf

from viraal.config import (flatten_dict, get_key, pass_conf,
                           register_interpolations, save_config, set_seeds)
from viraal.train.text import TrainText, batch_to_device, get_checkpoint
from viraal.core.utils import destroy_trainer, apply
from viraal.queries.k_center_greedy import k_center_greedy

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/rerank.yaml", strict=False)
def train_text(cfg):
    register_interpolations()

    pretrain_cfg = OmegaConf.load(os.path.join(cfg.rerank.pretrain, 'config.yaml'))

    cfg = OmegaConf.merge(pretrain_cfg, cfg)

    cfg_yaml = cfg.pretty(resolve=True)
    logger.info("====CONFIG====\n%s", cfg_yaml)
    save_config(cfg_yaml)
    set_seeds(cfg.misc.seed)

    if cfg.misc.wandb:
        pass_conf(wandb.init, cfg, 'wandb')(config=cfg.to_container(resolve=True))

    tr = TrainText(cfg)
    tr.restore(get_checkpoint(os.path.join(cfg.rerank.pretrain, 'model'),cfg.rerank.iteration))
    logger.info("Reranking training instances")
    rerank(tr, cfg)
    logger.info("Resetting model weights")
    tr.instantiate_model()
    logger.info("Training model")
    tr.train_loop()

    destroy_trainer(tr)

def normalize(criter):
    c = criter.detach().cpu().numpy()
    return c/np.quantile(c, 0.99)

def rerank(trainer, cfg):
    trainer.model.train()

    unlabeled = trainer.get_unlabeled_instances()
    nb_instances = len(unlabeled)
    to_select = int(cfg.rerank.part*len(trainer.train_instances))

    iterator = trainer.iterator(unlabeled, num_epochs=1)
    if cfg.misc.tqdm:
        iterator = tqdm(iterator)
    criter_epoch = []

    if cfg.rerank.presoftmax:
        presoftmax = []
        def hook(module, inp):
            presoftmax.append(inp[0].detach().cpu().numpy())
        trainer.model.output2label.register_forward_pre_hook(hook)

    for batch in iterator:
        batch_to_device(batch, cfg.misc.device)
        embeddings = trainer.word_embeddings(batch["sentence"])
        mask = get_text_field_mask(batch["sentence"])
        logits = trainer.model(embeddings=embeddings, mask=mask)
        labeled = np.array(batch["labeled"], dtype=bool)
        dist = Categorical(logits=logits)

        criter = np.zeros(logits.size(0))
        if "ce" in cfg.rerank.criteria:
            ce_criter = dist.entropy()
            criter += normalize(ce_criter)

        if "vat" in cfg.rerank.criteria:
            model_forward = lambda embeddings: trainer.model(
                embeddings=embeddings, mask=mask
            )
            vat_criter = trainer.losses["vat"](logits, model_forward, embeddings, mask)
            criter += normalize(vat_criter)
        
        if "random" in cfg.rerank.criteria:
            criter += np.random.rand(logits.size(0))

        criter_epoch.append(criter)

    criter_epoch = np.concatenate(criter_epoch)
    if cfg.rerank.presoftmax: presoftmax = np.concatenate(presoftmax)

    if "clustering" in cfg.rerank.criteria:
        clusterer = instantiate(cfg.rerank.cluster, to_select)
        logger.info("Clustering")
        clusters = clusterer.fit_predict(presoftmax) # (nb_samples,)
        clusters = OneHotEncoder(sparse=False).fit_transform(clusters.reshape(-1, 1)) #(nb_samples, nb_to_select)
        clusters = clusters.astype(bool)
        selected = []
        for i in range(to_select):
            cluster_instances = np.arange(nb_instances)[clusters[:,i]]
            selected.append(max(cluster_instances, key=lambda idx: criter_epoch[idx]))
    else:
        selected = np.argsort(criter_epoch)[-to_select:]

    for idx in selected:
        unlabeled[idx]['labeled'].metadata = True


if __name__ == "__main__":
    try:
        train_text()
    except Exception:
        logger.exception("Fatal error")
