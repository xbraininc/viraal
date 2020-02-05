import os
import gc
import torch
import logging
import wandb

logger= logging.getLogger(__name__)

def setup_wandb(cfg):
    #If wandb is activated we initialize it
    if cfg.misc.wandb:
        resolv_cfg = cfg.to_container(resolve=True)
        if cfg.select('wandb'):
            wandb.init(config=resolv_cfg, **resolv_cfg['wandb'])
        else:
            wandb.init(config=resolv_cfg)

def instances_info(phase, instances):
    message = f"{phase} instances : {len(instances)}. "
    nb_labeled = 0 if 'labeled' in instances[0].fields else None
    for ins in instances:
        if 'labeled' in ins.fields:
            nb_labeled += int(ins['labeled'].metadata)

    message += f"{nb_labeled} labeled" if nb_labeled else ""
    logger.info(message)

def ensure_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def destroy_trainer(tr):
    logger.info('Destroying trainer')
    wandb.join()
    del tr
    gc.collect()
    torch.cuda.empty_cache()

def batch_to_device(batch, device):
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
    for k, v in dico.items():
        if isinstance(v, dict):
            new_dico[k] = apply(v, func)
        else:
            new_dico[k] = func(v)
    return new_dico