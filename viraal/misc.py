import hydra
import wandb
import logging
import matplotlib.pyplot as plt
import numpy as np
from viraal.config import pass_conf, get_func, register_interpolations, save_config, set_seeds
from viraal.queries.k_center_greedy import k_center_greedy 

logger = logging.getLogger(__name__)

def setup(cfg):
    register_interpolations()

    cfg_yaml = cfg.pretty(resolve=True)
    logger.info("====CONFIG====\n%s", cfg_yaml)
    save_config(cfg_yaml)
    set_seeds(cfg.misc.seed)

    if cfg.misc.wandb:
        pass_conf(wandb.init, cfg, 'wandb')(config=cfg.to_container(resolve=True))

@hydra.main(config_path="config/misc.yaml", strict=False)
def misc(cfg):
    setup(cfg)

    dataset = get_func(cfg.dataset)

    X,y = dataset()

    selected = k_center_greedy(X, lambda x, y: np.linalg.norm(x-y), cfg.k_center.k)
    c = ["red" if i in selected else "blue" for i in range(len(X))]
    

    logger.info("Dataset shape : %s", X.shape)
    fig = plt.scatter(x=X[:,0], y=X[:,1], c=c)

    wandb.log({"dataset" : fig})

if __name__ == "__main__":
    try:
        misc()
    except Exception:
        logger.exception("Fatal error")