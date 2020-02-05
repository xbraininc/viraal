import hydra
import wandb
import plotly
import ometrics
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from viraal.config import pass_conf, get_func, register_interpolations, save_config, set_seeds
from viraal.queries.k_center_greedy import k_center_greedy 
import seaborn as sns

logger = logging.getLogger(__name__)

@hydra.main()
def misc(cfg):
    api = wandb.Api()
    runs = api.runs("cetosignis/viraal-final", {'config.wandb.group':{'$regex': 'atis.*joint|atis.*tag'}}, per_page=1000)
    configs = ometrics.Metrics()
    summaries = ometrics.Metrics()

    for run in runs:
        configs.append(run.config)
        summaries.append(run.summary)
    df = pd.DataFrame()

    x = "Labeled part"
    # y = "Test tag (f1)"
    y = "Test tag (f1)"

    df[x] = configs["training/unlabeler/params/labeled_part"]
    df["Loss"] = configs["training/loss"]
    df["Task"] = configs["training/task"]
    df[y] = summaries["test_tag conlleval f1"]
    # df[y] = summaries["test_int_acc"]

    sns.set()
    fig = sns.lineplot(x=x, y=y, hue="Task", style="Loss", data=df)
    plt.title("Performence on ATIS")
    plt.savefig("atis_tag.png", dpi=300)
    
if __name__ == "__main__":
    try:
        misc()
    except Exception:
        logger.exception("Fatal error")