import hydra
import wandb
import plotly
import ometrics
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from viraal.config import pass_conf, get_func, register_interpolations, save_config, set_seeds
import seaborn as sns
from pprint import pprint

logger = logging.getLogger(__name__)

@hydra.main()
def misc(cfg):
    api = wandb.Api()
    runs = api.runs("cetosignis/viraal-rerank-full",per_page=3000)
    configs = ometrics.Metrics()
    summaries = ometrics.Metrics()
    print(len(runs))
    for run in runs:
        # if  test_metric[cfg.task] in run.summary:
        if 'test_int_acc' not in run.summary:
            run.summary['test_int_acc'] = -1
        if 'test_tag conlleval f1' not in run.summary:
            run.summary['test_tag conlleval f1'] = -1
        configs.append(run.config)
        summaries.append(run.summary)

    df = pd.DataFrame()

    x = "Labeled part"
    filename = f'rerank' 
    print(len(configs["training/task"]))
    print(len(configs["training/dataset"]))
    print(len(configs["training/loss"]))
    print(len(configs["rerank/criteria"]))
    print(len(summaries["test_int_acc"]))
    print(len(summaries["test_tag conlleval f1"]))
    df["Task"] = configs["training/task"]
    df["Dataset"] = configs["training/dataset"]
    df["Loss"] = configs["training/loss"]
    df["Criteria"] = [i[0] for i in configs["rerank/criteria"]]
    df["Test tag f1"] = summaries["test_int_acc"]
    df["Test int acc"] = summaries["test_tag conlleval f1"]
    df[x] = configs["training/unlabeler/params/labeled_part"]
    # df[y] = summaries[test_metric[cfg.task]]
    cols = ['Dataset', 'Task', 'Loss', 'Criteria', x]
    df.groupby(cols).mean().to_csv(f'{filename}.csv')
    
if __name__ == "__main__":
    try:
        misc()
    except Exception:
        logger.exception("Fatal error")