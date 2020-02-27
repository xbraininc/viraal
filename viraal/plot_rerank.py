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
from pprint import pprint

logger = logging.getLogger(__name__)

@hydra.main()
def misc(cfg):
    api = wandb.Api()
    test_metric = {
        'int' : "test_int_acc",
        'tag' : "test_tag conlleval f1"
    }
    y_title = {
        'int' : "Test int (acc)",
        'tag' : "Test tag (f1)"
    }

    title = {
        'atis' : f"Performance on ATIS",
        'snips' : f"Performance on SNIPS"
    }

    batch = {
        'atis' : {
            'ce' : 16,
            'vat' : 64
        },
        'snips' : {
            'ce' :  64,
            'vat' : 64
        }
    }
    xticks = {
        'atis' : [0.1, 0.2, 0.4],
        # 'snips' : [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        'snips' : [0.02,0.04,0.06,0.08,0.1,0.2,0.4]
    }

    runs = api.runs("cetosignis/viraal-rerank-full", {
            '$and' : [
                {'config.training.dataset': cfg.dataset},
                {'config.training.task': cfg.task} 
            ]
        }, per_page=1000)
    configs = ometrics.Metrics()
    summaries = ometrics.Metrics()
    print(len(runs))
    for run in runs:
        if  test_metric[cfg.test_task] in run.summary:
            configs.append(run.config)
            summaries.append(run.summary)
    df = pd.DataFrame()

    x = "Labeled part"
    y = y_title[cfg.test_task]
    filename = f'{cfg.dataset}_{cfg.task}_{cfg.test_task}' 

    df["Task"] = configs["training/task"]
    df["Dataset"] = cfg.dataset
    df["Loss"] = configs["training/loss"]
    df["Batch Size"] = configs["training/iterator/params/batch_size"]
    df["Criteria"] = [i[0] for i in configs["rerank/criteria"]]
    df["Loss+Criteria"] = df["Loss"] + "+" + df["Criteria"]
    df[x] = configs["training/unlabeler/params/labeled_part"]
    df[y] = summaries[test_metric[cfg.test_task]]
    yticks = np.arange(np.floor(df[y].min()*100)/100, np.ceil(df[y].max()*100)/100, 0.01)
    yticks = yticks if len(yticks) < 30 else np.arange(np.floor(df[y].min()*100)/100, np.ceil(df[y].max()*100)/100, 0.05)
    cols = ['Dataset', 'Task', 'Loss', 'Batch Size', 'Criteria', x]
    df.groupby(cols).mean().to_csv(f'{filename}.csv')
    # df[y] = summaries["test_int_acc"]

    
    plt.figure()
    sns.set()
    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig = sns.barplot(x=x, y=y, hue="Loss+Criteria", data=df, palette="Blues_d")
    plt.ylim(yticks[0],yticks[-1])
    plt.title(title[cfg.dataset])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    
if __name__ == "__main__":
    try:
        misc()
    except Exception:
        logger.exception("Fatal error")