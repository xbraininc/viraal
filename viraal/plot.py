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
    regex_group = {
        'int' : f'{cfg.dataset}.*_int|{cfg.dataset}.*_joint',
        'tag' : f'{cfg.dataset}.*_tag|{cfg.dataset}.*_joint'
    } 
    test_metric = {
        'int' : "test_int_acc",
        'tag' : "test_tag conlleval f1"
    }
    y_title = {
        'int' : "Test int (acc)",
        'tag' : "Test tag (f1)"
    }

    title = {
        'atis' : "Performance on ATIS",
        'snips' : "Performance on SNIPS"
    }

    batch = {
        'atis' : {
            'ce' : 16,
            'vat' : 64
        },
        'snips' : {
            'ce' : 16 if cfg.less else 64,
            'vat' : 64
        }
    }
    xticks = {
        'atis' : [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        # 'snips' : [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        'snips' : [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09] if cfg.less  else [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    }

    runs = api.runs("cetosignis/viraal-final", {
            '$and' : [
                {'config.wandb.group': {'$regex': regex_group[cfg.task]}}, 
                {'$or':[
                    {'$and' : [{'config.training.loss':'ce'}, {'config.training.iterator.params.batch_size': batch[cfg.dataset]['ce']}]},
                    {'$and' : [{'config.training.loss':{'$regex': 'vat'}}, {'config.training.iterator.params.batch_size': batch[cfg.dataset]['vat']}]},
                ]},
                {'config.training.unlabeler.params.labeled_part': {'$lte':0.09 if cfg.less and cfg.dataset == 'snips' else 1.0}},
                {'config.training.unlabeler.params.labeled_part': {'$gte':0.1 if not cfg.less and cfg.dataset == 'snips' else 0.0}}
            ]
        }, per_page=1000)
    configs = ometrics.Metrics()
    summaries = ometrics.Metrics()
    print(len(runs))
    for run in runs:
        if  test_metric[cfg.task] in run.summary:
            configs.append(run.config)
            summaries.append(run.summary)
    df = pd.DataFrame()

    x = "Labeled part"
    y = y_title[cfg.task]
    less = 'less' if cfg.less else 'more'
    filename = f'{cfg.dataset}_{cfg.task}_{less}' 

    df["Task"] = configs["training/task"]
    df["Dataset"] = cfg.dataset
    df["Loss"] = configs["training/loss"]
    df["Batch Size"] = configs["training/iterator/params/batch_size"]
    df[x] = configs["training/unlabeler/params/labeled_part"]
    df[y] = summaries[test_metric[cfg.task]]
    yticks = np.arange(np.floor(df[y].min()*100)/100, np.ceil(df[y].max()*100)/100, 0.01)
    yticks = yticks if len(yticks) < 30 else np.arange(np.floor(df[y].min()*100)/100, np.ceil(df[y].max()*100)/100, 0.05)
    cols = ['Dataset', 'Task', 'Loss', 'Batch Size', x]
    df.groupby(cols).mean().to_csv(f'{filename}.csv')
    # df[y] = summaries["test_int_acc"]

    
    plt.figure()
    sns.set()
    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig = sns.lineplot(x=x, y=y, hue="Loss", style="Task", data=df, palette="Blues_d")
    plt.xticks(xticks[cfg.dataset], rotation=90)
    plt.yticks(yticks)
    plt.title(title[cfg.dataset])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    
if __name__ == "__main__":
    try:
        misc()
    except Exception:
        logger.exception("Fatal error")