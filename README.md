# VirAAL repository

This repository implements the basic blocks needed for VirAAL in a configurable manner.

## Installation

Clone the repository in a local folder and then once in the folder install viraal

```
pip install -e .
```

## Repo structure

This repo is mainly based on configurability. There are scripts that receive the config and instantiate the different objects needed for the task at hand and make them interact.

As of this moment there are two main scripts: train_text.py and rerank.py

These scripts can be invoked as follows

```
python -m viraal.train_text
python -m viraal.rerank
```

### Config

The config is created by composing different blocks together. Hydra (the configuration manager) merges together the different parts and sends the whole config to the script. If no command line arguments are specified, the default `config/train_text.yaml` or `config/rerank.yaml` is used.

You can also override the defaults by specifying command line arguments. These arguments can either change a whole block `python -m viraal.train_text dataset=atis` or individual values `python -m viraal.train_text training.epochs=80`.

When a script is launched, hydra creates a new directory in the `runs` folder to store the run.

### Multiple experiments in parallel

One can also launch multiple experiments in parallel very simply 

```
python -m viraal.train_text -m dataset=atis,imdb5000,snips
python -m viraal.train_text -m misc.seed=100:110
```

Either using comma seperated values or a range.

This launches the experiments using ray. You can override in the configuration how many cpus and gpus a single run uses in `ray.remote`.

### TrainText

This object takes a configuration and instantiates objects directly from the config by importing the object and passing the parameters specified in the config to it.

```yaml
iterator:
    class: allennlp.data.iterators.basic_iterator.BasicIterator
    params:
      batch_size: 64
```

### Rerank

This script takes a path to a pretrained run and loads it to rerank the training data and trains a new classifier based on it.

## Weights and biases (wandb)

It is very useful to aggregate the results in a single place for comparing runs and visualizing what goes on during training. For that one can enable wandb using the `misc.wandb=True` flag in the config.

The following is an example of running multiple experiments to average them. Grouping them with wandb.group makes it possible to directly see the average as it evolves.

```
PRETRAIN_NAME="imdb_vat_attention"
python -m viraal.train_text -m dataset=imdb5000 \
                               model=attention \
                               hydra=no_console \
                               hydra.sweep.dir=multiruns/pretrain/$PRETRAIN_NAME \
                               misc.seed=293920:293928 \
                               misc.wandb=True \
                               wandb.group=$PRETRAIN_NAME \
                               wandb.project=viraal-pretrain \
```

## Data

The different paths to the datasets used can be found in the configurations in config/dataset

## Visualizing results

We use weights and biases to visualize results. Please refer to https://www.wandb.com/ for more detail.

For a quickstart, create an account at wandb and get your API key and then launch

```
wandb login
```

in a terminal. Then the results are automatically uploaded to your workspace online.