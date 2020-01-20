#Pretrain

python -m viraal.train_text -m dataset=atis_intent \
                               training=atis \
                               losses=ce \
                               hydra=no_console \
                               misc.seed=293920:293928 \
                               misc.wandb=True \
                               wandb.group=atis_ce_attention \
                               wandb.project=viraal-pretrain \
                               ray.remote.num_gpus=0.5 \

python -m viraal.train_text -m dataset=atis_intent \
                               training=atis \
                               hydra=no_console \
                               hydra.sweep.dir=multiruns/pretrain/atis_vat_attention \
                               misc.seed=293920:293928 \
                               misc.wandb=True \
                               wandb.group=atis_vat_attention \
                               wandb.project=viraal-pretrain \
                               ray.remote.num_gpus=0.5 \

#Rerank
PRETRAIN_NAME="atis_vat_attention"
PRETRAIN_DIRS=`echo /u/home/badr/Expe/VirAAL/multiruns/pretrain/$PRETRAIN_NAME/{0..7} | tr ' ' ,`                   

python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
                           rerank.part=0.1 \
                           rerank.criteria=[random] \
                           misc.wandb=True \
                           wandb.project=viraal-rerank \
                           wandb.group=atis_random_attention \
                           hydra=no_console \
                           ray.remote.num_gpus=0.5

python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
                           rerank.part=0.1 \
                           rerank.criteria=[ce] \
                           misc.wandb=True \
                           wandb.project=viraal-rerank \
                           wandb.group=atis_ce_attention \
                           hydra=no_console \
                           ray.remote.num_gpus=0.5

python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
                           rerank.part=0.1 \
                           rerank.criteria=[clustering] \
                           misc.wandb=True \
                           wandb.project=viraal-rerank \
                           wandb.group=atis_cluster_attention \
                           hydra=no_console \
                           ray.remote.num_gpus=0.5

python -m viraal.rerank -m rerank=cluster_ce \
                           rerank.part=0.1 \
                           rerank.pretrain=$PRETRAIN_DIRS \
                           misc.wandb=True \
                           wandb.project=viraal-rerank \
                           wandb.group=atis_cluster_ce_attention\
                           hydra=no_console \
                           ray.remote.num_gpus=0.5