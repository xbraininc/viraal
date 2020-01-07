#Pretrain
PRETRAIN_NAME="imdb_vat_attention"
# python -m viraal.train_text -m model=attention \
#                                hydra=no_console \
#                                hydra.sweep.dir=multiruns/pretrain/$PRETRAIN_NAME \
#                                misc.seed=293920:293928 \
#                                misc.wandb=True \
#                                wandb.group=$PRETRAIN_NAME \
#                                wandb.project=viraal-pretrain \

#Rerank
PRETRAIN_DIRS=`echo /u/home/badr/Expe/VirAAL/multiruns/pretrain/$PRETRAIN_NAME/{0..7} | tr ' ' ,`                   
# python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
#                            rerank.part=0.1 \
#                            misc.wandb=True \
#                            wandb.project=viraal-rerank \
#                            wandb.group=imdb_vat_pre_ce \
#                            hydra=no_console

python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
                           rerank.part=0.1 \
                           rerank.criteria=[vat] \
                           misc.wandb=True \
                           wandb.project=viraal-rerank \
                           wandb.group=imdb_vat_pre_vat\
                           hydra=no_console