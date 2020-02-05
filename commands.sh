#Pretrain atis

# python -m viraal.train.text -m dataset=atis_intent \
#                                training=atis \
#                                training.loss=ce \
#                                training.task=int \
#                                losses=ce \
#                                hydra=no_console \
#                                misc.seed=1234500:1234508 \
#                                misc.test=True \
#                                training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#                                ray.remote.num_gpus=0.5 \
#                                wandb=final

# python -m viraal.train.text -m dataset=atis_intent \
#                                training=atis \
#                                training.loss=vat \
#                                training.task=int \
#                                losses=vat \
#                                hydra=no_console \
#                                misc.seed=1234500:1234508 \
#                                misc.test=True \
#                                training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#                                wandb=final \
#                                ray.remote.num_gpus=0.5

# python -m viraal.train.joint -m dataset=atis \
#                                training=atis \
#                                training.loss=ce \
#                                training.task=joint \
#                                losses=ce \
#                                hydra=no_console \
#                                misc.seed=1234500:1234508 \
#                                misc.test=True \
#                                training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#                                wandb=final \
#                                ray.remote.num_gpus=0.5 

# python -m viraal.train.joint -m dataset=atis \
#                                training=atis \
#                                training.loss=vat \
#                                training.task=joint \
#                                losses=vat_joint \
#                                hydra=no_console \
#                                misc.seed=1234500:1234508 \
#                                misc.test=True \
#                                training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#                                wandb=final \
#                                ray.remote.num_gpus=0.5 

# python -m viraal.train.tag -m dataset=atis \
#                                training=atis \
#                                training.loss=ce \
#                                training.task=tag \
#                                losses=ce \
#                                hydra=no_console \
#                                misc.seed=1234500:1234508 \
#                                misc.test=True \
#                                training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#                                wandb=final \
#                                ray.remote.num_gpus=0.5 

# python -m viraal.train.tag -m dataset=atis \
#                                training=atis \
#                                training.loss=vat \
#                                training.task=tag \
#                                losses=vat \
#                                hydra=no_console \
#                                misc.seed=1234500:1234508 \
#                                misc.test=True \
#                                training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#                                wandb=final \
#                                ray.remote.num_gpus=0.5 

#Pretrain snips

python -m viraal.train.text -m dataset=snips \
                               training=snips \
                               training.loss=ce \
                               training.task=int \
                               losses=ce \
                               hydra=no_console \
                               misc.seed=1234500:1234508 \
                               misc.test=True \
                               training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
                               ray.remote.num_gpus=0.5 \
                               wandb=final

python -m viraal.train.text -m dataset=snips \
                               training=snips \
                               training.loss=vat \
                               training.task=int \
                               losses=vat \
                               hydra=no_console \
                               misc.seed=1234500:1234508 \
                               misc.test=True \
                               training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
                               wandb=final \
                               ray.remote.num_gpus=0.5

python -m viraal.train.joint -m dataset=snips \
                               training=snips \
                               training.loss=ce \
                               training.task=joint \
                               losses=ce \
                               hydra=no_console \
                               misc.seed=1234500:1234508 \
                               misc.test=True \
                               training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
                               wandb=final \
                               ray.remote.num_gpus=0.5 

python -m viraal.train.joint -m dataset=snips \
                               training=snips \
                               training.loss=vat \
                               training.task=joint \
                               losses=vat_joint \
                               hydra=no_console \
                               misc.seed=1234500:1234508 \
                               misc.test=True \
                               training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
                               wandb=final \
                               ray.remote.num_gpus=0.5 

python -m viraal.train.tag -m dataset=snips \
                               training=snips \
                               training.loss=ce \
                               training.task=tag \
                               losses=ce \
                               hydra=no_console \
                               misc.seed=1234500:1234508 \
                               misc.test=True \
                               training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
                               wandb=final \
                               ray.remote.num_gpus=0.5 

python -m viraal.train.tag -m dataset=snips \
                               training=snips \
                               training.loss=vat \
                               training.task=tag \
                               losses=vat \
                               hydra=no_console \
                               misc.seed=1234500:1234508 \
                               misc.test=True \
                               training.unlabeler.params.labeled_part=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
                               wandb=final \
                               ray.remote.num_gpus=0.5 

####################

#Pretrain

# python -m viraal.train.***REMOVED*** -m dataset=atis_intent \
#                                training=atis \
#                                hydra=no_console \
#                                hydra.sweep.dir=multiruns/pretrain/atis_vat_***REMOVED*** \
#                                misc.seed=293920:293928 \
#                                misc.wandb=True \
#                                misc.test=True \
#                                wandb.group=atis_vat_***REMOVED*** \
#                                wandb.project=viraal-pretrain \
#                                ray.remote.num_gpus=0.5 \

# Rerank
# PRETRAIN_NAME="atis_vat_***REMOVED***"
# PRETRAIN_DIRS=`echo /u/home/badr/Expe/VirAAL/multiruns/pretrain/$PRETRAIN_NAME/{0..7} | tr ' ' ,`                   

# python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
#                            rerank.part=0.1 \
#                            rerank.criteria=[random] \
#                            misc.wandb=True \
#                            wandb.project=viraal-rerank \
#                            wandb.group=atis_random_attention \
#                            hydra=no_console \
#                            ray.remote.num_gpus=0.5

# python -m viraal.rerank.***REMOVED*** -m rerank.pretrain=$PRETRAIN_DIRS \
#                            rerank.part=0.1 \
#                            rerank.criteria=[ce] \
#                            misc.wandb=True \
#                            misc.test=True \
#                            wandb.project=viraal-rerank \
#                            wandb.group=atis_ce_***REMOVED*** \
#                            hydra=no_console \
#                            ray.remote.num_gpus=0.5

# python -m viraal.rerank -m rerank.pretrain=$PRETRAIN_DIRS \
#                            rerank.part=0.1 \
#                            rerank.criteria=[clustering] \
#                            misc.wandb=True \
#                            wandb.project=viraal-rerank \
#                            wandb.group=atis_cluster_attention \
#                            hydra=no_console \
#                            ray.remote.num_gpus=0.5

# python -m viraal.rerank -m rerank=cluster_ce \
#                            rerank.part=0.1 \
#                            rerank.pretrain=$PRETRAIN_DIRS \
#                            misc.wandb=True \
#                            wandb.project=viraal-rerank \
#                            wandb.group=atis_cluster_ce_attention\
#                            hydra=no_console \
#                            ray.remote.num_gpus=0.5
