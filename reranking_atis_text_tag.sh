set -e

DATASET=atis
PARTS=(
    0.2
)
TYPES=(
    tag
)
LOSSES=(
    vat
)
BATCH_SIZES=(
    64
)

for TYPE in ${TYPES[@]}
do
    for index in ${!LOSSES[@]}
    do
        LOSS=${LOSSES[$index]}
        BATCH_SIZE=${BATCH_SIZES[$index]}
        for PART in ${PARTS[@]}
        do
            PRETRAIN_NAME="${DATASET}_${TYPE}_${LOSS}_${PART}"

            python -m viraal.train.$TYPE -m dataset=$DATASET \
                                        training=$DATASET \
                                        training.dataset=$DATASET \
                                        training.loss=$LOSS \
                                        training.task=$TYPE \
                                        training.iterator.params.batch_size=${BATCH_SIZE} \
                                        training.unlabeler.params.labeled_part=$PART \
                                        hydra=no_console \
                                        losses=$LOSS \
                                        hydra.sweep.dir=multiruns/pretrain/final/$DATASET/$PRETRAIN_NAME \
                                        misc.seed=293920:293928 \
                                        misc.wandb=True \
                                        misc.test=True \
                                        wandb.group=$PRETRAIN_NAME \
                                        wandb.project=viraal-pretrain-full \
                                        ray.remote.num_gpus=0.5
            
            PRETRAIN_DIRS=`echo /u/home/badr/Expe/VirAAL/multiruns/pretrain/final/$DATASET/$PRETRAIN_NAME/{0..7} | tr ' ' ,`    
            
            
            for RERANK in ce random 
            do
                python -m viraal.rerank.$TYPE -m rerank.pretrain=$PRETRAIN_DIRS \
                                                rerank.criteria=[$RERANK] \
                                                misc.wandb=True \
                                                misc.test=True \
                                                wandb.project=viraal-rerank-full \
                                                wandb.group=${DATASET}_${TYPE}_l_${LOSS}_r_${RERANK}_p${PART}_r${PART} \
                                                hydra=no_console \
                                                ray.remote.num_gpus=0.5
            done
        done
    done
done