pretrain() {
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
                            wandb.project=$WANDB_PRETRAIN_PROJ_NAME \
                            ray.remote.num_gpus=0.5
}

rerank() {
    python -m viraal.rerank.$TYPE -m rerank.pretrain=$PRETRAIN_DIRS \
                                    rerank.criteria=[$RERANK] \
                                    misc.wandb=True \
                                    misc.test=True \
                                    wandb.project=$WANDB_RERANK_PROJ_NAME \
                                    wandb.group=${DATASET}_${TYPE}_l_${LOSS}_r_${RERANK}_p${PART}_r${PART} \
                                    hydra.sweep.dir=multiruns/rerank/final/$DATASET/$PRETRAIN_NAME/$RERANK \
                                    hydra=no_console \
                                    ray.remote.num_gpus=0.5
}

loop_on_expes() {
    for TYPE in ${TYPES[@]}
    do
        for index in ${!LOSSES[@]}
        do
            LOSS=${LOSSES[$index]}
            BATCH_SIZE=${BATCH_SIZES[$index]}
            for PART in ${PARTS[@]}
            do
                PRETRAIN_NAME="${DATASET}_${TYPE}_${LOSS}_${PART}"

                if [ "$DO_PRETRAIN" = true ] ; then
                    pretrain
                fi

                if [ "$DO_RERANK" = true ] ; then
                    PRETRAIN_DIRS=`echo /u/home/badr/Expe/VirAAL/multiruns/pretrain/final/$DATASET/$PRETRAIN_NAME/{0..7} | tr ' ' ,`    
                
                    for RERANK in ${CRITERIA[@]}
                    do
                        rerank
                    done
                fi
                
            done
        done
    done
}