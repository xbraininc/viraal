set -e

source base_functions.sh

DO_PRETRAIN=true
DO_RERANK=false

WANDB_PRETRAIN_PROJ_NAME=viraal-pretrain-full

## ======= ATIS

DATASET=atis
PARTS=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

## ======= ATIS text + tag

TYPES=(text tag)
LOSSES=(ce vat)
BATCH_SIZES=(16 64) #Each batch size corresponds to a loss

loop_on_expes

## ======= ATIS joint

TYPES=(joint)
LOSSES=(ce vat_joint)
BATCH_SIZES=(16 64)

loop_on_expes

## ======= SNIPS 

DATASET=snips
PARTS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

## ======== SNIPS text+tag

TYPES=(text tag)
LOSSES=(ce vat)
BATCH_SIZES=(64 64)

loop_on_expes

## ======= SNIPS joint

TYPES=(joint)
LOSSES=(ce vat_joint)
BATCH_SIZES=(64 64)

loop_on_expes