set -e

source base_functions.sh

DO_PRETRAIN=false # Set this to true if you haven't launched the pretrain experiments before
DO_RERANK=true

WANDB_PRETRAIN_PROJ_NAME=viraal-pretrain-full
WANDB_RERANK_PROJ_NAME=viraal-rerank-full

## ======= RERANK PARAMS

CRITERIA=(ce random)

## ======= ATIS

DATASET=atis
PARTS=(0.05 0.1 0.15 0.2 0.25 0.3)

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
PARTS=(0.01 0.02 0.03 0.04 0.05 0.1 0.2)

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