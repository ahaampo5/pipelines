#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_TIMEOUT=6000

BATCH_SIZE=1
ACCUMULATION_STEPS=2
MAX_LENGTH=16384
NUM_GPUS=8
TARGET=code_logic_math_stem_table

export WANDB_PROJECT="foundationModel"
export WANDB_NAME="0.6B_b_$BATCH_SIZE*$ACCUMULATION_STEPS*$NUM_GPUS-$MAX_LENGTH-$TARGET" # 6*2=12
export WANDB_NOTES="Smaller learning rate, more regularization."

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" = "" ]; then
    OUTPUT=./output_step1_qwen3_4b
fi
if [ "$ZERO_STAGE" = "" ]; then
    ZERO_STAGE=1
fi
mkdir -p $OUTPUT

deepspeed --hostfile=hostfile main.py \
   --data_path HuggingFaceTB/smoltalk \
   --data_name everyday-conversations \
   --data_split 2,4,4 \
   --model_name_or_path Qwen/Qwen3-4B \
   --per_device_train_batch_size $BATCH_SIZE \
   --per_device_eval_batch_size $BATCH_SIZE \
   --max_seq_len $MAX_LENGTH \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps $ACCUMULATION_STEPS \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --offload \
   --dtype bf16 \
   > $OUTPUT/training.log
