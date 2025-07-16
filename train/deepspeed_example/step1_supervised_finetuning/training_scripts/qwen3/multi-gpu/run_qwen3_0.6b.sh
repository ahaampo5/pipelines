#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" = "" ]; then
    OUTPUT=./output_step1_qwen3_0.6b_single
fi
if [ "$ZERO_STAGE" = "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

BATCH_SIZE=1
ACCUMULATION_STEPS=8
MAX_LENGTH=8192
NUM_GPUS=4
TARGET=code_logic_math_simulation_stem_table

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="foundationModel"
export WANDB_NAME="0.6B_b_$BATCH_SIZE*$ACCUMULATION_STEPS*$NUM_GPUS-$MAX_LENGTH" # 6*2=12
export WANDB_NOTES="Smaller learning rate, more regularization."

# Issue
# /root/workspace/DeepSpeedExamples/raw_data/train/codegen__livecodebench_440.parquet \
DATA_DIR_PATH="/root/workspace/DeepSpeedExamples/raw_data"
# Code
LEET_CODE_DATASET=$DATA_DIR_PATH/train/codegen__leetcode2k_1.3k.parquet
PRIME_INTELLECT_DATASET=$DATA_DIR_PATH/train/codegen__primeintellect_7.5k.parquet
TACO_DATASET=$DATA_DIR_PATH/train/codegen__taco_8.8k.parquet
HUMAN_EVAL_DATASET=$DATA_DIR_PATH/offline_eval/codegen__humaneval_164.parquet
MBPP_DATASET=$DATA_DIR_PATH/offline_eval/codegen__mbpp_500.parquet
# Logic
ARC_AGI1_DATASET=$DATA_DIR_PATH/train/logic__arcagi1_111.parquet
ARC_AGI2_DATASET=$DATA_DIR_PATH/train/logic__arcagi2_190.parquet
BARC_DATASET=$DATA_DIR_PATH/train/logic__barc_1.6k.parquet
GRAPH_LOGICAL_DATASET=$DATA_DIR_PATH/train/logic__graph_logical_1.2k.parquet
ORDERING_PUZZLE_DATASET=$DATA_DIR_PATH/train/logic__ordering_puzzle_1.9k.parquet
ZEBRA_PUZZLE_DATASET=$DATA_DIR_PATH/train/logic__zebra_puzzle_1.3k.parquet
# Math
MATH_COMBINED_DATASET=$DATA_DIR_PATH/train/math__combined_54.4k.parquet
AIME_REPEATED_DATASET=$DATA_DIR_PATH/offline_eval/math__aime_repeated_8x_240.parquet
MATH_500_DATASET=$DATA_DIR_PATH/offline_eval/math__math_500.parquet
# SIMULATION
CODEIO_DATASET=$DATA_DIR_PATH/train/simulation__codeio_3.7k.parquet
# STEM
WEB_DATASET=$DATA_DIR_PATH/train/stem__web_3.6k.parquet
# Table
HITAB_DATASET=$DATA_DIR_PATH/train/table__hitab_4.3k.parquet
MULTIHIER_DATASET=$DATA_DIR_PATH/train/table__multihier_1.5k.parquet

DATA_PATH=""
if [[ "$TARGET" == *"code"* ]]; then
  DATA_PATH+=$LEET_CODE_DATASET,$PRIME_INTELLECT_DATASET,$TACO_DATASET,$HUMAN_EVAL_DATASET,$MBPP_DATASET
fi

if [[ "$TARGET" == *"logic"* ]]; then
  DATA_PATH+=$ARC_AGI1_DATASET,$ARC_AGI2_DATASET,$BARC_DATASET,$GRAPH_LOGICAL_DATASET,$ORDERING_PUZZLE_DATASET,$ZEBRA_PUZZLE_DATASET
fi

if [[ "$TARGET" == *"math"* ]]; then
  DATA_PATH+=$MATH_COMBINED_DATASET,$AIME_REPEATED_DATASET,$MATH_500_DATASET
fi

if [[ "$TARGET" == *"simulation"* ]]; then
  DATA_PATH+=$CODEIO_DATASET
fi

if [[ "$TARGET" == *"stem"* ]]; then
  DATA_PATH+=$WEB_DATASET
fi

if [[ "$TARGET" == *"table"* ]]; then
  DATA_PATH+=$HITAB_DATASET,$MULTIHIER_DATASET
fi

if [[ "$DATA_PATH" == "" ]]; then
  echo "No data path specified for target: $TARGET"
  exit 1
fi
deepspeed main.py \
   --data_path \
    mncai/math_sample_ko \
    mncai/math_sample_en \
    mncai/intellect_code_ko \
    mncai/intellect_code_en \
    mncai/foundation_model_smoltalk_en_15K_seed41 \
    mncai/foundation_model_smoltalk_ko_translate_15K_seed42 \
   --data_name default default default default default default \
   --data_split 1,0,0 \
   --model_name_or_path Qwen/Qwen3-0.6B \
   --per_device_train_batch_size $BATCH_SIZE \
   --per_device_eval_batch_size $BATCH_SIZE \
   --save_steps 10 \
   --max_seq_len $MAX_LENGTH \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps $ACCUMULATION_STEPS \
   --lr_scheduler_type cosine \
   --num_warmup_steps 500 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --dtype bf16 \
   --output_dir $OUTPUT \
   > $OUTPUT/training.log
