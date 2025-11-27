#!/bin/bash

export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="meta-llama/Llama-4-Scout-17B-16E"

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=eager,parallelize=True \
    --tasks leaderboard_mmlu_pro,leaderboard_bbh,leaderboard_gpqa,leaderboard_math_hard,leaderboard_musr \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=eager,parallelize=True \
    --tasks kmmlu_redux \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=eager,parallelize=True \
    --tasks kobest \
    --num_fewshot 10 \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=eager,parallelize=True \
    --tasks mbpp \
    --num_fewshot 3 \
    --batch_size auto:4 \
    --output_path results \
    --confirm_run_unsafe_code \
