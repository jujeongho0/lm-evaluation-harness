#!/bin/bash

export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="/path/to/WBL-112B-A10B-HF"

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks leaderboard_mmlu_pro \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks leaderboard_bbh \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks leaderboard_gpqa \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks kmmlu_redux \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks kobest \
    --num_fewshot 10 \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks mbpp \
    --batch_size auto \
    --output_path results \
    --trust_remote_code \
    --confirm_run_unsafe_code \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks leaderboard_musr \
    --batch_size auto \
    --trust_remote_code \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True,max_length=32768 \
    --tasks ruler \
    --batch_size auto \
    --trust_remote_code \
    --metadata '{"max_seq_lengths":[4096,8192,16384,32768]}' \
    --output_path results \
