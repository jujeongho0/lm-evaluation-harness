#!/bin/bash

export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="/path/to/wbl_model"

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks click \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs max_gen_toks=8192 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks hrm8k \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs max_gen_toks=8192 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks ifbench \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs max_gen_toks=8192 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks ifeval \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs max_gen_toks=8192 \
    --output_path results \
