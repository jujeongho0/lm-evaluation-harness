#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="/path/to/wbl_model"

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8 \
    --tasks click \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=8192 \
    --apply_chat_template \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8 \
    --tasks kobalt \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=8192 \
    --apply_chat_template \
    --output_path results \
    
lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8 \
    --tasks hrm8k \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=8192 \
    --apply_chat_template \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8 \
    --tasks ifbench \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=8192 \
    --apply_chat_template \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8 \
    --tasks ifeval \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=8192 \
    --apply_chat_template \
    --output_path results \
