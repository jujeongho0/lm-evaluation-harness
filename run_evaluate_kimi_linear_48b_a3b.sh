#!/bin/bash

# pip install tiktoken
# pip install flash-linear-attention

export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="moonshotai/Kimi-Linear-48B-A3B-Base"

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \
    --trust_remote_code \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks kmmlu \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \
    --trust_remote_code \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks kobest \
    --num_fewshot 10 \
    --batch_size auto:4 \
    --output_path results \
    --trust_remote_code \

# huggingface-cli login
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \
    --trust_remote_code \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks mbpp \
    --num_fewshot 3 \
    --batch_size auto:4 \
    --output_path results \
    --confirm_run_unsafe_code \
    --trust_remote_code \
