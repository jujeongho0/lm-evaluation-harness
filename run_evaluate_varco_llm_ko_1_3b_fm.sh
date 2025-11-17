#!/bin/bash

MODEL_PATH="/path/to/varco-llm-ko-1.3b-FM"

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks kmmlu \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks kobest \
    --num_fewshot 10 \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results \

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},attn_implementation=flash_attention_2,parallelize=True \
    --tasks mbpp \
    --num_fewshot 3 \
    --batch_size auto:4 \
    --output_path results \
    --confirm_run_unsafe_code \
