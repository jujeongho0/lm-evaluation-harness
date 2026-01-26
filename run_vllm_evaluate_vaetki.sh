#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="NC-AI-consortium-VAETKI/VAETKI"

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks leaderboard_mmlu_pro \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks leaderboard_bbh \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks leaderboard_gpqa \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks kmmlu_redux \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks kobest \
    --num_fewshot 10 \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks mbpp \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \
    --confirm_run_unsafe_code \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8 \
    --tasks leaderboard_musr \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True,max_gen_toks=32768 \
    --output_path results \

lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,max_length=32768 \
    --tasks ruler \
    --batch_size auto \
    --trust_remote_code \
    --gen_kwargs spaces_between_special_tokens=True \
    --metadata '{"max_seq_lengths":[4096,8192,16384,32768]}' \
    --output_path results \
