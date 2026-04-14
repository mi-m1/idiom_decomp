#!/bin/bash

models=("bert-base-uncased" "bert-large-uncased" "bert-base-cased" "bert-large-cased" "answerdotai/ModernBERT-base" "answerdotai/ModernBERT-large")

# python predictability/predictability.py \
#     --model_name "answerdotai/ModernBERT-base" \
#     --dataset_name "impli" \
#     --dataset_path "data/impli/checked_manual_e.csv" \
#     --save_dir "predictability/results/"


for model in "${models[@]}"; do
    python3 predictability/predictability.py \
        --model_name "$model" \
        --dataset_name "impli" \
        --dataset_path "data/impli/checked_manual_e.csv" \
        --save_dir "predictability/results/"
done

for model in "${models[@]}"; do
    python3 predictability/predictability.py \
        --model_name "$model" \
        --dataset_name "liuhwa" \
        --dataset_path "data/liuhwa/liuhwa.csv" \
        --save_dir "predictability/results/"
done

