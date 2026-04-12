#!/bin/bash

###############  bert-large-uncased runs ############################
model="google-bert/bert-large-uncased"

now=$(date +"%Y-%m-%d_%H:%M:%S")
save_dir="examples_of_decomp/${model//\//_}/${now}/"
mkdir -p "$save_dir"


sim="wasser"
y="sum"
layer=23



mkdir -p "$save_dir"

python /Users/mmi/Documents/projects/idioms_decomposability/decomp_code/idioms_decomposability/decomp_measure/src/decomp_v2.py \
    --model_name "bert-large-uncased" \
    --dataset_name "impli" \
    --dataset_path "/Users/mmi/Documents/projects/idioms_decomposability/decomp_code/idioms_decomposability/data/processed/checked_manual_e_w_cql.csv" \
    --sim_func "$sim" \
    --agg_metric "$y" \
    --layer $layer \
    --drop_cql_cols \
    --save_dir "$save_dir" > src/logs/best_decomp.log 2>&1

