#!/bin/bash


python syntax_frequency/src/frequency_count.py \
    "--dataset_name" "liuhwa" \
    "--shard_path" "data/liuhwa/shard_0.csv" \
    "--save_dir" "data/frequencies/liuhwa" \
    "--n_shards" "2" \
    "--total_data" "data/liuhwa/liuhwa_w_cql.csv" \
    "--username" "golz.atefi" \
    "--api_key" "62521cfb5c26f93852c64fe899b931b6" 

python syntax_frequency/src/frequency_count.py \
    "--dataset_name" "liuhwa" \
    "--shard_path" "data/liuhwa/shard_1.csv" \
    "--save_dir" "data/frequencies/liuhwa" \
    "--n_shards" "2" \  
    "--total_data" "data/liuhwa/liuhwa_w_cql.csv" \
    "--username" "YOUR_USERNAME" \
    "--api_key" "86c1d1e1d09cf582469923206482431c" 

