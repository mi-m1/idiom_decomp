#!/bin/bash

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/data/jade-beta/jade3396/cache/"
export HF_HUB_CACHE="/data/jade-beta/jade3396/cache/"
export HF_DATASETS_CACHE="/data/jade-beta/jade3396/cache/"
export HF_DATASETS_TRUST_REMOTE_CODE=true
source /data/jade-beta/jade3396/envs/nov2025/bin/activate

cd /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/infini_freq/src
python main.py \
    --input /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/data/processed/checked_manual_e_w_cql.csv \
    --output /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/infini_freq/output/checked_manual_e_w_cql_olmo2-freq.csv \
    --column base_form \
    --index v4_olmo-mix-1124_llama \
    --sleep-s 1.0 \
    --retries 2 \
    --timeout-s 30
