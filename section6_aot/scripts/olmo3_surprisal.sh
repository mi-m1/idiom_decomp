#!/bin/bash

num_shards=$1
shard_idx=$2
if [ -z "$num_shards" ] || [ -z "$shard_idx" ]; then
  echo "Usage: $0 <num_shards> <shard_idx>"
  exit 1
fi

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/data/jade-beta/jade3396/cache/"
export HF_HUB_CACHE="/data/jade-beta/jade3396/cache/"
export HF_DATASETS_CACHE="/data/jade-beta/jade3396/cache/"
export HF_DATASETS_TRUST_REMOTE_CODE=true
source /data/jade-beta/jade3396/envs/nov2025/bin/activate

mkdir -p /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/aot/output/impli_surprisal

cd /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/aot/src
python surprisal_checkpoint.py \
  --model_name allenai/Olmo-3-1025-7B \
  --dtype bf16 \
  --num_checkpoints 100 0 \
  --idioms_file /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/data/processed/checked_manual_e_w_cql.csv \
  --output_dir /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/aot/output/impli_surprisal \
  --on_missing_span zeros \
  --num_shards $num_shards \
  --shard_idx $shard_idx
