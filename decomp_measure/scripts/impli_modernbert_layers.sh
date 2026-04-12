#!/usr/bin/env bash
set -euo pipefail

sim_func=("cos" "cka" "wasser")
aff_metric=("entropy" "gini" "mean" "max" "sum")
models=(
  "answerdotai/ModernBERT-base"
  "answerdotai/ModernBERT-large"
)

# Map model -> space-separated list of layers
get_layers() {
  case "$1" in
    "answerdotai/ModernBERT-base")   echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21" ;;
    "answerdotai/ModernBERT-large") echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27" ;;
    *)
      echo "Unknown model: $1" >&2
      return 1
      ;;
  esac
}

now="$(date +"%Y-%m-%d_%H:%M:%S")"


for model in "${models[@]}"; do
    layers="$(get_layers "$model")"

    for layer in $layers; do
        for sim in "${sim_func[@]}"; do
            for y in "${aff_metric[@]}"; do

                echo "Processing model: $model, layer: $layer, sim: $sim, metric: $y"
                save_dir="decomp_measure/scores/impli_layers/${model//\//_}/${now}"

                save_dir_layer="${save_dir}/layer_${layer}"
                mkdir -p "$save_dir_layer"

                python decomp_measure/src/decomp_v2.py \
                    --model_name "$model" \
                    --dataset_name "impli" \
                    --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
                    --sim_func "$sim" \
                    --agg_metric "$y" \
                    --layer "$layer" \
                    --drop_cql_cols \
                    --save_dir "$save_dir_layer"
            done
        done
    done
done

# for model in "${models[@]}"; do
#   layers="$(get_layers "$model")"

#   for layer in $layers; do
#     for sim in "${sim_func[@]}"; do
#       for y in "${aff_metric[@]}"; do
#         echo "Processing model: $model, layer: $layer, sim: $sim, metric: $y"
#       done
#     done
#   done
# done