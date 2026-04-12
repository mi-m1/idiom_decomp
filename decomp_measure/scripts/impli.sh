#!/bin/bash

sim_func=("cos" "cka" "wasser")
aff_metric=("entropy" "gini")


# ###############  bert-base-uncased runs ############################
# now=$(date +"%Y-%m-%d_%H:%M:%S")
# save_dir="decomp_measure/scores/impli/${now}/"
# mkdir -p "$save_dir"

# for sim in "${sim_func[@]}"; do
#     for y in "${aff_metric[@]}"; do

#         mkdir -p "$save_dir"

#         python decomp_measure/src/decomp_v2.py \
#             --model_name "bert-base-uncased" \
#             --dataset_name "impli" \
#             --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
#             --sim_func "$sim" \
#             --agg_metric "$y" \
#             --drop_cql_cols \
#             --save_dir "$save_dir"
#     done
# done

###############  bert-base-cased runs ############################
model="google-bert/bert-base-cased"
now=$(date +"%Y-%m-%d_%H:%M:%S")
save_dir="decomp_measure/scores/impli/${model//\//_}/${now}/"
mkdir -p "$save_dir"

for sim in "${sim_func[@]}"; do
    for y in "${aff_metric[@]}"; do

        mkdir -p "$save_dir"

        python decomp_measure/src/decomp_v2.py \
            --model_name "bert-base-cased" \
            --dataset_name "impli" \
            --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
            --sim_func "$sim" \
            --agg_metric "$y" \
            --drop_cql_cols \
            --save_dir "$save_dir"
    done
done

###############  bert-large-uncased runs ############################
model="google-bert/bert-large-uncased"

now=$(date +"%Y-%m-%d_%H:%M:%S")
save_dir="decomp_measure/scores/impli/${model//\//_}/${now}/"
mkdir -p "$save_dir"

for sim in "${sim_func[@]}"; do
    for y in "${aff_metric[@]}"; do

        mkdir -p "$save_dir"

        python decomp_measure/src/decomp_v2.py \
            --model_name "bert-large-uncased" \
            --dataset_name "impli" \
            --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
            --sim_func "$sim" \
            --agg_metric "$y" \
            --drop_cql_cols \
            --save_dir "$save_dir"
    done
done

###############  bert-large-cased runs ############################
model="google-bert/bert-large-cased"

now=$(date +"%Y-%m-%d_%H:%M:%S")
save_dir="decomp_measure/scores/impli/${model//\//_}/${now}/"
mkdir -p "$save_dir"

for sim in "${sim_func[@]}"; do
    for y in "${aff_metric[@]}"; do

        mkdir -p "$save_dir"

        python decomp_measure/src/decomp_v2.py \
            --model_name "bert-large-cased" \
            --dataset_name "impli" \
            --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
            --sim_func "$sim" \
            --agg_metric "$y" \
            --drop_cql_cols \
            --save_dir "$save_dir"
    done
done


# ############## ModernBERT-base runs ##############

# model="answerdotai/ModernBERT-base"
# now=$(date +"%Y-%m-%d_%H:%M:%S")
# save_dir="decomp_measure/scores/impli/${model//\//_}/${now}/"
# mkdir -p "$save_dir"

# for sim in "${sim_func[@]}"; do
#     for y in "${aff_metric[@]}"; do

#         mkdir -p "$save_dir"

#         python decomp_measure/src/decomp_v2.py \
#             --model_name "answerdotai/ModernBERT-base" \
#             --dataset_name "impli" \
#             --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
#             --sim_func "$sim" \
#             --agg_metric "$y" \
#             --drop_cql_cols \
#             --save_dir "$save_dir"
#     done
# done



# ############## ModernBERT-large runs ##############

# model="answerdotai/ModernBERT-large"
# now=$(date +"%Y-%m-%d_%H:%M:%S")
# save_dir="decomp_measure/scores/impli/${model//\//_}/${now}/"
# mkdir -p "$save_dir"

# for sim in "${sim_func[@]}"; do
#     for y in "${aff_metric[@]}"; do

#         mkdir -p "$save_dir"

#         python decomp_measure/src/decomp_v2.py \
#             --model_name "answerdotai/ModernBERT-large" \
#             --dataset_name "impli" \
#             --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
#             --sim_func "$sim" \
#             --agg_metric "$y" \
#             --drop_cql_cols \
#             --save_dir "$save_dir"
#     done
# done