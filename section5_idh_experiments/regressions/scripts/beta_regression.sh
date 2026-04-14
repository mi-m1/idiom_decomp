#!/bin/bash
# exec > output.log 2>&1

python section5_idh_experiments/regressions/src/regressions.py \
                "--data_path" "data/human/bulkes_tanner_data_subset.csv" \
                "--independent" "predictability" "log_frequency" \
                "--output_dir" "mixed_effect_analysis/results/" \

