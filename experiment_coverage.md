# Experiment Coverage

Mapping of paper sections to code in this repository.

| Paper section | Experiment | Code |
|---|---|---|
| §3.2 | Decomposability measure (BERT/ModernBERT, mask-and-score) | `decomp_measure/src/decomp_v2.py`, `decomp_measure/src/testing_search.py`, `decomp_measure/scripts/impli.sh`, `decomp_measure/scripts/impli_bert_layers.sh`, `decomp_measure/scripts/impli_modernbert_layers.sh` |
| §3.3 | Syntactic flexibility (CQL entropy via Sketch Engine) | No code — extracted via Sketch Engine web interface; output embedded in processed data file |
| §3.4 | Frequency (Sketch Engine / enTenTen) | No code — extracted externally; output embedded in processed data file |
| §3.5 | Predictability (masked last-word log-probability) | No code — pre-computed and stored in processed data file |
| §4.3 | Frequency for AOT (Infini-gram) | `aot/src/aot_frequency.py` (reads pre-computed Infini-gram CSVs; no querying script) |
| §5 | Spearman correlations across models and layers | `correlation_experiment/ranked_correlations.py`, `correlation_experiment/correlation_by_bins_layer.py`, `correlation_experiment/layer_analysis.ipynb` |
| §5 | Category-wise analysis (VP/PP/NP) | `correlation_experiment/correlation_by_bins.py`, `correlation_experiment/correlation_by_bins_layer_mostHuman.py` |
| §5 | Regression (Table 3, frequency x predictability) | `correlation_experiment/process_layer_results.ipynb` |
| Fig. 1 | Best-configuration decomp run | `rebuttal/src/decomp_best.sh` |
| §6 | AOT — checkpoint similarity (OLMo-2/3) | `aot/src/aot_checkpoint.py`, `aot/scripts/olmo2.sh`, `aot/scripts/olmo3.sh` |
| §6 | AOT — surprisal trajectories | `aot/src/aot_surprisal.py` |
| §6 | AOT — frequency trajectories | `aot/src/aot_frequency.py` |
| §6 | AOT — decomposability trajectories | `aot/src/aot_decomp.py` |
| §6 | LMM (Table 4) | `aot/src/1_lmm_data.py`, `aot/src/2_lmm_analysis.py` |
| Fig. 2 | Learning curve plots | `aot/src/aot_visualisation.py` |
| Appendix | Robustness analyses and bootstrap CI | `rebuttal/src/bootstrap_ci.ipynb`, `rebuttal/src/model_decomp_vs_syntactic_flex.ipynb` |

## Notes

The three measures with no accompanying code — syntactic flexibility, frequency, and predictability — are pre-computed values stored in `data/processed/checked_manual_e_w_cql.csv`. This file is not included in the repository. It is available on request or via [link to data release].
