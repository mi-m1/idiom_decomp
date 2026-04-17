# idiom_decomp
Code repository for our ACL'26 paper:

> **Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning**

Read here: [`📝 Paper`](paper.pdf)


## Data

See [`data/`](data/README.md) for instructions on obtaining the required datasets (Bulkes et al., 2017 and IMPLI).

The processed data file (`data/processed/checked_manual_e_w_cql.csv`) contains pre-computed values for syntactic flexibility (CQL entropy), corpus frequency, and predictability. This file is not included in the repository and is available on request.


## Decomposability Measure (§3.2)

The core measure computes per-token importance scores for idiom words by masking each token and measuring the resulting drop in similarity between an idiom-in-context sentence and its figurative paraphrase (gloss).

**Script:** `decomp_measure/src/decomp.py`

**Arguments:**

| Argument | Description |
|---|---|
| `--model_name` | HuggingFace model ID (e.g., `bert-base-cased`, `answerdotai/ModernBERT-base`) |
| `--dataset_name` | Dataset name (`impli`) |
| `--dataset_path` | Path to the processed CSV file |
| `--layer` | Layer to extract embeddings from (`-1` = last layer) |
| `--sim_func` | Similarity function: `cos` (cosine), `cka` (centred kernel alignment), `wasser` (sliced Wasserstein) |
| `--agg_metric` | Aggregation metric: `entropy`, `gini`, `mean`, `sum`, `max` |
| `--save_dir` | Output directory for results CSV |
| `--drop_cql_cols` | Drop syntactic flexibility columns from output |
| `--testing` | Run on a small subset (5 rows) for testing |

**Example:**
```bash
python decomp_measure/src/decomp.py \
    --model_name "google-bert/bert-base-cased" \
    --dataset_name "impli" \
    --dataset_path "data/processed/checked_manual_e_w_cql.csv" \
    --sim_func "cos" \
    --agg_metric "entropy" \
    --layer -1 \
    --drop_cql_cols \
    --save_dir "decomp_measure/scores/impli/"
```

To run all model/similarity/metric combinations, use the shell scripts in `decomp_measure/scripts/`:
- `impli.sh` — BERT-base/large (cased and uncased)
- `impli_bert_layers.sh` — layer-wise BERT runs
- `impli_modernbert_layers.sh` — layer-wise ModernBERT runs


## Syntactic Flexibility (§3.3) and Frequency (§3.4)

Scripts in `syntactic_flexibility_frequency/enTenTen/` query the [Sketch Engine](https://www.sketchengine.eu/) API against the enTenTen corpus to retrieve CQL (corpus query language) hit counts across syntactic transformations, then compute entropy over those counts as the flexibility measure.

**Requires a Sketch Engine account.** Set your credentials in `get_cql.py` and `frequency_count.py`:
```python
USERNAME = 'YOUR_USERNAME'
API_KEY  = 'YOUR_API_KEY'
```

| Script | Description |
|---|---|
| `src/get_cql.py` | Queries Sketch Engine for CQL hits per syntactic transformation |
| `src/frequency_count.py` | Retrieves raw corpus frequency counts via the Sketch Engine API |
| `src/entropy.py` | Computes entropy over transformation frequencies (= syntactic flexibility score) |
| `scripts/script.sh` | Runs the full pipeline |

The pre-computed output is embedded in the processed data file (`data/processed/checked_manual_e_w_cql.csv`).


## Predictability (§3.5)

Predictability is computed as the masked last-word log-probability using `predictability/predictability.py`. Pre-computed values are stored in the processed data file.


## Correlation Experiments (§5)

Scripts in `section5_idh_experiments/` compute Spearman correlations between model decomposability scores and human ratings / syntactic flexibility, with category-wise (VP/PP/NP) breakdowns and layer-wise analysis.

Key scripts:
- `section5/correlation_by_bins.py` — category-wise correlations
- `section5/correlation_by_bins_layer.py` — layer-wise category correlations
- `layer_analysis.ipynb` — layer-wise heatmap analysis
- `visualisation/` — heatmap and distribution plots


## Acquisition-of-Training (AOT) Experiments (§6)

Scripts in `section6_aot/` track how idiom-related representations evolve across training checkpoints of OLMo-2 and OLMo-3.

| Script | Description |
|---|---|
| `src/aot_checkpoint.py` | Layer-wise cosine similarity across checkpoints |
| `src/aot_decomp.py` | Decomposability trajectories across checkpoints |
| `src/aot_surprisal.py` | Surprisal trajectories |
| `src/aot_frequency.py` | Frequency trajectories (reads pre-computed Infini-gram CSVs) |
| `src/aot_visualisation.py` | Learning curve plots (Fig. 2) |
| `src/1_lmm_data.py` | Prepare data for linear mixed models (Table 4) |
| `src/2_lmm_analysis.py` | Run linear mixed model analysis |

Shell scripts in `section6_aot/scripts/` wrap the OLMo-2 and OLMo-3 checkpoint runs.


## Robustness Analyses (Appendix)

- `robustness/src/bootstrap_ci.ipynb` — Bootstrap confidence intervals
- `robustness/src/partial_correlations.py` — Partial correlation analyses
- `rebuttal/src/model_decomp_vs_syntactic_flex.ipynb` — Model decomposability vs. syntactic flexibility


## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{idiom_decomp_acl26,
  title     = {Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning},
  author    = {},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026},
}
```

If you use the Bulkes et al. human norms or IMPLI datasets, please also cite the original authors — see [`data/README.md`](data/README.md) for their citation entries.
