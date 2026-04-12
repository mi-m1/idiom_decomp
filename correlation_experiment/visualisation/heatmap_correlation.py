import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# example: df = pd.read_csv("results.csv")
df = pd.read_csv("correlation_experiment/impli_correlation_results.csv")

#clean model names
model_map = {
    "google-bert/bert-base-uncased": "BERT-base Uncased",
    "google-bert/bert-base-cased": "BERT-base Cased",
    "google-bert/bert-large-uncased": "BERT-large Uncased",
    "google-bert/bert-large-cased": "BERT-large Cased",
    "answerdotai/ModernBERT-base": "ModernBERT-base",
    "answerdotai/ModernBERT-large": "ModernBERT-large",
}

df["model_clean"] = df["model"].map(model_map)

# # clean similarity function names
# sim_func_map = {
#     "cos": "Cosine",
#     "cka": "CKA",
#     "wasser": "Wasserstein",
# }

# df["sim_func"] = df["sim_func"].map(sim_func_map)

# # clean aggregation metric names
# agg_metric_map = {
#     "entropy": "Entropy",
#     "gini": "Gini",
# }

# df["agg_metric"] = df["agg_metric"].map(agg_metric_map)


# combine similarity + aggregation into one column
df["condition"] = df["sim_func"] + "-" + df["agg_metric"]

# pivot for heatmap
heatmap_data = df.pivot(
    index="model_clean",
    columns="condition",
    values="spearmanr"
)

# pivot p-values for significance masking
pvals = df.pivot(
    index="model_clean",
    columns="condition",
    values="p_value"
)

# significance mask
significant = pvals < 0.05

plt.figure(figsize=(10, 6))

ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Spearman ρ"}
)

# overlay significance markers
for y in range(significant.shape[0]):
    for x in range(significant.shape[1]):
        if significant.iloc[y, x]:
            ax.text(
                x + 0.88,
                y + 0.5,
                "•",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
            )

plt.xlabel("Similarity Function – Aggregation Metric")
plt.ylabel("Model")
# plt.title("Spearman Correlation Heatmap (• p < 0.05)")
plt.tight_layout()
# plt.show()
plt.savefig(
    "correlation_experiment/visualisation/plots/spearman_correlation_heatmap.pdf",
    dpi=300,
    bbox_inches="tight",
    format="pdf",
    )
