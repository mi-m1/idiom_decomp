import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# df = pd.read_csv("decomp_measure/scores/impli/google-bert_bert-large-cased/2025-12-22_19:16:23/impli_wasser_gini_bert-large-cased.csv")

# df = pd.read_csv("decomp_measure/scores/impli_layers/google-bert_bert-large-cased/layer_1/2025-12-23_12:34:45/impli_layers_wasser_gini_bert-large-cased.csv")
# df = pd.read_csv("/Users/mmi/Documents/projects/idioms_decomposability/decomp_code/idioms_decomposability/decomp_measure/scores/impli_layers/answerdotai_ModernBERT-base/2025-12-25_02:34:31/layer_0/impli_wasser_gini_ModernBERT-base.csv")

# model = "answerdotai_ModernBERT-large"
# layer = "layer_1"
# condition = "wasser_gini"

# df = pd.read_csv(f"decomp_measure/scores/impli_layers/{model}/2025-12-25_02:34:31/{layer}/impli_{condition}_ModernBERT-large.csv")
# sns.stripplot(data=df, y="decomp_score", )
# plt.show()

def find_score_file_layers(project_dir, dataset, model, layer, sim, agg):
    model_dir = model.replace("/", "_")
    model_name = model.split("/")[-1]

    base_path = (
        Path(project_dir)
        / "decomp_measure"
        / "scores"
        / dataset
        / model_dir
    )

    pattern = f"impli_{sim}_{agg}_{model_name}.csv"

    matches = list(base_path.glob(f"*/{layer}/{pattern}"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for {pattern} under {base_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for {pattern}: {matches}")

    return matches[0]


if __name__ == "__main__":

    project_dir = "."

    sim_funcs = ["cos", "cka", "wasser"]
    agg_metrics = ["entropy", "gini", "max", "sum", "mean"]

    # datasets = ["impli", "liuhwa"]
    datasets = ["impli_layers"]

    models = [
        "google-bert/bert-base-uncased",
        "google-bert/bert-base-cased",
        "google-bert/bert-large-uncased",
        "google-bert/bert-large-cased",
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ]

    layer = "layer_1"

    for dataset in datasets:
        rows = []

        for model in models:
            for sim in sim_funcs:
                for agg in agg_metrics:

                    csv_path = find_score_file_layers(
                        project_dir,
                        dataset,
                        model,
                        layer,
                        sim,
                        agg,
                    )

                    df = pd.read_csv(csv_path)

                    df = df.rename(columns={"decomp_score": "Decomposability Score"})

                    print(f"{df.base_form.to_list()}")
                    ax = sns.stripplot(data=df, y="Decomposability Score", )

                    # --- choose exactly three base_form values ---
                    selected = [
                        ("bite the bullet", "bite the bullet", 0.25),
                        ("on cloud nine", "on cloud nine", 0.25),
                        ("touch a nerve", "touch a nerve", 0.25),
                    ]

                    for label, base_form, x_offset in selected:
                        y = df.loc[df["base_form"] == base_form, "Decomposability Score"].iloc[0]

                        # optional: highlight the dot
                        # ax.scatter(0, y, s=70, color="black", zorder=10)

                        ax.annotate(
                            label,
                            xy=(0, y),               # where the dot is
                            xytext=(x_offset, y),    # where the label goes
                            arrowprops=dict(arrowstyle="->"),
                            va="center",
                            fontsize=9
                        )

                    plt.tight_layout()
                    plt.savefig(f"correlation_experiment/visualisation/plots/decomp_vis/{dataset}_{model.replace('/', '_')}_{layer}_{sim}_{agg}.pdf")
                    plt.close()