import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


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

    # project_dir = "."
    project_dir = "/Users/mmi/Documents/projects/idioms_decomposability/decomp_code/idioms_decomposability"

    sim_funcs = ["cos", "cka", "wasser"]
    agg_metrics = ["entropy", "gini", "max", "sum", "mean"]

    # datasets = ["impli", "liuhwa"]
    datasets = ["impli_layers"]

    models = [
        # "google-bert/bert-base-uncased",
        # "google-bert/bert-base-cased",
        # "google-bert/bert-large-uncased",
        # "google-bert/bert-large-cased",
        "answerdotai/ModernBERT-base",
        # "answerdotai/ModernBERT-large",
    ]

    # layer = "layer_23"
    layer = "layer_20"

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

                    print(df.base_form.to_list())

                    plt.figure(figsize=(6, 5))
                    ax = sns.stripplot(data=df, y="Decomposability Score")

                    # label, base_form, x_offset, y_offset
                    selected = [
                        ("water under the bridge", "water under the bridge", 0.25, 0.020),
                        ("bite the bullet", "bite the bullet", 0.25, -0.015),
                        ("round the clock", "round the clock", 0.25, 0.030),
                        ("on cloud nine", "on cloud nine", 0.25, -0.020),
                        ("put someone's heart and soul", "put someone's heart and soul", 0.25, 0.015),
                        ("in the saddle", "in the saddle", 0.25, -0.010),
                    ]

                    for label, base_form, x_offset, y_offset in selected:
                        match = df.loc[df["base_form"] == base_form, "Decomposability Score"]

                        if match.empty:
                            print(f"Skipping '{base_form}' because it was not found.")
                            continue

                        y = match.iloc[0]

                        ax.annotate(
                            label,
                            xy=(0, y),                          # dot location
                            xytext=(x_offset, y + y_offset),    # label location
                            arrowprops=dict(
                                arrowstyle="->",
                                lw=1,
                                connectionstyle="arc3,rad=0.15"
                            ),
                            va="center",
                            ha="left",
                            fontsize=10,
                            rotation=5,
                            rotation_mode="anchor"
                        )

                    plt.tight_layout()
                    output_path = Path(
                        f"examples_of_decomp/plots/decomp_vis/{model.replace('/', '_')}/{layer}_{sim}_{agg}.pdf"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    plt.savefig(output_path)
                    plt.close()