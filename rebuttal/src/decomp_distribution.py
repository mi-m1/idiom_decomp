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
        "google-bert/bert-large-uncased",
        # "google-bert/bert-large-cased",
        # "answerdotai/ModernBERT-base",
        # "answerdotai/ModernBERT-large",
    ]

    layer = "layer_23"
    # layer = "layer_20"

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

                    y_min = df["Decomposability Score"].min()
                    y_max = df["Decomposability Score"].max()
                    y_range = y_max - y_min

                    # (label, base_form, label_y, colour)
                    selected = [
                        ("in the saddle",                "in the saddle",                y_min + 0.00 * y_range, "#e41a1c"),
                        ("bite the bullet",              "bite the bullet",              y_min + 0.18 * y_range, "#ff7f00"),
                        ("on cloud nine",                "on cloud nine",                y_min + 0.28 * y_range, "#4daf4a"),
                        ("water under the bridge",       "water under the bridge",       y_min + 0.38 * y_range, "#377eb8"),
                        ("round the clock",              "round the clock",              y_min + 0.50 * y_range, "#984ea3"),
                        ("put someone's heart and soul", "put someone's heart and soul", y_min + 1.00 * y_range, "#a65628"),
                    ]

                    highlighted = {base_form for _, base_form, _, _ in selected}
                    df["_highlight"] = df["base_form"].isin(highlighted)

                    # background dots in grey
                    ax = sns.stripplot(
                        data=df[~df["_highlight"]],
                        y="Decomposability Score",
                        color="#bbbbbb",
                        size=4,
                        jitter=True,
                    )
                    # highlighted dots coloured individually, on top
                    for _, base_form, _, colour in selected:
                        subset = df.loc[df["base_form"] == base_form]
                        if subset.empty:
                            continue
                        ax.scatter(
                            x=[0] * len(subset),
                            y=subset["Decomposability Score"],
                            color=colour,
                            s=40,
                            zorder=3,
                        )

                    x_dot = 0.0   # centre of the strip (data coords)

                    for label, base_form, label_y, colour in selected:
                        match = df.loc[df["base_form"] == base_form, "Decomposability Score"]

                        if match.empty:
                            print(f"Skipping '{base_form}' because it was not found.")
                            continue

                        dot_y = match.iloc[0]

                        ax.annotate(
                            label,
                            xy=(x_dot, dot_y),          # dot: data coords
                            xycoords="data",
                            xytext=(0.62, label_y),      # label: x in axes fraction, y in data
                            textcoords=ax.get_yaxis_transform(),
                            arrowprops=dict(
                                arrowstyle="->",
                                color=colour,
                                lw=1.2,
                                connectionstyle="arc3,rad=0.2"
                            ),
                            va="center",
                            ha="left",
                            fontsize=9,
                            color=colour,
                        )

                    plt.tight_layout()
                    output_path = Path(
                        f"examples_of_decomp/plots/decomp_vis/{model.replace('/', '_')}/{layer}_{sim}_{agg}.pdf"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    plt.savefig(output_path)
                    plt.close()