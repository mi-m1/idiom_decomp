import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import spacy

# load once
nlp = spacy.load("en_core_web_sm")

POS_KEEP = {
    "NOUN", "PROPN", "PRON",
    "VERB", "AUX",
    "ADJ", "ADV",
    "ADP", "DET",
    "PART", "NUM",
    "CCONJ", "SCONJ"
}


def pos_signature(doc):
    sig = []
    for t in doc:
        if t.is_punct or t.is_space:
            continue
        p = t.pos_
        sig.append(p if p in POS_KEEP else "X")
    return " ".join(sig)


def coarse_shape(doc):
    sent = next(doc.sents)
    root = sent.root

    # Clause-like
    if root.pos_ in {"VERB", "AUX"}:
        has_subj = any(
            ch.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"}
            for ch in root.children
        )
        return "S" if has_subj else "VP"

    # Noun phrase
    if root.pos_ in {"NOUN", "PROPN", "PRON"}:
        return "NP"

    # Prepositional phrase
    if root.pos_ == "ADP":
        return "PP"

    # Adjective/adverb phrases
    if root.pos_ == "ADJ":
        return "ADJP"
    if root.pos_ == "ADV":
        return "ADVP"

    return f"OTHER({root.pos_})"


def tag_idiom_structure(text):
    doc = nlp(text)
    return pd.Series({
        "pos_signature": pos_signature(doc),
        "structure_type": coarse_shape(doc),
    })


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

    project_dir = "/Users/mmi/Documents/projects/idioms_decomposability/decomp_code/idioms_decomposability"

    sim_funcs = ["cos", "cka", "wasser"]
    agg_metrics = ["entropy", "gini", "max", "sum", "mean"]
    datasets = ["impli_layers"]

    models = [
        "google-bert/bert-large-uncased",
    ]

    layer = "layer_23"

    # fixed order + colours for consistency across plots
    shape_order = ["S", "VP", "NP", "PP", "ADJP", "ADVP"]
    shape_palette = {
        "S": "tab:blue",
        "VP": "tab:orange",
        "NP": "tab:green",
        "PP": "tab:red",
        "ADJP": "tab:purple",
        "ADVP": "tab:brown",
    }

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

                    # tag idioms from the idiom/base-form column
                    # if your idioms column is literally called "idioms", replace "base_form" below with "idioms"
                    structure_df = df["base_form"].apply(tag_idiom_structure)
                    df = pd.concat([df, structure_df], axis=1)

                    print(df[["base_form", "structure_type", "pos_signature"]].head())

                    # dummy x column so hue colours show clearly in one strip
                    df["_all"] = ""

                    plt.figure(figsize=(7, 5))
                    ax = sns.stripplot(
                        data=df,
                        x="_all",
                        y="Decomposability Score",
                        hue="structure_type",
                        hue_order=[s for s in shape_order if s in df["structure_type"].unique()],
                        palette=shape_palette,
                        dodge=False,
                        jitter=0.25,
                        size=6,
                        alpha=0.85,
                    )

                    ax.set_xlabel("")
                    ax.set_xticks([])

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
                            xy=(0, y),
                            xytext=(x_offset, y + y_offset),
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

                    ax.legend(title="Structure Type", bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()

                    output_path = Path(
                        f"examples_of_decomp/plots/decomp_vis/{model.replace('/', '_')}/{layer}_{sim}_{agg}.pdf"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    plt.savefig(output_path)
                    plt.close()