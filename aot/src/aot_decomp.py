import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

MODEL = "OLMO-2-1124-7B"
LAYER = 2
FOLDER = Path(f"aot/output/{MODEL}/")

def extract_layerwise_rows(model, folder, layer,):

    #use 100 checkpoints to match surprisal
    checkpoints_dir = Path(f"aot/output/impli_surprisal/{model}/")

    checkpoints = []

    for csv_file in checkpoints_dir.glob("*.csv"):
        checkpoints.append(os.path.basename(csv_file))


    dfs = []

    for csv_file in folder.glob("*.csv"):
        if os.path.basename(csv_file) in checkpoints:
            df = pd.read_csv(csv_file)
            df_filtered = df[df["layer"] == layer]
            dfs.append(df_filtered)

    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

def load_decomp_data(file):
    # dfs = []

    # for csv_file in folder.glob("*.csv"):
    #     df = pd.read_csv(csv_file)
    #     dfs.append(df)

    # result_df = pd.concat(dfs, ignore_index=True)
    # return result_df

    df = pd.read_csv(file)

    return df


# 2) merge
def merge_scores_with_decomp(scores_df, decomp_df):
    merged = scores_df.merge(
        decomp_df,
        left_on="sentence",
        right_on="premise",
        how="left",
    )

    return merged

# 3) bin
def add_bins(df, col, n_bins=4,):
    out = df.copy()
    out[f"{col}_q_bin"] = pd.qcut(out[col], q=n_bins, labels=False, duplicates='drop')

    mapping = {
    0: "Very low",
    1: "Low",
    2: "High",
    3: "Very High"
    }

    out[f"{col}_bin"] = out[f"{col}_q_bin"].map(mapping)

    return out


def extract_step(checkpoint):
    return pd.to_numeric(
        pd.Series(checkpoint).str.extract(r'step(\d+)')[0],
        errors='coerce'
    ).iloc[0]    

# 4) plot
def plot_learning_curves(df, score_col, x, title, model, layer, measure, type, decomp_col, for_paper=False):
    PALETTE_4 = sns.color_palette("colorblind", 4)
    hue_order = ["Very low", "Low", "High", "Very High"]

    plt.figure(figsize=(5, 3))
    sns.lineplot(data=df, x=x, y=score_col, hue=f"{decomp_col}_bin", marker="o", markersize=4, hue_order=hue_order, palette=PALETTE_4)
    plt.xlabel("Training Steps", fontsize=8)

    if for_paper:
        plt.ylim(0.875, 1.0)
    else:
        plt.ylim(0.8, 1.0)
    plt.ylabel(f"Cosine Similarity (S, S_g)", fontsize=8)
    plt.legend(title="Decomp Bins", fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True)
    # plt.show()
    plt.tight_layout()
    if for_paper:
        plt.savefig(f"aot/visualisation/for_paper/{type}_{model}_{layer}.pdf", dpi=300)
    else:
        plt.savefig(f"aot/visualisation/{type}/{model}/{score_col}/{model}_{layer}_{score_col}.pdf", dpi=300)
    
    plt.close()


def run_analysis(
    *,
    model: str,
    layer: int,
    score_col: str,
    type: str,
    output_root: Path = Path("aot/output/impli/"),
    surprisal_dir: Path = Path("aot/output/impli_surprisal/"),
    n_bins: int = 4,
    x_col: str = "checkpoint_step",
    checkpoint_col: str = "checkpoint",
    decomp_col: str = "phrase_surprisal_mean",
    debug: bool = True,
    for_paper: bool = False,
) -> pd.DataFrame:
    """
    Runs the full pipeline:
      1) load and filter scores CSVs for a given layer
      2) load frequency CSV
      3) merge
      4) bin frequencies
      5) extract checkpoint step + plot

    Returns the final merged dataframe.
    """

    os.makedirs(f"aot/visualisation/{type}/{model}/{score_col}", exist_ok=True)

    folder = output_root / model
    result_df = extract_layerwise_rows(model, folder, layer)
    print(result_df.shape)
    print(result_df.head())

    decomp_file= "decomp_measure/scores/impli_layers/google-bert_bert-large-uncased/2025-12-25_02:34:13/layer_23/impli_wasser_sum_bert-large-uncased.csv"
    decomp_df = load_decomp_data(decomp_file)


    print(decomp_df.shape)
    print(decomp_df.head())
    
    merged = merge_scores_with_decomp(
        result_df,
        decomp_df,
    )

    print(merged.head())
    print(merged.shape)
    print(merged.columns)


    merged = add_bins(merged, n_bins=n_bins, col=decomp_col)


    merged.to_csv("aot/merged_decomp.csv", index=False)

    print(merged.head())
    print(f"merged shape: {merged.shape}")
    print(merged.columns)

    merged["_matched"] = merged["decomp_score"].notna()
    print(merged["_matched"].value_counts())

    merged[x_col] = merged[checkpoint_col].apply(extract_step)

    plot_learning_curves(
        merged,
        score_col=score_col,
        x=x_col,
        title=f"{model} | layer {layer} | {score_col}",
        model=model,
        layer=layer,
        measure=score_col,
        type=type,
        decomp_col=decomp_col,
        for_paper=for_paper,
    )


    return merged




if __name__ == "__main__":

    layers = list(range(0, 33))
    # layers = [32]

    for layer in tqdm(layers):  

        merged = run_analysis(
            model="OLMO-2-1124-7B",
            layer=layer,
            score_col="cosine_sentence_paraphrase",
            type="decomp", # as opposed to "frequency", "decomp"
            # frequency_csv=Path("...")  # swap this per model if needed
            decomp_col="decomp_score",
            checkpoint_col="checkpoint",
        )

        merged_3 = run_analysis(
            model="Olmo-3-1025-7B",
            layer=layer,
            score_col="cosine_sentence_paraphrase",
            type="decomp", # as opposed to "frequency", "decomp"
            # frequency_csv=Path("...")  # swap this per model if needed
            decomp_col="decomp_score",
            checkpoint_col="checkpoint",
        )

    layer = 13

    merged = run_analysis(
        model="OLMO-2-1124-7B",
        layer=layer,
        score_col="cosine_sentence_paraphrase",
        type="decomp", # as opposed to "frequency", "decomp"
        # frequency_csv=Path("...")  # swap this per model if needed
        decomp_col="decomp_score",
        checkpoint_col="checkpoint",
        for_paper=True,
    )

    merged_3 = run_analysis(
        model="Olmo-3-1025-7B",
        layer=layer,
        score_col="cosine_sentence_paraphrase",
        type="decomp", # as opposed to "frequency", "decomp"
        # frequency_csv=Path("...")  # swap this per model if needed
        decomp_col="decomp_score",
        checkpoint_col="checkpoint",
        for_paper=True,
    )


