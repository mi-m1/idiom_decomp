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

def extract_layerwise_rows(folder, layer,):

    dfs = []

    print("CSV count:", len(list(Path(folder).glob("*.csv"))))

    for csv_file in folder.glob("*.csv"):
        # print(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)
        df_filtered = df[df["layer"] == layer]
        dfs.append(df_filtered)

    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

def load_surprisal_data(folder):
    dfs = []

    for csv_file in folder.glob("*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


# 2) merge
def merge_scores_with_surprisal(scores_df, surprisal_Df):
    merged = scores_df.merge(
        surprisal_Df,
        on=["checkpoint", "sentence"],
        # how="left"
        how="inner" # drop unmatched checkpoints and rows
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
def plot_learning_curves(df, score_col, x, title, model, layer, measure, type, surprisal_col, for_paper=False):
    PALETTE_4 = sns.color_palette("colorblind", 4)
    hue_order = ["Very low", "Low", "High", "Very High"]

    plt.figure(figsize=(5, 3))
    sns.lineplot(data=df, x=x, y=score_col, hue=f"{surprisal_col}_bin", hue_order=hue_order, marker="o", markersize=4, palette=PALETTE_4)
    plt.xlabel("Training Steps", fontsize=8)
    if for_paper:
        plt.ylim(0.875, 1.0)
    else:
        plt.ylim(0.8, 1.0)
    plt.ylabel(f"Cosine Similarity (S, S_g)", fontsize=8)
    plt.legend(title="Surprisal Bins", fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True)
    # plt.title(title)
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
    surprisal_col: str = "phrase_surprisal_mean",
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
    result_df = extract_layerwise_rows(folder, layer)
    print(f"result shape: {result_df.shape}")

    surprisal_folder = surprisal_dir / model
    surprisal_df = load_surprisal_data(surprisal_folder)

    print(f"surprisal shape: {surprisal_df.shape}")

    print("Checkpoint uniques:")
    print(len(result_df.checkpoint.unique()))
    print(len(surprisal_df.checkpoint.unique()))

    print("Differences in checkpoints")
    diff = set(result_df.checkpoint.unique()) - set(surprisal_df.checkpoint.unique())
    print(len(diff))
    
    merged = merge_scores_with_surprisal(
        result_df,
        surprisal_df,
    )

    print(merged.head())
    print(f"merged shape: {merged.shape}")
    print(merged.columns)

    merged["_matched"] = merged["phrase_surprisal_mean"].notna()
    print(merged["_matched"].value_counts())

    merged = add_bins(merged, n_bins=n_bins, col=surprisal_col)

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
        surprisal_col=surprisal_col,
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
            type="surprisal", # as opposed to "frequency", "decomp"
            # frequency_csv=Path("...")  # swap this per model if needed
            surprisal_col="phrase_surprisal_mean",
            checkpoint_col="checkpoint",
        )

        merged_3 = run_analysis(
            model="Olmo-3-1025-7B",
            layer=layer,
            score_col="cosine_sentence_paraphrase",
            type="surprisal", # as opposed to "frequency", "decomp"
            # frequency_csv=Path("...")  # swap this per model if needed
            surprisal_col="phrase_surprisal_mean",
            checkpoint_col="checkpoint",
        )

    layer = 13

    merged = run_analysis(
        model="OLMO-2-1124-7B",
        layer=layer,
        score_col="cosine_sentence_paraphrase",
        type="surprisal", # as opposed to "frequency", "decomp"
        # frequency_csv=Path("...")  # swap this per model if needed
        surprisal_col="phrase_surprisal_mean",
        checkpoint_col="checkpoint",
        for_paper=True,
    )

    merged_3 = run_analysis(
        model="Olmo-3-1025-7B",
        layer=layer,
        score_col="cosine_sentence_paraphrase",
        type="surprisal", # as opposed to "frequency", "decomp"
        # frequency_csv=Path("...")  # swap this per model if needed
        surprisal_col="phrase_surprisal_mean",
        checkpoint_col="checkpoint",
        for_paper=True,
    )
