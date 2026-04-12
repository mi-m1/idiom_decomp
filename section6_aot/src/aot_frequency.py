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

# result_df = extract_layerwise_rows(FOLDER, LAYER)


# frequency_df = pd.read_csv("data/frequencies_infini/impli/frequencies_infini_impli_v4_olmo-2-1124-13b-instruct_llama.csv")  # your frequency data
# print(frequency_df["frequency"].head())


# print(result_df.columns)
# print(result_df.shape)

# print('type checks')
# print(result_df["phrase"].dtype, frequency_df["base_form"].dtype)
# print("phrase NaNs:", result_df["phrase"].isna().sum())
# print("base_form NaNs:", frequency_df["base_form"].isna().sum())


# 2) merge
def merge_scores_with_freq(scores_df, frequency_df):
    merged = scores_df.merge(
        frequency_df,
        left_on="sentence",
        right_on="premise",
        how="left"
    )
    return merged


# merged = merge_scores_with_freq(result_df, frequency_df)


# # merged.to_csv("aot/merged.csv", index=False)
# print(merged.columns)
# print(merged.shape)
# # print(f'freqwuency: {merged["frequency"].describe()}')

# 3) bin
def add_frequency_bins(df, col, n_bins=4,):
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

# merged = add_frequency_bins(merged, n_bins=4, col="frequency")
# print("----")
# print(merged.columns)
# print(merged.shape)
# print(merged[[ "phrase", "frequency", "frequency_bin"]].head())

# missing = merged[merged["frequency"].isna()]
# print("Missing freq rows:", len(missing))
# print(missing["phrase"].head(20).tolist())

def extract_step(checkpoint):
    return pd.to_numeric(
        pd.Series(checkpoint).str.extract(r'step(\d+)')[0],
        errors='coerce'
    ).iloc[0]    

# 4) plot
HUE_ORDER = ["Very low", "Low", "High", "Very High"]
PALETTE_4  = dict(zip(HUE_ORDER, sns.color_palette("colorblind", 4)))

def plot_learning_curves(df, score_col, x, title, model, layer, measure, type, for_paper=False):
    hue_order = HUE_ORDER

    plt.figure(figsize=(5, 3))
    sns.lineplot(data=df, x=x, y=score_col, hue="frequency_bin", marker="o", markersize=4, palette=PALETTE_4, hue_order=hue_order)
    plt.xlabel("Training Steps", fontsize=8)

    if for_paper:
        plt.ylim(0.875, 1.0)
    else:
        plt.ylim(0.8, 1.0)
    plt.ylabel(f"Cosine Similarity (S, S_g)", fontsize=8)
    plt.legend(title="Frequency Bins", fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True)
    # plt.show()
    plt.tight_layout()

    if for_paper:
        plt.savefig(f"aot/visualisation/for_paper_rebuttal/{type}_{model}_{layer}.pdf", dpi=300)
    else:
        plt.savefig(f"aot/visualisation/{type}/{model}/{score_col}/{model}_{layer}_{measure}.pdf", dpi=300)
    plt.close()



def run_analysis(
    *,
    model: str,
    layer: int,
    score_col: str,
    type: str,
    output_root: Path = Path("aot/output/impli/"),
    frequency_csv: Path = Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_olmo-2-1124-13b-instruct_llama.csv"),
    n_bins: int = 4,
    x_col: str = "checkpoint_step",
    checkpoint_col: str = "checkpoint",
    frequency_col: str = "frequency",
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
    print(f"result shape: {result_df.shape}")

    result_df.to_csv("aot/result_df_freq.csv", index=False)

    frequency_df = pd.read_csv(frequency_csv)
    print(f"frequency shape: {frequency_df.shape}")

    if debug:
        print("Scores shape:", result_df.shape)
        print("Scores columns:", list(result_df.columns))
        print("Freq shape:", frequency_df.shape)
        print("Freq columns:", list(frequency_df.columns))
        if frequency_col in frequency_df.columns:
            print("Frequency head:", frequency_df[frequency_col].head())

    merged = merge_scores_with_freq(result_df, frequency_df)
    merged.to_csv("aot/merged_freq.csv", index=False)
    print(f"merged shape: {merged.shape}")

    merged["_matched"] = merged[frequency_col].notna()
    print(merged["_matched"].value_counts())

    # bins
    if frequency_col not in merged.columns:
        raise KeyError(f"Expected '{frequency_col}' column after merge, but it wasn't found.")
    merged = add_frequency_bins(merged, n_bins=n_bins, col=frequency_col)

    # extract steps for x-axis
    if checkpoint_col not in merged.columns:
        raise KeyError(f"Expected '{checkpoint_col}' column in merged df.")
    merged[x_col] = merged[checkpoint_col].apply(extract_step)

    # sanity checks
    if score_col not in merged.columns:
        raise KeyError(f"score_col='{score_col}' not found. Available columns: {list(merged.columns)}")

    if debug:
        missing = merged[merged[frequency_col].isna()]
        print("Merged shape:", merged.shape)
        print("Missing freq rows:", len(missing))
        if len(missing):
            print("Missing examples:", missing.get("phrase", pd.Series(dtype=str)).head(20).tolist())

    plot_learning_curves(
        merged,
        score_col=score_col,
        x=x_col,
        title=f"{model} | layer {layer} | {score_col}",
        model=model,
        layer=layer,
        measure=score_col,
        type=type,
        for_paper=for_paper,
    )

    return merged


if __name__ == "__main__":

    layers = list(range(0, 33))

    for layer in tqdm(layers):  

        merged = run_analysis(
            model="OLMO-2-1124-7B",
            layer=layer,
            score_col="cosine_sentence_paraphrase",
            type="frequency" # as opposed to "surprisal", "decomp"
            # frequency_csv=Path("...")  # swap this per model if needed
        )

        merged_3 = run_analysis(
            model="Olmo-3-1025-7B",
            layer=layer,
            score_col="cosine_sentence_paraphrase",
            type="frequency", # as opposed to "surprisal", "decomp"
            frequency_csv=Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_dolma-v1_7_llama.csv"),
        )

    layer = 13

    merged = run_analysis(
        model="OLMO-2-1124-7B",
        layer=layer,
        score_col="cosine_sentence_paraphrase",
        type="frequency", # as opposed to "surprisal", "decomp"
        # frequency_csv=Path("...")  # swap this per model if needed
        for_paper=True,
    )

    merged_3 = run_analysis(
        model="Olmo-3-1025-7B",
        layer=layer,
        score_col="cosine_sentence_paraphrase",
        type="frequency", # as opposed to "surprisal", "decomp"
        frequency_csv=Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_dolma-v1_7_llama.csv"),
        for_paper=True,
    )
