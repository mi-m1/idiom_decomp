import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
import statsmodels.formula.api as smf




def extract_step(checkpoint):
    return pd.to_numeric(
        pd.Series(checkpoint).str.extract(r'step(\d+)')[0],
        errors='coerce'
    ).iloc[0]  


def load_decomp_data(file):
    df = pd.read_csv(file)
    return df


def list_checkpoints(checkpoints_dir: Path) -> set[str]:
    return {p.stem for p in checkpoints_dir.glob("*.csv")}

def load_csvs(folder: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in folder.glob("*.csv")]
    print(f"len(dfs): {len(dfs)}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()



def build_big_df(
    model: str,
    score_folder: Path,
    surprisal_folder: Path,
    frequency_file: Path,
    decomp_file: str | Path,
    # adjust these if your real column names differ:
    score_col: str = "score",
    surprisal_col: str = "surprisal",
    freq_col: str = "freq",
    decomp_col: str = "decomp",
) -> pd.DataFrame:

    # 1) checkpoints to keep (your "use 100 checkpoints to match surprisal")
    checkpoints_dir = surprisal_folder
    keep_checkpoints = list_checkpoints(checkpoints_dir)
    print(len(keep_checkpoints), "checkpoints found for", model)

    # 2) load all score rows (ALL layers), but only from those checkpoints
    score_df = load_csvs(score_folder)
    if score_df.empty:
        raise ValueError(f"No score CSVs found in {score_folder}")
    print(score_df.shape, "score_df before filtering checkpoints")
    score_df = score_df[score_df["checkpoint"].isin(keep_checkpoints)].copy()
    print(score_df.shape, "score_df after filtering checkpoints")

    # 3) load surprisal (ALL layers)
    surprisal_df = load_csvs(surprisal_folder)
    if surprisal_df.empty:
        raise ValueError(f"No surprisal CSVs found in {surprisal_folder}")
    print(surprisal_df.shape, "surprisal_df shape")

    # 4) load frequency
    frequency_df = pd.read_csv(frequency_file)
    print(frequency_df.shape, "frequency_df shape")

    # 5) load decomp
    decomp_df = load_decomp_data(decomp_file)
    print(decomp_df.shape, "decomp_df shape")

    # --- Merge keys ---
    # Most common: checkpoint + idiom_id + layer
    # Add 'steps' if it exists in BOTH and is reliable.
    # merge_keys = ["checkpoint", "idiom_id", "layer"]
    # if "steps" in score_df.columns and "steps" in surprisal_df.columns:
    #     merge_keys = ["checkpoint", "steps", "idiom_id", "layer"]

    # 6) merge scoer + surprisal
    big_df = score_df.merge(
        surprisal_df,
        on=["checkpoint", "sentence",],
        how="inner",  # drop unmatched checkpoints and rows
    )

    print(big_df.shape, "after merging score + surprisal")
    print(big_df.columns)

    # 7) merge frequency
    big_df = big_df.merge(
        frequency_df,
        left_on="sentence",
        right_on='premise',
        how="inner",
    )
    print(big_df.shape, "after merging frequency")
    print(big_df.columns)

    # 8) merge decomp
    big_df = big_df.merge(
        decomp_df,
        left_on=["sentence"],
        right_on=["premise"],
        # on=["base_form", "hypothesis"],
        how="inner",
    )

    print(big_df.shape, "after merging decomp")
    print(big_df.columns)
    
    # 9) ensure required columns exist + add model
    big_df["model"] = model
    big_df["steps"] = big_df["checkpoint"].apply(extract_step)
    big_df.rename(
        columns={
            score_col: "score",
            surprisal_col: "surprisal",
            freq_col: "frequency",
            decomp_col: "decomp",
            'base_form_x': 'base_form',
            "premise_x": "premise",
            "extracted_idiom_x": "extracted_idiom",
            "hypothesis_x": "hypothesis",
        },
        inplace=True,
    )


    # 10) select exactly the columns you asked for (only keep ones that exist)
    wanted = ["model", "checkpoint", "steps", "layer", "base_form", "extracted_idiom", "score", "surprisal", "frequency", "decomp"]
    missing = [c for c in wanted if c not in big_df.columns]
    if missing:
        print("Warning: missing columns in final df:", missing)

    big_df = big_df[[c for c in wanted if c in big_df.columns]].copy()
    print(big_df.shape, "final big_df shape")
    print(big_df.columns)
    return big_df

if __name__ == "__main__":
    MODEL = "OLMO-2-1124-7B"
    SCORE_FOLDER = Path(f"aot/output/impli/{MODEL}/")
    SURPRISAL_FOLDER = Path(f"aot/output/impli_surprisal/{MODEL}/")

    if "OLMO-2" in MODEL:
        frequency_file = Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_olmo-2-1124-13b-instruct_llama.csv")
    else:
        frequency_file=Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_dolma-v1_7_llama.csv")

    decomp_file= "decomp_measure/scores/impli_layers/google-bert_bert-large-uncased/2025-12-25_02:34:13/layer_23/impli_wasser_sum_bert-large-uncased.csv"
    
    big_df = build_big_df(
        model=MODEL,
        score_folder=SCORE_FOLDER,
        surprisal_folder=SURPRISAL_FOLDER,
        frequency_file=frequency_file,
        decomp_file=decomp_file,
        score_col="cosine_sentence_paraphrase",
        freq_col="frequency",
        decomp_col="decomp_score",
        surprisal_col="phrase_surprisal_mean",
    )

    big_df.to_csv(f"data/processed/{MODEL}_lmm.csv", index=False)

    MODEL = "Olmo-3-1025-7B"
    SCORE_FOLDER = Path(f"aot/output/impli/{MODEL}/")
    SURPRISAL_FOLDER = Path(f"aot/output/impli_surprisal/{MODEL}/")

    if "OLMO-2" in MODEL:
        frequency_file = Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_olmo-2-1124-13b-instruct_llama.csv")
    else:
        frequency_file=Path("data/frequencies_infini/impli/frequencies_infini_impli_v4_dolma-v1_7_llama.csv")

    decomp_file= "decomp_measure/scores/impli_layers/google-bert_bert-large-uncased/2025-12-25_02:34:13/layer_23/impli_wasser_sum_bert-large-uncased.csv"
    
    big_df = build_big_df(
        model=MODEL,
        score_folder=SCORE_FOLDER,
        surprisal_folder=SURPRISAL_FOLDER,
        frequency_file=frequency_file,
        decomp_file=decomp_file,
        score_col="cosine_sentence_paraphrase",
        freq_col="frequency",
        decomp_col="decomp_score",
        surprisal_col="phrase_surprisal_mean",
    )

    big_df.to_csv(f"data/processed/{MODEL}_lmm.csv", index=False)


