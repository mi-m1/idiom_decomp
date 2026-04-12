import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCORE_COL = "cosine_sentence_phrase"   # your score column

# ---------- 1) Read checkpoint CSVs ----------
def parse_step_tokens(path):
    """
    Extract step and tokens from filenames like:
      stage1-step150-tokens1B.csv
      stage1-step600-tokens3B.csv
    tokens will be converted to a numeric count (e.g., 1B -> 1e9).
    """
    name = os.path.basename(path)
    m = re.search(r"step(\d+)-tokens(\d+(?:\.\d+)?)([KMBT])", name, re.IGNORECASE)
    if not m:
        return np.nan, np.nan

    step = int(m.group(1))
    val = float(m.group(2))
    unit = m.group(3).upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
    tokens = val * mult
    return step, tokens

def load_checkpoints(folder, score_col=SCORE_COL):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)

        # Keep only what we need
        keep = ["phrase", score_col]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing columns: {missing}")

        step, tokens = parse_step_tokens(fp)
        out = df[keep].copy()
        out["layer"] = df["layer"]
        out["checkpoint_file"] = os.path.basename(fp)
        out["step"] = step
        out["tokens"] = tokens
        dfs.append(out)

    all_scores = pd.concat(dfs, ignore_index=True)
    return all_scores

# Example:
scores_df = load_checkpoints("aot/output/OLMo-2-1124-7B")


# ---------- 2) Merge with frequencies ----------
def merge_scores_with_freq(scores_df, frequency_df):
    # Make sure types are clean
    freq = frequency_df.copy()
    freq["frequency"] = pd.to_numeric(freq["frequency"], errors="coerce")

    merged = scores_df.merge(
        freq[["base_form", "frequency"]],
        left_on="phrase",
        right_on="base_form",
        how="left"
    )

    # Quick sanity check: if everything is NaN, the merge keys didn't match
    if merged["frequency"].isna().all():
        # helpful diagnostics
        scores_set = set(scores_df["phrase"].dropna().astype(str).str.strip().str.lower().unique())
        freq_set = set(freq["base_form"].dropna().astype(str).str.strip().str.lower().unique())
        overlap = len(scores_set.intersection(freq_set))
        raise ValueError(
            "After merge, all frequencies are NaN. Likely your idiom strings don't match.\n"
            f"Overlap (casefold+strip) between scores phrases and frequency base_form: {overlap}\n"
            "Consider normalizing both columns (strip, lowercase, etc.) before merging."
        )

    return merged


# ---------- 3) Safe frequency binning ----------
def add_frequency_bins(df, n_bins=4, col="frequency"):
    out = df.copy()

    # Work only with non-null frequencies for binning
    s = pd.to_numeric(out[col], errors="coerce")

    # If too few unique values, qcut can't make bins
    uniq = s.dropna().nunique()
    if uniq < 2:
        out["freq_bin"] = "unknown"
        return out

    # qcut with duplicates dropped (prevents "Bin edges must be unique")
    # We bin only non-null values, then align back
    binned = pd.qcut(s.dropna(), q=min(n_bins, uniq), duplicates="drop")

    # If duplicates="drop" collapsed bins too far (e.g. 1 bin), fallback
    if getattr(binned.dtype, "categories", None) is None or len(binned.dtype.categories) < 2:
        # fallback: use rank-based bins
        r = s.rank(method="average", na_option="keep")
        binned = pd.qcut(r.dropna(), q=min(n_bins, r.dropna().nunique()), duplicates="drop")

    out.loc[s.notna(), "freq_bin"] = binned.astype(str)
    out.loc[s.isna(), "freq_bin"] = "unknown"
    return out


# ---------- 4) Plot learning curves by frequency bin ----------
def plot_learning_curves(merged, score_col=SCORE_COL, x="tokens"):
    df = merged.copy()
    df = df.dropna(subset=[score_col])  # keep rows with a score

    # aggregate mean score per checkpoint and freq bin
    agg = (
        df.groupby([x, "freq_bin"], as_index=False)[score_col]
          .agg(mean="mean", n="count", std="std")
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))

    # sort x for nice lines
    agg = agg.sort_values([x, "freq_bin"])

    fig, ax = plt.subplots(figsize=(10, 6))

    # stable bin order
    bin_order = sorted(agg["freq_bin"].unique(), key=lambda t: (t == "unknown", t))

    for b in bin_order:
        sub = agg[agg["freq_bin"] == b]
        ax.plot(sub[x], sub["mean"], marker="o", label=b)
        ax.fill_between(sub[x], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=0.2)

    ax.set_xlabel(x)
    ax.set_ylabel(f"Mean {score_col}")
    ax.set_title(f"Idiom score over checkpoints by frequency bin")
    ax.legend(title="Frequency bin", fontsize=9)
    ax.grid(True, alpha=0.3)

    # If using tokens, log-scale often reads better
    if x == "tokens":
        ax.set_xscale("log")

    plt.tight_layout()
    plt.show()


# ---------- Putting it together ----------

frequency_df = pd.read_csv("data/frequencies_infini/impli/frequencies_infini_impli_v4_olmo-2-1124-13b-instruct_llama.csv")  # your frequency data
# 1) read scores
scores_df = load_checkpoints("aot/output/OLMo-2-1124-7B", score_col=SCORE_COL)

print(scores_df.columns)
print(scores_df.shape)

scores_df_layer = scores_df[scores_df["layer"] == 2]

print(scores_df_layer.columns)
print(scores_df_layer.shape)


# 2) merge
merged = merge_scores_with_freq(scores_df_layer, frequency_df)

# # 3) bin
merged = add_frequency_bins(merged, n_bins=4, col="frequency")

# # 4) plot
plot_learning_curves(merged, score_col=SCORE_COL, x="tokens")  # or x="step"
