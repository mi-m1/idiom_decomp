import pandas as pd
from scipy.stats import spearmanr
from collections import Counter
import spacy
from ranked_correlations import find_score_file

# df_decomp = pd.read_csv("decomp_measure/scores/impli/google-bert_bert-base-cased/2025-12-22_18:50:31/impli_cka_gini_bert-base-cased.csv")
# df_syntax = pd.read_csv("data/total_entropy.csv")

# df = df_syntax.merge(df_decomp, on="premise")

# base_forms = df["base_form_x"].to_list()

nlp = spacy.load("en_core_web_sm")

# Keep signatures stable by collapsing variants
POS_KEEP = {"NOUN","PROPN","PRON","VERB","AUX","ADJ","ADV","ADP","DET","PART","CCONJ","SCONJ","NUM","INTJ"}
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
        # If it looks like an imperative or has a subject, treat as VP/clause
        has_subj = any(ch.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"} for ch in root.children)
        return "S" if has_subj else "VP"

    # Noun phrase
    if root.pos_ in {"NOUN", "PROPN", "PRON"}:
        return "NP"

    # Prepositional phrase often has ADP as root in fragments ("in a nutshell")
    if root.pos_ == "ADP":
        return "PP"

    # Adjective/adverb phrases
    if root.pos_ == "ADJ":
        return "ADJP"
    if root.pos_ == "ADV":
        return "ADVP"

    return f"OTHER({root.pos_})"

def parse_and_tally(idioms):
    rows = []
    coarse_counts = Counter()
    sig_counts = Counter()
    pair_counts = Counter()

    for text in idioms:
        doc = nlp(text)
        cshape = coarse_shape(doc)
        sig = pos_signature(doc)

        coarse_counts[cshape] += 1
        sig_counts[sig] += 1
        pair_counts[(cshape, sig)] += 1

        rows.append((text, cshape, sig))

    return rows, coarse_counts, sig_counts, pair_counts

# rows, coarse_counts, sig_counts, pair_counts = parse_and_tally(base_forms)

# print("Coarse shapes:")
# for k,v in coarse_counts.most_common():
#     print(f"{k:10s} {v}")

# print("\nTop POS signatures:")
# for k,v in sig_counts.most_common(10):
#     print(f"{v:3d}  {k}")

# print("\nExamples:")
# for r in rows:
#     print(r)


def coarse_shape_from_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "EMPTY"
    doc = nlp(text)
    return coarse_shape(doc)

# df["coarse_shape"] = df["base_form_x"].apply(coarse_shape_from_text)

def spearman_by_group(df, group_col, x_col, y_col, min_n=3):
    rows = []

    for shape, g in df.groupby(group_col):
        g = g[[x_col, y_col]].dropna()

        n = len(g)
        if n < min_n:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(g[x_col], g[y_col])

        rows.append({
            group_col: shape,
            "n": n,
            "spearman_rho": rho,
            "p_value": p
        })

    return pd.DataFrame(rows)

def combine_df(path_to_decomp, path_to_syntax, column_to_merge):

    df_syntax = pd.read_csv(path_to_syntax)
    df_decomp = pd.read_csv(path_to_decomp)
    df = df_syntax.merge(df_decomp, on=column_to_merge)

    return df


if __name__ == "__main__":

    project_dir = "."

    sim_funcs = ["cos", "cka", "wasser"]
    agg_metrics = ["entropy", "gini"]

    datasets = ["impli", "liuhwa"]
    models = [
        "google-bert/bert-base-uncased",
        "google-bert/bert-base-cased",
        "google-bert/bert-large-uncased",
        "google-bert/bert-large-cased",
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ]

    for dataset in datasets:
        rows = []

        for model in models:
            for sim in sim_funcs:
                for agg in agg_metrics:

                    csv_path = find_score_file(
                        project_dir,
                        dataset,
                        model,
                        sim,
                        agg,
                    )

                    df = combine_df(csv_path, "data/total_entropy.csv", column_to_merge="premise")
                    # print(df.columns)

                    df["coarse_shape"] = df["base_form_x"].apply(coarse_shape_from_text)

                    spearman_df = spearman_by_group(
                    df,
                    group_col="coarse_shape",
                    x_col="entropy_full",
                    y_col="decomp_score",
                    min_n=5
                    )

                    spearman_df = spearman_df.assign(
                        dataset=dataset,
                        model=model,
                        sim_func=sim,
                        agg_metric=agg,
                    )

                    rows.extend(spearman_df.to_dict(orient="records"))

        df = pd.DataFrame(rows)
        df.to_csv(f"correlation_experiment/binned_{dataset}_correlation_results.csv", index=False)

        break
