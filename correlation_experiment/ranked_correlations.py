
#Spearman’s rank correlation (ρ).

import pandas as pd
from scipy.stats import spearmanr



def run_correlation(path_to_decomp_scores, column_to_merge):
    # Load files
    df_syntax = pd.read_csv("data/total_entropy.csv")
    df_decomp = pd.read_csv(path_to_decomp_scores)

    # Merge if needed (example: common ID)
    df = df_syntax.merge(df_decomp, on=column_to_merge)

    # print(df["decomp_score"].tolist())

    # print(df["entropy_full"].tolist())

    # Compute Spearman correlation
    corr, p_value = spearmanr(df["entropy_full"], df["decomp_score"])

    print("Spearman correlation", corr)
    print("p-value:", p_value)

    dict_to_return = {
        "spearmanr": corr,
        "p_value": p_value,
    }

    return dict_to_return

from pathlib import Path

def find_score_file(project_dir, dataset, model, sim, agg):
    model_dir = model.replace("/", "_")

    base_path = (
        Path(project_dir)
        / "decomp_measure"
        / "scores"
        / dataset
        / model_dir
    )

    pattern = f"{dataset}_{sim}_{agg}_{model.split('/')[-1]}.csv"

    matches = list(base_path.glob(f"*/{pattern}"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for {pattern}: {matches}")

    return matches[0]


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
                    
                    result = run_correlation(
                        # f"decomp_measure/scores/{dataset}/{model.replace('/', '_')}/liuhwa_{sim}_{agg}_{model.replace('/', '_')}.csv",
                        csv_path,
                        column_to_merge="premise",
                    )

                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "sim_func": sim,
                        "agg_metric": agg,
                        **result,
                    })
        

        df = pd.DataFrame(rows)
        df.to_csv(f"correlation_experiment/{dataset}_correlation_results.csv", index=False)

        break

