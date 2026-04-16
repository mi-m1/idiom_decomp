import numpy as np
import math
import json
import os
import pandas as pd


def compute_entropy(probs, normalize=False):

    ent = - sum(p_i * math.log(p_i+ 1e-12) for p_i in probs)
    if normalize:
        ent = ent / math.log(len(probs))

    return ent


FILE_NAME = "/total_frequencies.csv"
CORPUS = FILE_NAME.split("_")[-2]
FILE_PATH = "data/frequencies"
SAVE_DIR = "syntax_frequency/frequency_results"


if __name__ == "__main__":

    df = pd.read_csv(f"{FILE_PATH}/{FILE_NAME}")

    frequencies_full = df[["adj_insertion_full_1", "adj_insertion_full_2", "adv_insertion_full", "adv_insertion_full_adj", "identity_full", "nominalization_full", "passive_full"]].to_numpy()
    frequencies_full = np.nan_to_num(frequencies_full, nan=0.0)

    sum_full = np.sum(frequencies_full, axis=1)
    sum_full = sum_full[:, None]
    prob_full = frequencies_full / (sum_full+1e-12)
    df["entropy_full"] = -np.sum(prob_full * np.log(prob_full + 1e-12), axis=1)


    frequencies_rel = df[["adj_insertion_rel_1", "adj_insertion_rel_2", "adv_insertion_rel", "adv_insertion_rel_adj", "identity_rel", "nominalization_rel", "passive_rel"]].to_numpy()
    frequencies_rel = np.nan_to_num(frequencies_rel, nan=0.0)

    sum_rel = np.sum(frequencies_rel, axis=1)
    sum_rel = sum_rel[:, None]
    prob_rel = frequencies_rel / (sum_rel + 1e-12)
    df["entropy_rel"] = -np.sum(prob_rel * np.log(prob_rel + 1e-12), axis=1)
    
    df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)

    df.to_csv(f"{SAVE_DIR}/total_entropy.csv")
    