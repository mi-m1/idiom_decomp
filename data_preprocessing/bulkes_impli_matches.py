import pandas as pd

bulkes_df = pd.read_csv("data/bulkes/13428_2016_747_MOESM1_ESM 2/GLOBAL DECOMPOSABILITY-Table 1.csv")

impli_df = pd.read_csv("data/processed/checked_manual_e_w_cql.csv")


impli_idioms = set(impli_df["base_form"].unique())
bulkes_idioms = set(bulkes_df["idiom"].unique())

print(impli_idioms)
print(bulkes_idioms)

import re

def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_overlaps(list_a, list_b):
    A = {norm(x) for x in list_a}
    B = {norm(x) for x in list_b}
    inter = A & B
    return len(inter), sorted(inter)

# usage:
n, overlaps = exact_overlaps(impli_idioms, bulkes_idioms)
print(f"----- Exact Overlaps -----")
print(f"Number of exact overlaps: {n}")
print(f"Overlapping idioms:{overlaps}")


import re

def tokens(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return set(s.split())

def jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0

def token_overlaps_sorted(list_a, list_b, threshold=0.0):
    matches = []
    for a in list_a:
        for b in list_b:
            score = jaccard(a, b)
            if score >= threshold:
                matches.append((a, b, score))

    # sort by highest Jaccard first
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches

# usage:
matches = token_overlaps_sorted(impli_idioms, bulkes_idioms, threshold=0.4)
print(f"----- Token Overlaps (Jaccard >= 0.4) -----")
print(f"Number of token overlaps: {len(matches)}")
print(f"Overlapping idioms (Impli, Bulkes, Jaccard):")
for m in matches:
    print(m)

import csv
with open("data/processed/bulkes_jaccard_0.4.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["impli", "bulkes", "jaccard"])  # header
    for a, b, score in matches:
        writer.writerow([a, b, score])