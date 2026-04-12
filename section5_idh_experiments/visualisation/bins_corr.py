import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load results
df = pd.read_csv("correlation_experiment/binned_impli_correlation_results.csv")

#### clean model name (optional, reuse from before)
df["model_clean"] = df["model"].str.split("/").str[-1]

# filter out small groups
top_groups = (
    df.groupby("coarse_shape")["n"]
      .max()               # n is constant per group, but max is safe
      .sort_values(ascending=False)
      .head(3)
      .index
)

df_top3 = df[df["coarse_shape"].isin(top_groups)].copy()


sig = df_top3[df_top3["p_value"] < 0.05].copy()
# sig["abs_rho"] = sig["spearman_rho"].abs()
# sig = sig.sort_values("abs_rho", ascending=False)

print(sig.sort_values("spearman_rho", ascending=True))


print(sig.shape)

# # sort by correlation magnitude
# df = df.sort_values("spearman_rho")

# plt.figure(figsize=(6, 4))

# ax = sns.barplot(
#     data=df,
#     y="coarse_shape",
#     x="spearman_rho",
#     orient="h",
#     color="steelblue"
# )

# # vertical zero line
# ax.axvline(0, color="black", linewidth=1)

# # annotate n and significance
# for i, row in df.iterrows():
#     label = f"n={int(row['n'])}"
#     if row["p_value"] < 0.05:
#         label += " *"
#     ax.text(
#         row["spearman_rho"] + (0.01 if row["spearman_rho"] >= 0 else -0.01),
#         df.index.get_loc(i),
#         label,
#         va="center",
#         ha="left" if row["spearman_rho"] >= 0 else "right",
#         fontsize=9
#     )

# plt.xlabel("Spearman ρ")
# plt.ylabel("Constituent Type")
# plt.title("Group-wise Spearman Correlations (Impli)")

# plt.tight_layout()
# plt.savefig("groupwise_correlations.pdf", bbox_inches="tight")
# plt.show()
