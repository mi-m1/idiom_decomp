# pip install pandas numpy matplotlib seaborn statsmodels linearmodels pingouin scikit-learn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

MODELS = ["OLMO-2-1124-7B", "Olmo-3-1025-7B"]


df_olmo2 = pd.read_csv(f"data/processed/{MODELS[0]}_lmm.csv")
df_olmo3 = pd.read_csv(f"data/processed/{MODELS[1]}_lmm.csv")

print(df_olmo2.shape, df_olmo3.shape)
df = pd.concat([df_olmo2, df_olmo3])

print("Combined df shape:", df.shape)
# Basic sanity
required = ["model", "checkpoint", "steps", "score",
            "frequency", "surprisal", "decomp",]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.sort_values(["model", "steps"]).reset_index(drop=True)

# Optional: log transforms if freq is heavy-tailed
df["log_frequency"] = np.log1p(df["frequency"])

# Standardize continuous predictors (helps interactions)
cols_to_z = ["log_frequency", "surprisal", "decomp", "steps"]
scaler = StandardScaler()
df[[c + "_z" for c in cols_to_z]] = scaler.fit_transform(df[cols_to_z])

# Plot layer-wise learning curves
for layer in df["layer"].unique():
    g = df[df["layer"] == layer]
    plt.plot(g["steps"], g["score"], alpha=0.1)
plt.xscale("log")
plt.xlabel("Steps")
plt.ylabel("Score")
plt.title("Layer-wise score trajectories")
# plt.show()

# formula = """
# score ~ steps_z
#       + C(layer)
#       + log_frequency_z + decomp_z + surprisal_z
#       + steps_z:log_frequency_z
#       + steps_z:decomp_z
#       + steps_z:surprisal_z
#       + C(layer):steps_z
# """

# Main interaction model
formula = """
score ~ steps_z
      + C(layer)
      + C(model)
      + log_frequency_z
      + surprisal_z
      + decomp_z
      + steps_z:log_frequency_z
      + steps_z:surprisal_z
      + steps_z:decomp_z
      """

    # + steps_z:log_frequency_z * surprisal_z * decomp_z
m = smf.ols(formula, data=df).fit(cov_type="HC3")
print(m.summary())

vars_ = ["log_frequency", "surprisal", "decomp"]
corr = df[vars_].corr(method="pearson")
print(corr.round(3))


# import numpy as np
# import matplotlib.pyplot as plt

# steps = np.linspace(df["steps_z"].min(), df["steps_z"].max(), 100)

# def pred(freq=0, surprisal=0, decomp=0):
#     return (
#         m.params["Intercept"]
#         + m.params["steps_z"] * steps
#         + m.params["steps_z:log_frequency_z"] * steps * freq
#         + m.params["steps_z:surprisal_z"] * steps * surprisal
#         + m.params["steps_z:decomp_z"] * steps * decomp
#     )

# plt.plot(np.exp(steps), pred(freq=-1), label="Low frequency")
# plt.plot(np.exp(steps), pred(freq=+1), label="High frequency")
# plt.xscale("log")
# plt.xlabel("Training steps")
# plt.ylabel("Predicted score")
# plt.legend()
# plt.title("Frequency moderates learning over time")
# plt.show()
