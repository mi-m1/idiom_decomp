import pandas as pd
import pingouin as pg
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_partial_correlations(df, cols):
    pcorr = df[cols].pcorr()
    print("partial correlation matrix: \n", pcorr)


def compute_vif(df, cols):
    X = df[cols]
    X = sm.add_constant(X)  # adding a constant for VIF calculation
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("VIF values: \n", vif_data)


if __name__ == "__main__":
    df_model = pd.read_csv("mixed_effect_analysis/impli_mixed_effect_data.csv")
    df_model = df_model[df_model["model"]=="google-bert_bert-large-cased"]
    df_human = pd.read_csv("data/human/bulkes_tanner_data_subset.csv")


    cols = ["predictability_score", "log_frequency", "decomp_score"]

    print("\n\n\nModel data partial correlations:\n")
    compute_partial_correlations(df_model, cols)
    print("\n\n\nModel data VIF values:\n")
    compute_vif(df_model, cols)

    print("\n\n\nHuman data partial correlations:\n")
    compute_partial_correlations(df_human, cols)
    print("\n\n\nHuman data VIF values:\n")
    compute_vif(df_human, cols)
