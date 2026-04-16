import pandas as pd
import argparse
from statsmodels.othermod.betareg import BetaModel


def beta_regression(formula, data):
    # DV must be strictly in (0, 1)
    y = formula.split("~")[0].strip()
    eps = 1e-6
    data = data.copy()
    data[y] = data[y].clip(eps, 1 - eps)

    mod = BetaModel.from_formula(formula, data)
    res = mod.fit(maxiter=1000, disp=False)  # method defaults to 'bfgs'
    print(res.summary())
    return res



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="mixed_effect_analysis/impli_mixed_effect_data.csv", help="Path to the mixed effect data csv file")
    parser.add_argument("--independent", nargs="+", type=str, default=["predictability_score", "frequency", "structure"], help="Dependent variable")
    parser.add_argument("--random", nargs="+", type=str, help="Random effects")
    parser.add_argument("--output_dir", type=str, default="mixed_effect_analysis/results/", help="Directory to save the results")
    parser.add_argument("--per_lm", type=bool, help="Whether to save the trained model")

    return parser.parse_args()




if __name__ == "__main__":

    args = parse_args()
    data = args.data_path
    independent = args.independent
    random = args.random
    output_dir = args.output_dir
    
    print(

    f"  Independent variables: {independent}\n\n"
    f"  Random effects: {random}\n\n"
    f"  Dependent variable: decomposability\n\n"
)


    data = pd.read_csv(data)
    data = data.convert_dtypes()


    formula = "decomp_score ~ " + " * ".join(independent) # + " + " + " + ".join([f"(1|{r})" for r in random])
    if random:
        formula = formula + " + " + " + ".join([f"(1|{r})" for r in random])


    # fix structure variable
    if "structure" in independent:
        data = data[data["structure"].isin(["VP", "NP", "PP"])]
        data["structure"] = data["structure"].astype("category")

    if args.per_lm:
        for model_name, df_m in data.groupby("model"):
            print(f"\nAnalyzing model: {model_name}\n")
            m = beta_regression(formula, df_m)
            print(f"\nfinished analysis for model {model_name}n")

    else:   
        m = beta_regression(formula, data)


    print(f"\nfinished analysis\n")
