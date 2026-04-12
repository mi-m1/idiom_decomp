import pandas as pd


# check extracted idioms is actually in the sentence

df = pd.read_csv('/Users/mmi/Documents/projects/idioms_decomposability/decomp_code/idioms_decomposability/data/processed/checked_baseFrom_gpt-5.1-2025-11-13_gpt-5.1-2025-11-13_manual_e.csv')

def check_extraction(sentence, extracted_idiom):

    if extracted_idiom in sentence:

        return "True"
    else:
        return "False"
    

# df["check_extraction"] = df.apply(lambda x: check_extraction(df["premise"], df["extracted_idiom"]))

# print(df["check_extraction"].head())


df["is_substring"] = df.apply(lambda r: r["extracted_idiom"] in r["premise"], axis=1)

print(df["is_substring"].value_counts())

print(df[df["is_substring"] == False])