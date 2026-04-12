import pandas as pd

gold = pd.read_csv("data/impli30.csv")

gpt_nano = pd.read_csv("data_processed/gpt-5-nano-2025-08-07_impli30.csv")

gpt5 = pd.read_csv("data_processed/gpt-5.1-2025-11-13_impli30.csv")

gpt_nano["cleaned"] = gpt_nano["extracted_idiom"].str.strip().str.strip("'")
gpt5["cleaned"] = gpt5["extracted_idiom"].str.strip().str.strip("'")

print(gpt_nano.head())

print(gpt_nano["cleaned"].equals(gold["idiom_extracted"]))
print(gpt5["cleaned"].equals(gold["idiom_extracted"]))  

print(set(gpt_nano['cleaned']) == set(gold['idiom_extracted']))

print(set(gpt5['cleaned']) == set(gold['idiom_extracted']))



print(gpt_nano[gpt_nano['cleaned'] != gold['idiom_extracted']])
print(gpt5[gpt5['cleaned'] != gold['idiom_extracted']])


# gpt 5 is better!