import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ast
import argparse
    
tqdm.pandas()

def load_model_and_tokenizer(MODEL_NAME,):

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    return model, tokenizer, DEVICE


def decode(token_ids, tokenizer):
  return tokenizer.decode(token_ids)


def encode(text, DEVICE):
    return tokenizer(text, return_tensors="pt").to(DEVICE)

def find_subtensor_span(a, b):
    a = a.squeeze(0)
    b = b.squeeze(0)

    L = a.size(0)
    M = b.size(0)

    windows = a.unfold(0, M, 1)          # (L-M+1, M)
    matches = (windows == b).all(dim=1)  # (L-M+1,)

    starts = matches.nonzero(as_tuple=True)[0]  # (K,)

    offsets = torch.arange(M, device=a.device)  # (M,)
    spans = starts[:, None] + offsets[None, :]  # (K, M)

    print(f"spans tensor: {spans}")

    return spans[0].tolist()


def test_quality_of_extraction(span_ls, sent_enc):

  extracted_idiom_ids = sent_enc[0, span_ls]
  extracted_idiom = tokenizer.decode(extracted_idiom_ids)
#   print(f"extracted_idiom: {extracted_idiom}")

  return extracted_idiom


def run_extraction_check(sentence, idiom, DEVICE):

    print(f"idiom to extract: {idiom}")
    print(f"sentence: {sentence}")
    
    sent_enc = encode(sentence,DEVICE)["input_ids"]
    idiom_token_ids = encode(idiom,DEVICE)["input_ids"]
    idiom_token_ids = idiom_token_ids[:, 1:-1]  # Remove special tokens
    
    span_ls = find_subtensor_span(sent_enc, idiom_token_ids)

    print(f"\tFound span positions: {span_ls}")
    
    extracted_idiom = test_quality_of_extraction(span_ls, sent_enc)


    if extracted_idiom.strip() == idiom.strip():
        print("Extraction successful!")
        return "y"
    else:
        print(f"Check this extraction! Expected:&{idiom}&, but got:&{extracted_idiom}&")
        return "n"
    

if __name__ == "__main__":

    def process_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, required=True, help="Bidirectional model")
        parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset, e.g., impli")
        parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file.")
        parser.add_argument("--sim_func", type=str, required=True, help="Similarity function to use to calculate embedding proxity, i.e., cosine similarity (cos), centred kernel alignment (cka)")
        parser.add_argument("--save_dir", type=str, required=True, help="Path to directory where the decomp scores will be saved.")

        args = parser.parse_args()

        return args
    
    args = process_args()

    # MODEL_NAME = "bert-base-uncased"
    # MODEL_NAME = "answerdotai/ModernBERT-base"
    # MODEL_NAME = "answerdotai/ModernBERT-large"
    model, tokenizer, DEVICE = load_model_and_tokenizer(args.model_name)


    #impli - subset30
    if args.dataset_name == "impli30":

        # df = pd.read_csv("data/idiom_subset_manual_e.tsv", sep="\t")
        df = pd.read_csv(args.dataset_path,)
        print(df.head())

        df["idiom_processed"] = ' ' + df["idiom_extracted"]
        print(df.head())

        df["search_quality"] = df.apply(lambda row: run_extraction_check(row["premise"], row["idiom_processed"], DEVICE), axis=1)

        # df["idiom_processed"] = df.apply(lambda row: process_impli(row), axis=1)

        # df["token_scores"] = df.progress_apply(lambda row: get_scores(row["premise"], row["hypothesis"], row["idiom_processed"], model, tokenizer, DEVICE), axis=1)
        # df["decomp_score"] = df.progress_apply(lambda row: get_decomp_score(row["token_scores"]), axis=1)

        # df["sim_function"] = args.sim_func
        # print(df.head)

        # df.to_csv(f"{args.save_dir}/{args.model_name.split('/')[-1]}.csv")


    





