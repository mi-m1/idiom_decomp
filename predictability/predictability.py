import pandas as pd
import numpy as np
import torch
import os
import huggingface_hub
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
from tqdm import tqdm
import ast
import math
import argparse
import re


tqdm.pandas()

def load_model_and_tokenizer(MODEL_NAME,*, decoder=False):

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if decoder:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,).to(DEVICE) # AutoModelForCausalLM
    else:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,).to(DEVICE)
    model.eval()

    return model, tokenizer, DEVICE



def encode(text, DEVICE):
    return tokenizer(text, return_tensors="pt").to(DEVICE)

def get_pooled_embedding(model_output, attention_mask):
    hidden = model_output.last_hidden_state  
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-6)
    return summed / count


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

    # print(f"spans tensor: {spans}")

    return spans[0].tolist()



def get_input_and_output_idiom_stimuli_dataset(row):
    idiom = row["Idiom"]
    last_word = idiom.split()[-1]
    sentence = row["Sentence"]
    input = sentence[: sentence.rfind(last_word)].strip()
    target = last_word
    return pd.Series([input, target])

def get_input_and_output(row, dataset_name):
    idiom = row["extracted_idiom"]
    last_word = idiom.split()[-1]
    if dataset_name == "impli":
        sentence = row["premise"]
    elif dataset_name == "liuhwa":
        sentence = row["sentence"]
    else:
        raise ValueError("dataset_name not recognized")
    # input = sentence[: sentence.rfind(last_word)].strip()
    # if return_mask:
        # pattern = rf"\b{re.escape(last_word)}\b"
        # sentence = re.sub(pattern, tokenizer.mask_token, sentence, flags=re.IGNORECASE)
    return pd.Series([sentence, last_word])




def get_next_token_prob_decoder(input, target, model, tokenizer, DEVICE):

    context_ids = tokenizer(input, return_tensors="pt").input_ids.to(DEVICE)
    target_ids = tokenizer(target, return_tensors="pt").input_ids.to(DEVICE)


    joint_prob = 1.0
    log_prob = 0.0
    input_ids = context_ids

    for token_id in target_ids[0]:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        next_token_logits = logits[0, -1]
        probs = torch.softmax(next_token_logits, dim=-1)
        token_prob = probs[token_id]
        joint_prob *= token_prob
        log_prob += torch.log(probs[token_id])

        # Append the current token to input_ids for next prediction
        input_ids = torch.cat([input_ids, torch.tensor([[token_id]], device=DEVICE)], dim=-1) # dim=1

        return pd.Series([joint_prob.item(), log_prob.item()], index=["prob", "log_prob"])



def get_next_token_prob_encoder(input, target, model, tokenizer, DEVICE, *, masked_string=True):

    context_ids = tokenizer(input, return_tensors="pt").input_ids.to(DEVICE)
    target_ids = tokenizer(target, return_tensors="pt").input_ids.to(DEVICE)
    context_ids = context_ids[:, 1:-1] # removing [CLS] and [SEP] tokens
    target_ids = target_ids[:, 1:-1] # removing [CLS] and [SEP] tokens


    if masked_string:
        pattern = rf"\b{re.escape(target[0])}\b"
        masked_input = re.sub(pattern, tokenizer.mask_token, input[0], flags=re.IGNORECASE)
        masked_input_ids = tokenizer(masked_input, return_tensors="pt").input_ids.to(DEVICE)
        masked_input_ids = masked_input_ids[:, 1:-1] # removing [CLS] and [SEP] tokens
        mask_token_index = torch.where(masked_input_ids == tokenizer.mask_token_id)[1]
        target_position = mask_token_index.tolist() # 1d list []
        target_ids = context_ids[0][mask_token_index].unsqueeze(0) # 2d tensor
    else:
        target_position = find_subtensor_span(context_ids, target_ids)
    
    print("target_position: ", target_position)


    context_tokens = tokenizer.convert_ids_to_tokens(context_ids[0])
    # print("context tokens: ", context_tokens)

    MASK_TOKEN = tokenizer.mask_token  
    joint_prob = 1.0
    log_prob = 0.0

    for pos, tok in zip(target_position, target_ids[0]):
        new_tokens = context_tokens.copy()
        new_tokens[pos] = MASK_TOKEN
        masked_text = tokenizer.convert_tokens_to_string(new_tokens)
        print(f"\tMasked text:{masked_text}")
        masked_enc = encode(masked_text,DEVICE)
        with torch.no_grad():
            masked_out = model(**masked_enc)
            logits = masked_out.logits
        mask_token_index = torch.where(masked_enc["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = logits[0, mask_token_index, :]
        probs = torch.softmax(mask_token_logits, dim=-1).squeeze(0)
        token_prob = probs[tok]
        print(f"\tToken prob for {tokenizer.convert_ids_to_tokens([tok])[0]}: {token_prob.item()}")
        joint_prob *= token_prob
        log_prob += torch.log(probs[tok])

    norm_log_prob = log_prob / len(target_position)
    print("total joint prob:", joint_prob.item())
    return pd.Series([joint_prob.item(), log_prob.item(), norm_log_prob.item()], index=["prob", "log_prob", "norm_log_prob"])

