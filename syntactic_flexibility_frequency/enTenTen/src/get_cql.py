import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from trial_data import DATASET
import json
import os
from tqdm import tqdm
from time import sleep
import spacy
import pandas as pd
import pyinflect

tqdm.pandas()

USERNAME = 'YOUR_USERNAME'
API_KEY = 'API_KEY'
base_url = 'https://api.sketchengine.eu/bonito/run.cgi'

def process_impli(row):

    processed = row.replace("’s ", "'s")
    processed = row.replace("(one)", "someone")
    processed = row.replace(" one", "someone")
    processed = row.replace(" one's", "someone's")

    return processed

# def lemmatize(row):
#     """
#     param row: string
#     output: list of tokens
#     """
#     doc = nlp(row)
#     tokens = [tok for tok in doc]
#     return tokens

def lemmatize(row):
    """
    param row: string
    output: list of lemmatized tokens
    """
    doc = nlp(row)
    lemma = [tok.lemma_ for tok in doc]
    row_lemma = " ".join(lemma)
    doc_lemma = nlp(row_lemma)
    tokens_lemma = [tok for tok in doc_lemma]
    # pos_tags = [(tok, tok.pos_, tok.dep_) for tok in tokens_lemma] # TODO: should i keep the pos tags of the unlemmatized idiom?? (got the well of ...)
    return tokens_lemma


def tokens_to_cql(tokens):
    """
    Build a CQL query from tokens, replacing
    the "somebody + 's" sequence with a POS-based pattern.
    
    Args:
        tokens: list of tokens
    Returns:
        CQL string
    """
    cql = "q"
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Detect the special "somebody + 's" sequence
        if i + 1 < len(tokens):
            next_tok = tokens[i+1]
            if (str(tok) == "someone" and str(next_tok) == "'s") or (str(tok) == "something" and str(next_tok) == "'s"):
                # Replace with POS-based pattern
                cql += '[tag="PP.?|N.*|DT|RP"]{1,2}'
                i += 2  # skip both tokens
                continue
        if str(tok) == "someone" or str(tok) == "something":
            cql += '[tag="PP.?|N.*|DT|RP"]{1,2}' 
            i += 1  # skip one token
            continue
        if str(tok) == "-":
            i += 1  # skip one token
            continue
        if str(tok) == "adj":
            cql += '[tag="J.*"]' # TODO: {1,2}
            i += 1 
            continue
        if str(tok) == "adv":
            cql += '[tag="RB.?"]' # TODO: {1,2}
            i += 1 
            continue
        if str(tok) == "'s" or str(tok) == "’s":
            i += 1 
            continue
        # if str(tok) == "a":
        #     cql += f'[lemma="a" | lemma="the"]'
        #     i += 1 
        #     continue
        
        # Default: add lemma-based token
        cql += f'[lemma="{str(tok)}"]'
        i += 1
    return cql


def get_identity_cql(row):
    """
    param row: list of tokens
    output: cql string
    """
    cql = tokens_to_cql(row)
    return cql


def get_passive_cql(row): 

    verbal_idiom = any(tok.pos_=="VERB" for tok in row)
    object = any(tok.dep_ == "dobj" for tok in row)
    if not verbal_idiom or not object:
        return None

    verb = next(tok for tok in row if tok.dep_ == "ROOT" and tok.pos_ == "VERB")
    dobj = next(tok for tok in row if tok.dep_ == "dobj")
    preps = [tok for tok in row if tok.dep_ == "prep"]
    adverbs = [tok for tok in row if tok.dep_ == "advmod"]
    # past_participle = verb._.inflect("VBN")

    passive_tokens = []
    for tok in dobj.subtree:
        passive_tokens.append(tok)
    
    passive_tokens.append("be")
    passive_tokens.append(verb)
    # for prep in preps:
    #     for tok in prep.subtree:
    #         passive_tokens.append(tok)
    
    for adv in adverbs:
        passive_tokens.append(adv)
    
    for tok in row:
        if tok not in passive_tokens:
            passive_tokens.append(tok)

    cql = tokens_to_cql(passive_tokens)
    return cql


def get_adj_insertion_cql(row):
    nouns = [tok for tok in row if tok.pos_ == "NOUN"]
    noun_0_id = row.index(nouns[0])if len(nouns)>0 else None
    noun_1_id = row.index(nouns[1]) if len(nouns)>1 else None

    tokens_0 = row.copy()
    tokens_1 = row.copy()

    # insert adjective placeholder before first noun
    if noun_0_id is not None:
        tokens_0.insert(noun_0_id, "adj")

    # insert adjective placeholder before second noun
    if noun_1_id is not None:
        tokens_1.insert(noun_1_id, "adj")

    cql_0 = tokens_to_cql(tokens_0) if noun_0_id is not None else None
    cql_1 = tokens_to_cql(tokens_1) if noun_1_id is not None else None

    return cql_0, cql_1

def get_adv_insertion_cql(row): # TODO: also afer the idiom?

    # verb = [tok for tok in row if tok.dep_ == "ROOT" and tok.pos_ == "VERB"]
    adj =  [tok for tok in row if tok.pos_ == "ADJ"]
    # verb_id = row.index(verb[0])if len(verb)>0 else None
    adj_id = row.index(adj[0]) if (len(adj)>0 and row.index(adj[0])!=0) else None
    if (len(adj)>0 and row.index(adj[0])!=0):
        adj_id = row.index(adj[0])
    elif len(adj)>1:
        adj_id = row.index(adj[1])
    else:
        adj_id = None
          


    tokens_whole = row.copy()
    tokens_adj = row.copy()

    # insert adjective placeholder before first noun
    # if verb_id is not None:
    #     tokens_verb.insert(verb_id, "adv")
    tokens_whole.insert(0, "adv")

    # insert adjective placeholder before second noun
    if adj_id is not None:
        tokens_adj.insert(adj_id, "adv")

    cql_whole = tokens_to_cql(tokens_whole) 
    cql_adj = tokens_to_cql(tokens_adj) if (adj_id is not None and adj_id !=0) else None

    return cql_whole, cql_adj

def get_nominalization_cql(row):

    verb = [tok for tok in row if tok.dep_ == "ROOT" and tok.pos_ == "VERB"]
    if len(verb) < 1:
        return None
    row_string = [str(tok) for tok in row]
    row_string = " ".join(row_string)
    if row_string == "up to speed":
        return None
    elif row_string == "rat fink":
        return None

    verb_id = row.index(verb[0])
    if verb_id != 0:
        print("Warning: verb is not the first token in the idiom:", row_string, "retrurning None cql")
        return None

    cql = "q"
    cql += f'[lemma="{verb[0]}" & tag="N.*"]'
    idx = verb_id +1

    if row[idx].pos_ == "ADP":
        cql+= f'[lemma="{row[verb_id +1]}"]'
        idx = verb_id +2

    cql+= f'[lemma="of"]'

    if len(row[idx:])>0:
        cql_rest = tokens_to_cql(row[idx:])
        cql_rest = cql_rest[1:] # remove "q"
        cql += cql_rest

    return cql
    

def get_obj_movement_cql(row):
    pass

def safe_get(url, params, auth, max_retries=5):
    for i in range(max_retries):
        try:
            return requests.get(url, params=params, auth=auth, timeout=300).json()
        except requests.exceptions.SSLError:
            print(f"SSL error, retry {i+1}/{max_retries}...")
            sleep(2 ** i)
    raise RuntimeError("Failed after retries")


if __name__ == "__main__":
    
    nlp = spacy.load("en_core_web_sm")
    file_path = "data"
    filename = "liuhwa"
    df = pd.read_csv(f"{file_path}/{filename}.csv")
    # df = df.iloc[100:]
    # df["idiom_processed"] = df.apply(lambda x: process_impli(x["base_form"]), axis=1)
    df["base_form"] = df["base_form"].str.strip()
    df["idiom_pos"] = df.apply(lambda x: lemmatize(x["base_form"]), axis=1)
    df["identity_cql"] = df.apply(lambda x: get_identity_cql(x["idiom_pos"]), axis=1)
    df["passive_cql"] = df.apply(lambda x: get_passive_cql(x["idiom_pos"]), axis=1)
    df[["adj_insertion_cql_1", "adj_insertion_cql_2"]] = df.apply(
    lambda x: get_adj_insertion_cql(x["idiom_pos"]), axis=1, result_type="expand")
    df[["adv_insertion_cql", "adv_insertion_cql_adj"]] = df.apply(
    lambda x: get_adv_insertion_cql(x["idiom_pos"]), axis=1, result_type="expand")

    df["nominalization_cql"] = df.apply(lambda x: get_nominalization_cql(x["idiom_pos"]), axis=1)
    df.to_csv(f"{file_path}/{filename}_w_cql.csv")
