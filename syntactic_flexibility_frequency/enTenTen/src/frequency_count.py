import numpy as np
import pandas as pd
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from trial_data import DATASET
import json
import os
from tqdm import tqdm
from time import sleep
# import spacy
import pandas as pd
# import pyinflect

tqdm.pandas()

# USERNAME = 'YOUR_USERNAME'
# API_KEY = '12345'
base_url = 'https://api.sketchengine.eu/bonito/run.cgi'


def shard_data(data_path, n_shards, save_dir):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=['base_form'])
    shards = np.array_split(df, n_shards)

    for i, shard in enumerate(shards):
        shard.to_csv(f"{save_dir}/shard_{i:01d}.csv", index=False)


def safe_get(url, params, auth, max_retries=3, timeout=300):
    """
    timeout = (connect_timeout, read_timeout)
    """
    for i in range(max_retries):
        try:
            r = requests.get(
                url,
                params=params,
                auth=auth,
                timeout=timeout
            )
            sleep(0.3)
            
        
            if r.status_code == 429:
                wait = 5 * (i + 1)
                print(f"429 rate limit → sleeping {wait}s")
                sleep(wait)
                continue

            # non-200 response
            if r.status_code != 200:
                print(f"HTTP {r.status_code} for query {params.get('q')}")
                return {}

            # empty response
            if not r.text.strip():
                print(f"Empty response for query {params.get('q')}")
                return {}

            return r.json()

        except requests.exceptions.ConnectTimeout:
            print(f"[ConnectTimeout] retry {i+1}/{max_retries}")
            sleep(5 * (i + 1))  # backoff

        except requests.exceptions.ReadTimeout:
            print(f"[ReadTimeout] retry {i+1}/{max_retries}")
            sleep(5 * (i + 1))

        except requests.exceptions.SSLError:
            print(f"[SSLError] retry {i+1}/{max_retries}")
            sleep(2 ** i)

        except requests.exceptions.RequestException as e:
            print(f"[RequestException] {e}")
            break

    print("Failed after retries")
    return {}


def get_freq_from_cql(cql, username, api_key):
    print(f"getting frequency for query {cql}")
    
    if pd.isna(cql) or cql is None:
        return pd.Series([0, 0])

    r = safe_get(
        base_url + '/concordance',
        params={
            'corpname': "preloaded/ententen21_tt31",
            'format': 'json',
            'q': cql,
            'default_attr': 'lemma',
            'asyn': 0,
            'pagesize': 0,
        },
        auth=(username, api_key)
    )

    try:
        full_size = r["fullsize"]
        rel_size = r["relsize"]
    
    except Exception:
        print(f"an error in query {cql}")
        full_size = None
        rel_size = None
    print(f"fullsize: {full_size}, relsize: {rel_size}")
    return pd.Series([full_size, rel_size])



def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset, e.g., impli")
    parser.add_argument("--shard_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to directory where the decomp scores will be saved.")
    parser.add_argument("--n_shards", type=int, required=True, help="number of shards")
    parser.add_argument("--total_data", type=str, required=True, help="path to the unsharded dataset")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = process_args()
    shard_path = args.shard_path
    shard_id = shard_path.split("_")[-1].replace(".csv", "")
    total_data = args.total_data
    os.makedirs(args.save_dir, exist_ok=True)
    # df.to_csv(f"{args.save_dir}/frequencies_shard_{shard_id}.csv", index=False)
    username = args.username
    api_key = args.api_key

    if not os.path.exists(shard_path):
        shard_data(total_data, n_shards=args.n_shards, save_dir="/".join(shard_path.split("/")[:-1]))

    df = pd.read_csv(shard_path)

    print(f"Processing shard {shard_id}, {len(df)} rows")


    CQL_COLUMNS = {
        "identity_cql": ("identity_full", "identity_rel"),
        "passive_cql":  ("passive_full",  "passive_rel"),
        "adj_insertion_cql_1":   ("adj_insertion_full_1",   "adj_insertion_rel_1"),
        "adj_insertion_cql_2":   ("adj_insertion_full_2",   "adj_insertion_rel_2"),
        "adv_insertion_cql":   ("adv_insertion_full",   "adv_insertion_rel"),
        "adv_insertion_cql_adj":   ("adv_insertion_full_adj",   "adv_insertion_rel_adj"),
        "nominalization_cql":   ("nominalization_full",   "nominalization_rel")
    }

    df_unique = df

    for _, (full_col, rel_col) in CQL_COLUMNS.items():
        if full_col not in df_unique:
            df_unique[full_col] = pd.NA
        if rel_col not in df_unique:
            df_unique[rel_col] = pd.NA


    # OUT_FILE = "data/df_unique_with_freqs.csv"

    for i, row in tqdm(df_unique.iterrows(), total=len(df_unique)):

        row_updated = False

        for cql_col, (full_col, rel_col) in CQL_COLUMNS.items():
            
            print(cql_col)

            # skip if already processed
            if pd.notna(row[full_col]):
                continue

            cql = row[cql_col]

            try:
                full, rel = get_freq_from_cql(cql, username, api_key)
                df_unique.at[i, full_col] = full
                df_unique.at[i, rel_col] = rel
                row_updated = True
                sleep(0.3)

            except Exception as e:
                print(f"❌ Error at row {i}, column {cql_col}: {e}")
                # df_unique.to_csv(OUT_FILE, index=False)
                df_unique.to_csv(f"{args.save_dir}/frequencies_shard_{shard_id}.csv", index=False)
                raise

        # save only if something changed
        if row_updated:
            # df_unique.to_csv(OUT_FILE, index=False)
            df.to_csv(f"{args.save_dir}/frequencies_shard_{shard_id}.csv", index=False)
