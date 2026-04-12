import pandas as pd
# import ot
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ast
import numpy as np
import math
from scipy.stats import wasserstein_distance
from testing_search import *
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

def get_layer_hidden(out, layer: int):
    """
    out: model output with hidden_states
    layer: int
      0 = embeddings
      1..N = transformer layers
      -1 = last layer (same as out.last_hidden_state)
    """
    if out.hidden_states is None:
        raise ValueError("hidden_states is None. Pass output_hidden_states=True.")
    return out.hidden_states[layer]  # [B, S, D]





def encode(text,tokenizer, DEVICE):
    return tokenizer(text, return_tensors="pt").to(DEVICE)

def get_pooled_embedding_from_layer(out, attention_mask, layer: int = -1):
    hidden = get_layer_hidden(out, layer)              # [B,S,D]
    mask = attention_mask.unsqueeze(-1).float()        # [B,S,1]
    summed = (hidden * mask).sum(dim=1)                # [B,D]
    count = mask.sum(dim=1).clamp(min=1e-6)            # [B,1]
    return summed / count                              # [B,D]

def get_token_embeddings_from_layer(out, attention_mask, layer: int = -1):
    hidden = get_layer_hidden(out, layer)[0]           # [S,D]
    mask = attention_mask[0].bool()                    # [S]
    return hidden[mask]                                # [n_tokens, D]

def cosine_similarity(a, b):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (a_norm * b_norm).sum(dim=-1)


def cka_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-12,
    min_samples: int = 2,
) -> torch.Tensor:
    """
    Linear CKA for token-level embeddings.
    - X: [n_tokens_x, d]
    - Y: [n_tokens_y, d]

    If there are fewer than `min_samples` tokens on either side,
    fall back to cosine similarity between mean-pooled embeddings.
    """
    if X.dim() == 1:
        X = X.unsqueeze(0)
    if Y.dim() == 1:
        Y = Y.unsqueeze(0)

    # match number of tokens
    n = min(X.size(0), Y.size(0))

    # not enough tokens for meaningful CKA → fallback
    if n < min_samples:
        X_pool = X.mean(dim=0, keepdim=True)  # [1, d]
        Y_pool = Y.mean(dim=0, keepdim=True)  # [1, d]
        # reuse your cosine_similarity(a, b) -> shape [1]
        return cosine_similarity(X_pool, Y_pool)[0]

    X = X[:n]
    Y = Y[:n]

    # standard linear CKA
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtY = X.T @ Y
    num = (XtY * XtY).sum()

    XtX = X.T @ X
    YtY = Y.T @ Y
    den = torch.sqrt((XtX * XtX).sum() + eps) * torch.sqrt((YtY * YtY).sum() + eps)

    return num / (den + eps)



def _wasserstein_1d_sorted(u: torch.Tensor, v: torch.Tensor, p: int = 2) -> torch.Tensor:
    u = u.flatten()
    v = v.flatten()
    u, _ = torch.sort(u)
    v, _ = torch.sort(v)
    m = min(u.numel(), v.numel())
    u = u[:m]
    v = v[:m]
    diff = torch.abs(u - v)
    if p == 1:
        return diff.mean()
    return (diff.pow(p).mean()).pow(1.0 / p)


def wasserstein_similarity(
    A: torch.Tensor,
    B: torch.Tensor,
    num_projections: int = 50,
    p: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Sliced Wasserstein similarity for token-level embeddings.
    A: [n, d], B: [m, d]
    Returns similarity scalar in (0,1] via 1/(1+dist).
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("Wasserstein expects token-level tensors of shape [n_tokens, hidden_dim].")

    # match feature dims
    d = min(A.size(1), B.size(1))
    A = A[:, :d]
    B = B[:, :d]

    # if too few tokens, fall back to 1D Wasserstein over flattened features
    if A.size(0) < 2 or B.size(0) < 2:
        dist = _wasserstein_1d_sorted(A.reshape(-1), B.reshape(-1), p=p)
        return 1.0 / (1.0 + dist + eps)

    device = A.device
    proj = torch.randn(num_projections, d, device=device)
    proj = proj / (proj.norm(dim=1, keepdim=True) + eps)

    Ap = (A @ proj.T).T  # [k, n]
    Bp = (B @ proj.T).T  # [k, m]

    dists = []
    for k in range(num_projections):
        dists.append(_wasserstein_1d_sorted(Ap[k], Bp[k], p=p))
    dist = torch.stack(dists).mean()

    return 1.0 / (1.0 + dist + eps)


def cka_similarity(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Linear CKA for token-level embeddings.
    X: [n, d], Y: [m, d]
    Returns scalar torch tensor.
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("CKA expects token-level tensors of shape [n_tokens, hidden_dim].")

    # match number of samples (tokens)
    n = min(X.size(0), Y.size(0))
    X = X[:n]
    Y = Y[:n]

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtY = X.T @ Y
    num = (XtY * XtY).sum()

    XtX = X.T @ X
    YtY = Y.T @ Y
    den = torch.sqrt((XtX * XtX).sum() + eps) * torch.sqrt((YtY * YtY).sum() + eps)

    return num / (den + eps)


def gini_coefficient(values, eps: float = 1e-12):
    """
    Gini coefficient for a list of nonnegative values.
    Returns G in [0, 1] (approximately), where 0 = perfectly uniform, 1 = maximally concentrated.
    """
    x = [max(float(v), 0.0) for v in values]
    n = len(x)
    if n == 0:
        return 0.0

    s = sum(x)
    if s <= eps:
        return 0.0

    x_sorted = sorted(x)
    # G = (2 * sum_{i=1..n} i*x_i) / (n*sum x) - (n+1)/n
    weighted_sum = 0.0
    for i, xi in enumerate(x_sorted, start=1):
        weighted_sum += i * xi

    g = (2.0 * weighted_sum) / (n * s + eps) - (n + 1.0) / n
    # Clamp for numerical safety
    return max(0.0, min(1.0, g))

def gini_dispersion(values, eps: float = 1e-12):
    """
    Gini dispersion (unnormalized).
    Works for any nonnegative values.
    """
    x = torch.tensor([max(float(v), 0.0) for v in values])
    n = x.numel()

    if n == 0:
        return 0.0

    s = x.sum()
    if s <= eps:
        return 0.0

    x_sorted, _ = torch.sort(x)

    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    g = (2.0 * (idx * x_sorted).sum()) / (n * s) - (n + 1.0) / n

    return g.item()


def get_scores(sentence, figurative_gloss, idiom_extracted,
               model, tokenizer, DEVICE, sim_func, layer: int = -1):

    sent_enc = encode(sentence, tokenizer, DEVICE)
    sent_output = model(**sent_enc, output_hidden_states=True, return_dict=True)

    gloss_enc = encode(figurative_gloss, tokenizer, DEVICE)
    gloss_output = model(**gloss_enc, output_hidden_states=True, return_dict=True)

    if sim_func == "cos":
        sent_repr  = get_pooled_embedding_from_layer(sent_output,  sent_enc["attention_mask"],  layer=layer)
        gloss_repr = get_pooled_embedding_from_layer(gloss_output, gloss_enc["attention_mask"], layer=layer)
        original_score = cosine_similarity(sent_repr, gloss_repr)[0]

    elif sim_func == "cka":
        sent_repr  = get_token_embeddings_from_layer(sent_output,  sent_enc["attention_mask"],  layer=layer)
        gloss_repr = get_token_embeddings_from_layer(gloss_output, gloss_enc["attention_mask"], layer=layer)
        original_score = cka_similarity(sent_repr, gloss_repr)

    elif sim_func == "wasser":
        sent_repr  = get_token_embeddings_from_layer(sent_output,  sent_enc["attention_mask"],  layer=layer)
        gloss_repr = get_token_embeddings_from_layer(gloss_output, gloss_enc["attention_mask"], layer=layer)
        original_score = wasserstein_similarity(sent_repr, gloss_repr)

    else:
        raise ValueError(f"Unknown similarity function: {sim_func}")

    tokens = tokenizer.convert_ids_to_tokens(sent_enc["input_ids"][0])
    print("\nTokens:", tokens)

    # Find idiom token span (your existing logic)
    sent_token_ids = sent_enc["input_ids"]

    idiom_token_ids = encode(idiom_extracted, tokenizer, DEVICE)["input_ids"]
    idiom_token_ids = idiom_token_ids[:, 1:-1]  # remove [CLS]/[SEP]
    idiom_positions = find_subtensor_span(sent_token_ids, idiom_token_ids)

    MASK_TOKEN = tokenizer.mask_token

    token_scores = []
    for idx in idiom_positions:
        new_tokens = tokens.copy()
        new_tokens[idx] = MASK_TOKEN
        masked_text = tokenizer.convert_tokens_to_string(new_tokens)
        print(f"\tMasked text:{masked_text}")

        masked_enc = encode(masked_text, tokenizer, DEVICE)
        masked_out = model(**masked_enc, output_hidden_states=True, return_dict=True)

        if sim_func == "cos":
            masked_repr = get_pooled_embedding_from_layer(masked_out, masked_enc["attention_mask"], layer=layer)
            masked_score = cosine_similarity(masked_repr, gloss_repr)[0]
        elif sim_func == "cka":
            masked_repr = get_token_embeddings_from_layer(masked_out, masked_enc["attention_mask"], layer=layer)
            masked_score = cka_similarity(masked_repr, gloss_repr)
        elif sim_func == "wasser":
            masked_repr = get_token_embeddings_from_layer(masked_out, masked_enc["attention_mask"], layer=layer)
            masked_score = wasserstein_similarity(masked_repr, gloss_repr)

        else:
            raise ValueError(f"Unknown similarity function: {sim_func}")

        delta = (original_score - masked_score).item()
        token_scores.append((idx, tokens[idx], delta))

    print("\nToken-level importance (score drop Δ_j):")
    for idx, tok, delta in token_scores:
        print(f"Token {idx:2d}: {tok:15s} Δ = {delta:.4f}")

    print("\nSorted by importance:")
    for idx, tok, delta in sorted(token_scores, key=lambda x: -x[2]):
        print(f"{tok:15s}  Δ = {delta:.4f}")

    return token_scores


#TODO: Neeed to check this and choose polarity
# def get_decomp_score(token_scores, normalise=False, metric="entropy", eps: float = 1e-12):
#     """
#     metric:
#       - "entropy": your current behavior (higher => more spread)
#       - "gini": returns 1 - Gini (higher => more spread)
#       - "gini_raw": returns Gini itself (higher => more concentrated)
#     """
#     deltas = [abs(delta) for (_, _, delta) in token_scores]
#     P = [max(d, 0.0) for d in deltas]
#     total_P = sum(P)

#     if total_P <= eps:
#         score = 0.0
#         print(f"\nIdiomatic decomposability score (metric={metric}, normalise={normalise}): {score:.3f}")
#         return score

#     w = [p / (total_P + eps) for p in P]
#     k = len(w)

#     if metric == "entropy":
#         H = -sum(wj * math.log(wj + eps) for wj in w)
#         score = H / math.log(k + eps) if normalise else H

#     elif metric == "gini":
#         # Convert inequality -> evenness
#         G = gini_coefficient(w, eps=eps)
#         score = 1.0 - G
#         # # Optional normalisation to keep [0,1] comparable across token counts
#         # # For a probability vector, max G is (k-1)/k, so 1-G min is 1/k.
#         # if normalise and k > 1:
#         #     G_max = (k - 1.0) / k
#         #     score = (score - (1.0 / k)) / ((1.0 - (1.0 / k)) + eps)  # map [1/k, 1] -> [0,1]
#         #     score = max(0.0, min(1.0, score))

#     elif metric == "gini_raw":
#         score = gini_coefficient(w, eps=eps)
#         # if normalise and k > 1:
#         #     # map [0, (k-1)/k] -> [0,1]
#         #     score = score / (((k - 1.0) / k) + eps)
#         #     score = max(0.0, min(1.0, score))

#     else:
#         raise ValueError(f"Unknown metric: {metric}. Use 'entropy', 'gini', or 'gini_raw'.")

#     print(f"\nIdiomatic decomposability score (metric={metric}, normalise={normalise}): {score:.3f}")
#     return score

def get_decomp_score(token_scores, normalise=False, metric="entropy", eps: float = 1e-12):
    """
    metric:
      - "entropy": your current behavior (higher => more spread)
      - "gini": returns 1 - Gini (higher => more spread)
      - "gini_raw": returns Gini itself (higher => more concentrated)
    """
    deltas = [abs(delta) for (_, _, delta) in token_scores]
    P = [max(d, 0.0) for d in deltas]
    total_P = sum(P)

    if total_P <= eps:
        score = 0.0
        print(f"\nIdiomatic decomposability score (metric={metric}, normalise={normalise}): {score:.3f}")
        return score

    w = [p / (total_P + eps) for p in P]
    k = len(w)

    if metric == "entropy":
        H = -sum(wj * math.log(wj + eps) for wj in w)
        score = H / math.log(k + eps) if normalise else H

    elif metric == "gini":
        deltas = [abs(delta) for (_, _, delta) in token_scores]
        G = gini_dispersion(deltas)
        k = len(deltas)
        score = (k - 1) / k - G   # distance from maximal concentration
    elif metric == "mean":
        score = np.mean(deltas)
        return score
    elif metric == "sum":
        score = np.sum(deltas)
        return score
    elif metric == "max":
        score = np.max(deltas)
        return score
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'entropy', 'gini', or '' .")

    print(f"\nIdiomatic decomposability score (metric={metric}, normalise={normalise}): {score:.3f}")
    return score


def process_impli(row):

    raw_idiom_string = row["idiom"]
    #rules based replacements
    processed = raw_idiom_string.replace("sb", "somebody").split()
    return processed



if __name__ == "__main__":

    def process_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, required=True, help="Bidirectional model")
        parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset, e.g., impli")
        parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file.")
        parser.add_argument("--layer", type=int, default=-1, help="Layer from which to extract embeddings. -1 for last layer.")
        parser.add_argument("--sim_func", type=str, required=True, help="Similarity function to use to calculate embedding proxity, i.e., cosine similarity (cos), centred kernel alignment (cka) and Wasserstein distance (wasser)")
        parser.add_argument("--agg_metric", type=str, default="entropy", help="Aggregation metric for decomposability: 'entropy', 'gini', or 'gini_raw'")
        parser.add_argument(
            "--drop_cql_cols",
            action="store_true",
            help="Drop columns whose names contain 'cql'"
        )
        parser.add_argument("--testing", action="store_true", help="Testing mode with a small subset of data.")
        parser.add_argument("--save_dir", type=str, required=True, help="Path to directory where the decomp scores will be saved.")
                
        args = parser.parse_args()

        return args
    
    args = process_args()

    # MODEL_NAME = "bert-base-uncased"
    # MODEL_NAME = "answerdotai/ModernBERT-base"
    # MODEL_NAME = "answerdotai/ModernBERT-large"
    model, tokenizer, DEVICE = load_model_and_tokenizer(args.model_name)


    df = pd.read_csv(args.dataset_path,)

    if args.testing:
        df = df.head(5)
        
    df["idiom_processed"] = ' ' + df["extracted_idiom"]

    if args.dataset_name == "impli":
        print(df.head())

        df["token_scores"] = df.progress_apply(lambda row: get_scores(row["premise"], row["hypothesis"], row["idiom_processed"], model, tokenizer, DEVICE, args.sim_func, layer=args.layer), axis=1)
        df["decomp_score"] = df.apply(lambda row: get_decomp_score(row["token_scores"], normalise=False, metric=args.agg_metric), axis=1)

        df["sim_function"] = args.sim_func
        print(df.head)
    
    elif args.dataset_name == "liuhwa":

        df["token_scores"] = df.progress_apply(lambda row: get_scores(row["sentence"], row["paraphrase"], row["idiom_processed"], model, tokenizer, DEVICE, args.sim_func), axis=1)
        df["decomp_score"] = df.apply(lambda row: get_decomp_score(row["token_scores"], normalise=False, metric=args.agg_metric), axis=1)

        df["sim_function"] = args.sim_func
        print(df.head)

    if args.drop_cql_cols:
        df = df.loc[:, ~df.columns.str.contains("cql", case=False)]
    else:
        pass

    df.to_csv(f"{args.save_dir}/{args.dataset_name}_{args.sim_func}_{args.agg_metric}_{args.model_name.split('/')[-1]}.csv", index=False)



    # #impli - subset30
    # if args.dataset_name == "impli30":
    #     # df = pd.read_csv("data/idiom_subset_manual_e.tsv", sep="\t")
    #     df = pd.read_csv(args.dataset_path,)
    #     # df["idiom_processed"] = df['idiom_extracted'].str.split().apply(lambda lst: [' '] + lst)
    #     df["idiom_processed"] = ' ' + df["idiom_extracted"]
    #     print(df.head())



    #     df["token_scores"] = df.progress_apply(lambda row: get_scores(row["premise"], row["hypothesis"], row["idiom_processed"], model, tokenizer, DEVICE, args.sim_func), axis=1)
    #     df["decomp_score"] = df.apply(lambda row: get_decomp_score(row["token_scores"],normalise=False, metric=args.agg_metric), axis=1)

    #     df["sim_function"] = args.sim_func
    #     print(df.head)

    #     df.to_csv(f"{args.save_dir}/{get_time_now()}_{args.dataset_name}_{args.sim_func}_{args.agg_metric}_{args.model_name.split('/')[-1]}.csv")


    # #impli - full
    # elif args.dataset_name == "impli":
    #     df = pd.read_csv(args.dataset_path,)
    #     # df["idiom_processed"] = df['idiom_extracted'].str.split().apply(lambda lst: [' '] + lst)
    #     df["idiom_processed"] = ' ' + df["idiom_extracted"]
    #     print(df.head())

    #     df["token_scores"] = df.progress_apply(lambda row: get_scores(row["premise"], row["hypothesis"], row["idiom_processed"], model, tokenizer, DEVICE, args.sim_func), axis=1)
    #     df["decomp_score"] = df.apply(lambda row: get_decomp_score(row["token_scores"], normalise=False, metric=args.agg_metric), axis=1)

    #     df["sim_function"] = args.sim_func
    #     print(df.head)

    #     df.to_csv(f"{args.save_dir}/{get_time_now()}_{args.dataset_name}_{args.sim_func}_{args.agg_metric}_{args.model_name.split('/')[-1]}.csv")
