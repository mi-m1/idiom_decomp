
import argparse
import logging
import math
import os
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import list_repo_refs
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

LOG = logging.getLogger(__name__)

# Given a model (e.g., OLMo/Pythia), compute layer-wise cosine similarities
# across checkpoints for:
# - sentence vs paraphrase
# - sentence vs idiom span (phrase inside the sentence)

def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    # auto
    if device.type == "cuda":
        # bf16 is great if supported; otherwise fp16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_tokenizer(
    model_name: str,
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, revision=checkpoint, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer
    

def get_all_checkpoints(model_name: str) -> List[str]:
    """Return all branch names for a HF repo (treated as checkpoints)."""
    out = list_repo_refs(model_name)
    return [b.name for b in out.branches]


def sort_ckpt_olmo2(checkpoints: Sequence[str], is_stage2=False) -> List[str]:
    """Sort OLMo2-style checkpoint names by the integer after 'step'."""
    if is_stage2:
        return sorted(checkpoints, key=lambda x: int(x.split("-")[2].split("step")[-1]))
    return sorted(checkpoints, key=lambda x: int(x.split("-")[1].split("step")[-1]))


def sort_ckpt_olmo3(checkpoints: Sequence[str]) -> List[str]:
    """Sort OLMo3-style checkpoint names by the integer after 'step'."""
    return sorted(checkpoints, key=lambda x: int(x.split("step")[-1]))


def logspace_sample_checkpoints(checkpoints: Sequence[str], num_samples: int) -> List[str]:
    """Log-space sample from an already-sorted checkpoint list."""
    if num_samples <= 0:
        return []
    if len(checkpoints) == 0:
        return []
    if num_samples >= len(checkpoints):
        return list(checkpoints)
    indices = np.unique(
        np.round(np.logspace(0, np.log10(len(checkpoints)), num_samples)).astype(int) - 1
    )
    return [checkpoints[i] for i in indices]

# Extract embeddings
def get_layerwise_sentence_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    *,
    show_progress: bool = True,
) -> np.ndarray:
    """Mean-pool token embeddings per layer for each text.

    Returns an array of shape (N, L, D).
    """
    all_layer_embeddings: List[List[np.ndarray]] = []
    iterator = tqdm(texts, desc="Embedding texts") if show_progress else texts
    with torch.no_grad():
        for text in iterator:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # (L, B, T, D)
            layer_embeds = [h.squeeze(0).mean(dim=0).cpu().float().numpy() for h in hidden_states]
            all_layer_embeddings.append(layer_embeds)
    return np.asarray(all_layer_embeddings)


# Find the subsequence of phrase_ids within sent_ids
def find_sublist(sub: Sequence[int], lst: Sequence[int]) -> Optional[int]:
    """Return first start index of sub inside lst; None if not found."""
    if len(sub) == 0:
        return 0
    for i in range(len(lst) - len(sub) + 1):
        if lst[i : i + len(sub)] == list(sub):
            return i
    return None

OnMissingSpan = Literal["error", "skip", "zeros"]


def compute_phrase_embeddings(
    model,
    tokenizer,
    sentences: Sequence[str],
    phrases: Sequence[str],
    *,
    on_missing: OnMissingSpan = "error",
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute layer-wise phrase embeddings for each (sentence, phrase).

    The phrase is located by finding its token id subsequence inside the
    sentence token ids (both encoded with add_special_tokens=False).

    Returns:
        phrase_embeds: (N, L, D)
        found_mask: (N,) bool, True if span was found
    """
    phrase_embeddings: List[List[np.ndarray]] = []
    found_mask: List[bool] = []

    iterator = (
        tqdm(list(zip(sentences, phrases)), desc="Embedding phrases")
        if show_progress
        else zip(sentences, phrases)
    )
    with torch.no_grad():
        for sent, phrase in iterator:
            sent_ids = tokenizer.encode(sent, add_special_tokens=False)
            phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)
            start = find_sublist(phrase_ids, sent_ids)
            if start is None:
                found_mask.append(False)
                msg = f"Phrase span not found in sentence. phrase={phrase!r} sent={sent!r}"
                if on_missing == "error":
                    raise ValueError(msg)
                if on_missing == "skip":
                    continue
                # zeros
                hidden_size = int(model.config.hidden_size)
                num_layers = int(model.config.num_hidden_layers) + 1
                phrase_embeddings.append(
                    [np.zeros(hidden_size, dtype=np.float32) for _ in range(num_layers)]
                )
                continue

            found_mask.append(True)
            indices = list(range(start, start + len(phrase_ids)))
            input_ids = torch.tensor([sent_ids], device=model.device)
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # (L, B, T, D)
            layer_embeds = [
                h[:, indices, :].squeeze(0).mean(dim=0).cpu().float().numpy()
                for h in hidden_states
            ]
            phrase_embeddings.append(layer_embeds)

    phrase_arr = np.asarray(phrase_embeddings)
    found_arr = np.asarray(found_mask, dtype=bool)
    return phrase_arr, found_arr


def cosine_similarity_sentence_phrase(
    sentence_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    eps: float = 1e-8,
):
    """
    Compute layer-wise cosine similarity between sentence and phrase embeddings.

    Args:
        sentence_embeddings: np.ndarray of shape (N, L, D)
        phrase_embeddings:   np.ndarray of shape (N, L, D)
        eps: small constant to avoid division by zero

    Returns:
        similarities: np.ndarray of shape (N, L)
            similarities[i, l] = cosine similarity between
            sentence i and phrase i at layer l
    """
    assert sentence_embeddings.shape == phrase_embeddings.shape, (
        "Sentence and phrase embeddings must have the same shape "
        "(N, L, D)."
    )

    # Normalize embeddings along the hidden dimension
    sent_norm = sentence_embeddings / (
        np.linalg.norm(sentence_embeddings, axis=-1, keepdims=True) + eps
    )
    phrase_norm = phrase_embeddings / (
        np.linalg.norm(phrase_embeddings, axis=-1, keepdims=True) + eps
    )

    # Cosine similarity = dot product of normalized vectors
    similarities = np.sum(sent_norm * phrase_norm, axis=-1)

    return similarities


def l2_distance_sentence_phrase(
    sentence_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
) -> np.ndarray:
    """Layer-wise Euclidean distance between paired embeddings.

    Args:
        sentence_embeddings: (N, L, D)
        phrase_embeddings: (N, L, D)

    Returns:
        distances: (N, L)
    """
    assert sentence_embeddings.shape == phrase_embeddings.shape, (
        "Sentence and phrase embeddings must have the same shape (N, L, D)."
    )
    return np.linalg.norm(sentence_embeddings - phrase_embeddings, axis=-1)


def pearson_corr_sentence_phrase(
    sentence_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Layer-wise Pearson correlation between paired embeddings.

    Treats each hidden dimension as an observation, correlating the two
    D-dimensional vectors per (example, layer).

    Returns:
        corr: (N, L)
    """
    assert sentence_embeddings.shape == phrase_embeddings.shape, (
        "Sentence and phrase embeddings must have the same shape (N, L, D)."
    )
    x = sentence_embeddings - sentence_embeddings.mean(axis=-1, keepdims=True)
    y = phrase_embeddings - phrase_embeddings.mean(axis=-1, keepdims=True)
    num = np.sum(x * y, axis=-1)
    denom = (
        np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + eps
    )
    return num / denom


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix (n x n) in feature space."""
    n = K.shape[0]
    if n == 0:
        return K
    one_n = np.ones((n, n), dtype=K.dtype) / float(n)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def _linear_gram(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def _hsic(Kc: np.ndarray, Lc: np.ndarray) -> float:
    """Unnormalized HSIC using centered Gram matrices."""
    return float(np.sum(Kc * Lc))


def linear_cka_layerwise(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute linear CKA between two sets of representations.

    X and Y are (N, D) matrices where rows are examples.
    Returns a scalar in [0, 1] (up to numerical error).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of examples")
    # Center features across examples.
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    K = _center_gram(_linear_gram(Xc))
    L = _center_gram(_linear_gram(Yc))
    hsic_xy = _hsic(K, L)
    hsic_xx = _hsic(K, K)
    hsic_yy = _hsic(L, L)
    denom = math.sqrt(hsic_xx * hsic_yy) + eps
    return hsic_xy / denom


def linear_cka_sentence_phrase(
    sentence_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute linear CKA per layer, across the batch of examples.

    Args:
        sentence_embeddings: (N, L, D)
        phrase_embeddings: (N, L, D)

    Returns:
        cka: (L,) array of CKA values, one per layer.
    """
    assert sentence_embeddings.shape == phrase_embeddings.shape, (
        "Sentence and phrase embeddings must have the same shape (N, L, D)."
    )
    _, num_layers, _ = sentence_embeddings.shape
    out = np.zeros((num_layers,), dtype=np.float64)
    for l in range(num_layers):
        out[l] = linear_cka_layerwise(sentence_embeddings[:, l, :], phrase_embeddings[:, l, :])
    return out


def select_checkpoints(model_name: str, all_checkpoints: Sequence[str], num_checkpoints: Sequence[int]) -> List[str]:
    """Pick checkpoint names to analyze based on model family conventions."""
    name = model_name.lower()

    if "olmo" in name:
        stage_1 = [x for x in all_checkpoints if "stage1" in x]
        stage_2 = [x for x in all_checkpoints if "stage2" in x]
        LOG.info("Found %d stage1 and %d stage2 branches", len(stage_1), len(stage_2))
        if len(num_checkpoints) != 2:
            raise ValueError("For OLMo models, pass exactly two ints via --num_checkpoints: stage1 stage2")
        n1, n2 = num_checkpoints
        if "olmo-2" in name:
            stage_1 = sort_ckpt_olmo2(stage_1)
            stage_2 = sort_ckpt_olmo2(stage_2, is_stage2=True)
        else:
            stage_1 = sort_ckpt_olmo3(stage_1)
            stage_2 = sort_ckpt_olmo3(stage_2)
        stage1 = logspace_sample_checkpoints(stage_1, n1)
        stage2 = logspace_sample_checkpoints(stage_2, n2)
        return stage1 + stage2

    if "pythia" in name:
        all_minus_main = [x for x in all_checkpoints if x != "main"]
        total = int(sum(num_checkpoints)) if len(num_checkpoints) > 0 else 0
        # Pythia checkpoints are commonly numeric or step-based; we keep order as returned
        # but sample logspace over that list.
        return logspace_sample_checkpoints(all_minus_main, total)

    # Default: if user provided counts, just take that many from the head.
    total = int(sum(num_checkpoints)) if len(num_checkpoints) > 0 else 0
    if total <= 0:
        raise ValueError("--num_checkpoints is required for non-OLMo/Pythia models")
    return list(all_checkpoints[:total])


def shard_items(
    items: Sequence[str],
    *,
    num_shards: int,
    shard_idx: int,
    strategy: Literal["contiguous", "round_robin"] = "contiguous",
) -> List[str]:
    """Return the subset of items assigned to a given shard.

    Args:
        items: Full ordered list of items to split.
        num_shards: Total number of shards.
        shard_idx: Which shard to return, in [0, num_shards).
        strategy:
            - contiguous: split into near-equal contiguous chunks
            - round_robin: take every num_shards-th item (items[shard_idx::num_shards])
    """
    if num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError("--shard_idx must satisfy 0 <= shard_idx < num_shards")
    if num_shards == 1:
        return list(items)

    if strategy == "round_robin":
        return list(items)[shard_idx::num_shards]

    # contiguous
    splits = np.array_split(np.arange(len(items)), num_shards)
    idxs = splits[shard_idx]
    return [items[int(i)] for i in idxs]



def process_args():

    parser = argparse.ArgumentParser(
        description="Analyze model checkpoints for idiom decomposability.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., allenai/Olmo-3-7B-Instruct).",
    )
    
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        nargs="+",
        help="List of checkpoints to analyze (e.g. --num_checkpoints 5 10 20)",
    )
    
       
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Data type for model weights.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )

    parser.add_argument(
        "--idioms_file",
        type=str,
        required=True,
        help="Path to CSV file containing sentences, extracted forms of idioms.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis results.",
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="If set, run in testing mode with only a few rows of df.",
    )

    parser.add_argument(
        "--on_missing_span",
        type=str,
        default="error",
        choices=["error", "skip", "zeros"],
        help="What to do if the phrase token span isn't found inside the sentence.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more detailed logging.",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["cosine"],
        choices=["cosine", "l2", "pearson", "linear_cka"],
        help=(
            "Similarity metrics to compute. "
            "Pairwise metrics (cosine/l2/pearson) are per-example per-layer; "
            "linear_cka is per-layer across the dataset (same value for all examples)."
        ),
    )

    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Shard checkpoint evaluation across multiple jobs (default: 1 = no sharding).",
    )

    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Which shard to run (0-indexed). Only used when --num_shards > 1.",
    )

    parser.add_argument(
        "--shard_strategy",
        type=str,
        default="contiguous",
        choices=["contiguous", "round_robin"],
        help=(
            "How to split checkpoints across shards: contiguous chunks or round-robin assignment."
        ),
    )

    return parser.parse_args()

    
if __name__ == "__main__":
    # Process command-line arguments
    args = process_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve device and dtype
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load idioms data
    df = pd.read_csv(args.idioms_file)
    if args.testing:
        LOG.info("Running in testing mode (first 5 rows)")
        df = df.head(5)

    # Get all checkpoints
    all_ckpt_names = get_all_checkpoints(args.model_name)
    LOG.info("Found %d checkpoints (branches)", len(all_ckpt_names))
    if not args.num_checkpoints:
        raise ValueError("--num_checkpoints must be provided")
    if args.testing:
        ckpt_names = list(all_ckpt_names[:2])
    else:
        ckpt_names = select_checkpoints(args.model_name, all_ckpt_names, args.num_checkpoints)

    # Optional: shard checkpoints across multiple parallel jobs.
    if args.num_shards and args.num_shards > 1:
        ckpt_names = shard_items(
            ckpt_names,
            num_shards=int(args.num_shards),
            shard_idx=int(args.shard_idx),
            strategy=args.shard_strategy,
        )
        LOG.info(
            "Sharding enabled: shard %d/%d using %s strategy -> %d checkpoints",
            int(args.shard_idx),
            int(args.num_shards),
            args.shard_strategy,
            len(ckpt_names),
        )

    LOG.info("Analyzing %d checkpoints", len(ckpt_names))
    
    # Validate required columns
    required_cols = {"premise", "hypothesis", "extracted_idiom"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in idioms_file: {sorted(missing)}")
    all_sentences = df["premise"].astype(str).tolist()
    all_paraphrases = df["hypothesis"].astype(str).tolist()
    all_phrases = df["extracted_idiom"].astype(str).tolist()
    all_phrases = [" " + x for x in all_phrases]

    # Create output directory
    model_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1])
    os.makedirs(model_dir, exist_ok=True)

    # Process each checkpoint
    LOG.info("Starting analysis over checkpoints...")
    LOG.info("Target checkpoints: %s", ckpt_names)
    for ckpt in ckpt_names:
        LOG.info("Processing checkpoint: %s", ckpt)
        # Load model and tokenizer
        model, tokenizer = load_model_tokenizer(args.model_name, ckpt, device, dtype)

        # Extract embeddings
        sentence_embeddings = get_layerwise_sentence_embeddings(all_sentences, tokenizer, model)
        paraphrase_embeddings = get_layerwise_sentence_embeddings(all_paraphrases, tokenizer, model)
        
        # Phrase embeddings
        phrase_embeddings, found_mask = compute_phrase_embeddings(
            model,
            tokenizer,
            all_sentences,
            all_phrases,
            on_missing=args.on_missing_span,
        )
        if found_mask.size > 0:
            LOG.info("Phrase spans found for %d/%d examples", int(found_mask.sum()), int(found_mask.size))

        # Compute requested metrics
        metrics = set(args.metrics)

        pair_sentence_phrase: dict[str, np.ndarray] = {}
        pair_sentence_paraphrase: dict[str, np.ndarray] = {}

        if "cosine" in metrics:
            pair_sentence_phrase["cosine"] = cosine_similarity_sentence_phrase(
                sentence_embeddings, phrase_embeddings
            )
            pair_sentence_paraphrase["cosine"] = cosine_similarity_sentence_phrase(
                sentence_embeddings, paraphrase_embeddings
            )

        if "l2" in metrics:
            pair_sentence_phrase["l2"] = l2_distance_sentence_phrase(
                sentence_embeddings, phrase_embeddings
            )
            pair_sentence_paraphrase["l2"] = l2_distance_sentence_phrase(
                sentence_embeddings, paraphrase_embeddings
            )

        if "pearson" in metrics:
            pair_sentence_phrase["pearson"] = pearson_corr_sentence_phrase(
                sentence_embeddings, phrase_embeddings
            )
            pair_sentence_paraphrase["pearson"] = pearson_corr_sentence_phrase(
                sentence_embeddings, paraphrase_embeddings
            )

        # CKA is computed across the batch (one scalar per layer).
        cka_sentence_phrase: Optional[np.ndarray] = None
        cka_sentence_paraphrase: Optional[np.ndarray] = None
        if "linear_cka" in metrics:
            cka_sentence_phrase = linear_cka_sentence_phrase(sentence_embeddings, phrase_embeddings)
            cka_sentence_paraphrase = linear_cka_sentence_phrase(sentence_embeddings, paraphrase_embeddings)

        # Prepare output DataFrame
        rows = []
        # Determine number of layers from embeddings.
        num_layers = int(sentence_embeddings.shape[1])
        for i, sent in enumerate(all_sentences):
            for layer in range(num_layers):
                row = {
                    "checkpoint": ckpt,
                    "sentence_id": i,
                    "sentence": sent,
                    "phrase": all_phrases[i],
                    "layer": layer,
                }

                for m, arr in pair_sentence_phrase.items():
                    row[f"{m}_sentence_phrase"] = float(arr[i, layer])
                for m, arr in pair_sentence_paraphrase.items():
                    row[f"{m}_sentence_paraphrase"] = float(arr[i, layer])

                if cka_sentence_phrase is not None:
                    row["linear_cka_sentence_phrase"] = float(cka_sentence_phrase[layer])
                if cka_sentence_paraphrase is not None:
                    row["linear_cka_sentence_paraphrase"] = float(cka_sentence_paraphrase[layer])

                rows.append(row)
        df_out = pd.DataFrame(rows)
        
        # Save to CSV
        out_path = os.path.join(model_dir, f"{ckpt}.csv")
        df_out.to_csv(out_path, index=False)
        LOG.info("Wrote %s", out_path)
