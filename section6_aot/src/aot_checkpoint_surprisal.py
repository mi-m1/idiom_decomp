
import argparse
import json
import logging
import os
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG = logging.getLogger(__name__)

# Given a model (e.g., OLMo/Pythia), compute surprisal of a specified phrase
# inside a sentence across model checkpoints.

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=checkpoint, torch_dtype=dtype
    )
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
    """Log-space sample from an already-sorted checkpoint list.

    Notes:
        A naive `np.logspace(...)->round()->unique` approach can collapse to fewer
        than `num_samples` indices due to duplicates (especially near the start
        of the curve). That, in turn, breaks expectations around sharding (e.g.
        "100 checkpoints" ends up being 75).

        This implementation guarantees returning exactly `num_samples` unique
        checkpoints whenever `len(checkpoints) >= num_samples`.
    """
    if num_samples <= 0 or len(checkpoints) == 0:
        return []

    n = len(checkpoints)
    if num_samples >= n:
        return list(checkpoints)

    # Oversample a geometric progression, then pick evenly across unique indices.
    oversample = max(num_samples * 10, num_samples + 50)
    raw = np.geomspace(1, n, oversample)
    cand = np.unique(np.clip(np.rint(raw).astype(int) - 1, 0, n - 1))

    # If uniqueness is still too low (can happen for small n), mix in linear indices.
    if cand.size < num_samples:
        lin = np.unique(np.rint(np.linspace(0, n - 1, num_samples)).astype(int))
        cand = np.unique(np.concatenate([cand, lin]))

    if cand.size < num_samples:
        # Ultimate fallback: just use the full range.
        cand = np.arange(n, dtype=int)

    pos = np.rint(np.linspace(0, cand.size - 1, num_samples)).astype(int)
    idx = np.unique(cand[pos])

    # Ensure we return exactly num_samples unique indices.
    if idx.size < num_samples:
        remaining = np.setdiff1d(np.arange(n, dtype=int), idx, assume_unique=False)
        need = num_samples - idx.size
        idx = np.sort(np.concatenate([idx, remaining[:need]]))

    if idx.size != num_samples:
        # This should not happen; keep it loud.
        raise RuntimeError(
            f"Failed to sample {num_samples} unique checkpoints from n={n} (got {idx.size})."
        )

    return [checkpoints[int(i)] for i in idx]


# Find the subsequence of phrase_ids within sent_ids
def find_sublist(sub: Sequence[int], lst: Sequence[int]) -> Optional[int]:
    """Return first start index of sub inside lst; None if not found."""
    if len(sub) == 0:
        return 0
    for i in range(len(lst) - len(sub) + 1):
        if lst[i : i + len(sub)] == list(sub):
            return i
    return None


def pretokenize_and_find_spans(
    tokenizer,
    sentences: Sequence[str],
    phrases: Sequence[str],
) -> Tuple[List[List[int]], List[List[int]], List[Optional[int]]]:
    """Tokenize (sentence, phrase) pairs and locate phrase span within sentence ids.

    Both sentence and phrase are encoded with add_special_tokens=False.

    Returns:
        sent_ids_list: list of token id lists for each sentence
        phrase_ids_list: list of token id lists for each phrase
        start_list: list of start indices (into sent_ids) where phrase begins, or None
    """
    sent_ids_list: List[List[int]] = []
    phrase_ids_list: List[List[int]] = []
    start_list: List[Optional[int]] = []
    for sent, phrase in zip(sentences, phrases):
        sent_ids = tokenizer.encode(str(sent), add_special_tokens=False)
        phrase_ids = tokenizer.encode(str(phrase), add_special_tokens=False)
        start = find_sublist(phrase_ids, sent_ids)
        sent_ids_list.append(list(sent_ids))
        phrase_ids_list.append(list(phrase_ids))
        start_list.append(start)
    return sent_ids_list, phrase_ids_list, start_list

OnMissingSpan = Literal["error", "skip", "zeros"]


def compute_phrase_surprisal(
    model,
    tokenizer,
    sent_ids_list: Sequence[Sequence[int]],
    phrase_ids_list: Sequence[Sequence[int]],
    start_list: Sequence[Optional[int]],
    *,
    on_missing: OnMissingSpan = "error",
    show_progress: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[List[float]],
    List[List[int]],
    List[List[str]],
]:
    """Compute surprisal of each phrase given its sentence context.

    Surprisal is computed as the negative log-probability of the phrase tokens
    under a causal LM, conditioned on the preceding context in the sentence.

    Implementation details:
        - We optionally prepend BOS if tokenizer.bos_token_id is available.
        - For a phrase token at position p in the model input, we use logits at (p-1)
          to score that token (standard next-token prediction).

    Returns:
        surprisal_sum: (N,) float32
        surprisal_mean: (N,) float32 (mean over phrase tokens scored)
        phrase_token_count: (N,) int32 (tokens scored)
        found_mask: (N,) bool
        per_token_surprisal: list of length N, each a list[float] of token NLLs
        per_token_ids: list of length N, each a list[int] of token ids scored
        per_token_strs: list of length N, each a list[str] of token strings scored
    """
    n = len(sent_ids_list)
    if not (len(phrase_ids_list) == n and len(start_list) == n):
        raise ValueError("Mismatched lengths for sent_ids/phrase_ids/starts")

    bos = tokenizer.bos_token_id
    surprisal_sum = np.full((n,), np.nan, dtype=np.float32)
    surprisal_mean = np.full((n,), np.nan, dtype=np.float32)
    phrase_token_count = np.zeros((n,), dtype=np.int32)
    found_mask = np.zeros((n,), dtype=bool)
    per_token_surprisal: List[List[float]] = [[] for _ in range(n)]
    per_token_ids: List[List[int]] = [[] for _ in range(n)]
    per_token_strs: List[List[str]] = [[] for _ in range(n)]

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing phrase surprisal")

    with torch.no_grad():
        for i in iterator:
            sent_ids = list(sent_ids_list[i])
            phrase_ids = list(phrase_ids_list[i])
            start = start_list[i]

            if start is None:
                msg = f"Phrase span not found for example {i}"
                if on_missing == "error":
                    raise ValueError(msg)
                if on_missing == "skip":
                    # Skip is handled upstream by filtering; keep NaNs here.
                    continue
                # zeros
                found_mask[i] = False
                surprisal_sum[i] = 0.0
                surprisal_mean[i] = 0.0
                phrase_token_count[i] = 0
                continue

            found_mask[i] = True

            # Build model input, optionally with BOS.
            if bos is not None:
                input_ids = [int(bos)] + [int(t) for t in sent_ids]
                offset = 1
            else:
                input_ids = [int(t) for t in sent_ids]
                offset = 0

            # Phrase positions in the model input.
            phrase_start = offset + int(start)
            phrase_positions = list(range(phrase_start, phrase_start + len(phrase_ids)))

            # If no BOS and phrase starts at 0, we can't score token 0 (needs logits[-1]).
            if offset == 0 and phrase_positions and phrase_positions[0] == 0:
                # Score from the 2nd token onward.
                phrase_positions = phrase_positions[1:]
                phrase_ids = phrase_ids[1:]

            if len(phrase_ids) == 0 or len(phrase_positions) == 0:
                # Empty phrase or nothing scoreable.
                surprisal_sum[i] = 0.0
                surprisal_mean[i] = 0.0
                phrase_token_count[i] = 0
                continue

            input_tensor = torch.tensor([input_ids], device=model.device)
            out = model(input_tensor)
            logits = out.logits  # (1, T, V)
            log_probs = F.log_softmax(logits, dim=-1)

            # Score each phrase token using the previous position's logits.
            token_nlls: List[float] = []
            token_ids_scored: List[int] = []
            for pos, tok in zip(phrase_positions, phrase_ids):
                prev_pos = pos - 1
                if prev_pos < 0 or prev_pos >= log_probs.shape[1]:
                    continue
                lp = float(log_probs[0, prev_pos, int(tok)].item())
                nll = -lp
                token_nlls.append(nll)
                token_ids_scored.append(int(tok))

            per_token_surprisal[i] = list(token_nlls)
            per_token_ids[i] = list(token_ids_scored)
            if len(token_ids_scored) > 0:
                per_token_strs[i] = list(tokenizer.convert_ids_to_tokens(token_ids_scored))

            if len(token_nlls) == 0:
                surprisal_sum[i] = 0.0
                surprisal_mean[i] = 0.0
                phrase_token_count[i] = 0
            else:
                s = float(np.sum(token_nlls))
                surprisal_sum[i] = s
                surprisal_mean[i] = float(s / len(token_nlls))
                phrase_token_count[i] = int(len(token_nlls))

    return (
        surprisal_sum,
        surprisal_mean,
        phrase_token_count,
        found_mask,
        per_token_surprisal,
        per_token_ids,
        per_token_strs,
    )


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
        description="Compute phrase surprisal across model checkpoints.",
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
        help="Path to CSV containing columns: premise, hypothesis, extracted_idiom.",
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
    LOG.info("Total checkpoints selected: %d", len(ckpt_names))
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
    # hypothesis is currently unused for surprisal-only analysis
    # (kept as a required column for compatibility with existing datasets)
    _all_paraphrases = df["hypothesis"].astype(str).tolist()
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

        # Locate spans once per checkpoint (tokenizer can vary across models).
        sent_ids_list, phrase_ids_list, start_list = pretokenize_and_find_spans(
            tokenizer, all_sentences, all_phrases
        )

        if args.on_missing_span == "skip":
            keep = [i for i, s in enumerate(start_list) if s is not None]
            LOG.info("--on_missing_span=skip: keeping %d/%d examples", len(keep), len(all_sentences))
            kept_ids = keep
            sentences = [all_sentences[i] for i in keep]
            phrases = [all_phrases[i] for i in keep]
            sent_ids_list = [sent_ids_list[i] for i in keep]
            phrase_ids_list = [phrase_ids_list[i] for i in keep]
            start_list = [start_list[i] for i in keep]
        else:
            kept_ids = list(range(len(all_sentences)))
            sentences = all_sentences
            phrases = all_phrases

        # Phrase surprisal (per example; not layer-specific).
        (
            phrase_surprisal_sum,
            phrase_surprisal_mean,
            phrase_token_count,
            surprisal_found,
            phrase_surprisal_per_token,
            phrase_surprisal_token_ids,
            phrase_surprisal_token_strs,
        ) = compute_phrase_surprisal(
            model,
            tokenizer,
            sent_ids_list,
            phrase_ids_list,
            start_list,
            on_missing=args.on_missing_span,
        )

        # Prepare output DataFrame (one row per example).
        rows = []
        for i, sent in enumerate(sentences):
            rows.append(
                {
                    "checkpoint": ckpt,
                    "sentence_id": int(kept_ids[i]),
                    "sentence": sent,
                    "phrase": phrases[i],
                    "phrase_surprisal_sum": float(phrase_surprisal_sum[i]),
                    "phrase_surprisal_mean": float(phrase_surprisal_mean[i]),
                    "phrase_surprisal_tokens": int(phrase_token_count[i]),
                    "phrase_surprisal_found": bool(surprisal_found[i]),
                    # JSON-serialized lists for easy roundtrip.
                    "phrase_surprisal_per_token": json.dumps(phrase_surprisal_per_token[i]),
                    "phrase_surprisal_token_ids": json.dumps(phrase_surprisal_token_ids[i]),
                    "phrase_surprisal_token_strs": json.dumps(phrase_surprisal_token_strs[i]),
                }
            )
        df_out = pd.DataFrame(rows)
        
        # Save to CSV
        out_path = os.path.join(model_dir, f"{ckpt}.csv")
        df_out.to_csv(out_path, index=False)
        LOG.info("Wrote %s", out_path)