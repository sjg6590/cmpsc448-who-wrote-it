"""Download UltraFeedback (if needed) and build a balanced, prompt-level split.

The processed table is written to data/processed/samples.csv.gz and is the
file training reads. Re-running this script rebuilds that table from the
public UltraFeedback JSONL files.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    DOMAINS,
    EXCLUDED_SOURCES,
    FAMILY,
    MAX_PER_CELL,
    MAX_STORED_CHARS,
    MIN_RESPONSE_CHARS,
    MODELS,
    RAW_DIR,
    SEED,
    SOURCES,
    SPLIT_RATIOS,
)
from src.domains import label_domain

HF_FILES = (
    "truthful_qa.jsonl",
    "false_qa.jsonl",
    "evol_instruct.jsonl",
    "ultrachat.jsonl",
    "flan.jsonl",
    "sharegpt.jsonl",
)

LETTER_RE = re.compile(r"[A-Za-z]")
NON_ASCII_LETTER_RE = re.compile(r"[^\x00-\x7F]")


def ensure_raw() -> None:
    """Fetch UltraFeedback JSONL files when they are not already on disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in HF_FILES if not (RAW_DIR / name).exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download

    for name in missing:
        print(f"downloading {name}", flush=True)
        hf_hub_download(
            "openbmb/UltraFeedback",
            name,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
        )


def _mostly_latin(text: str) -> bool:
    letters = LETTER_RE.findall(text)
    if len(letters) < 15:
        return False
    non_ascii = len(NON_ASCII_LETTER_RE.findall(text))
    return non_ascii / max(len(text), 1) < 0.08


def collect() -> tuple[pd.DataFrame, dict]:
    rows = []
    # iter_rows returns a generator; capture stats by wrapping collect logic here.
    stats = Counter()
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                source = obj.get("source") or ""
                stats["prompts_seen"] += 1
                if source in EXCLUDED_SOURCES:
                    stats["prompts_excluded_source"] += 1
                    continue
                if source not in SOURCES:
                    stats["prompts_unknown_source"] += 1
                    continue
                instruction = (obj.get("instruction") or "").strip()
                if len(instruction) < 12:
                    stats["prompts_short_instruction"] += 1
                    continue
                domain = label_domain(source, instruction)
                if domain not in DOMAINS:
                    stats["prompts_other_domain"] += 1
                    continue
                digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:16]
                # Released JSONL rows do not include the id field shown in the dataset card.
                prompt_id = f"{source}::{digest}"
                kept_any = False
                for completion in obj.get("completions") or []:
                    model = completion.get("model")
                    if model not in MODELS:
                        continue
                    stats["candidate_completions"] += 1
                    response = completion.get("response")
                    if not isinstance(response, str):
                        stats["drop_non_string"] += 1
                        continue
                    response = response.strip()
                    if len(response) < MIN_RESPONSE_CHARS:
                        stats["drop_short_response"] += 1
                        continue
                    if not _mostly_latin(response):
                        stats["drop_non_latin"] += 1
                        continue
                    kept_any = True
                    stats["kept_before_balance"] += 1
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "source": source,
                            "domain": domain,
                            "model": model,
                            "family": FAMILY[model],
                            "instruction": instruction,
                            "response": response[:MAX_STORED_CHARS],
                            "response_chars_full": len(response),
                            "response_words_full": len(response.split()),
                            "truncated_for_storage": int(len(response) > MAX_STORED_CHARS),
                        }
                    )
                if kept_any:
                    stats["prompts_kept"] += 1
    return pd.DataFrame(rows), dict(stats)


def balance_and_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Downsample each model to the same count inside each domain, then split prompts."""
    selected_index = []
    targets = {}
    for domain in DOMAINS:
        part = df[df["domain"] == domain]
        counts = part.groupby("model").size()
        target = int(counts.min()) if len(counts) == len(MODELS) else 0
        target = min(target, MAX_PER_CELL)
        targets[domain] = target
        for model in MODELS:
            pool = part.index[part["model"] == model].to_numpy()
            chosen = rng.choice(pool, size=target, replace=False)
            selected_index.extend(chosen.tolist())
    balanced = df.loc[selected_index].copy()

    # Prompt-level split, stratified by domain. Every completion of a prompt
    # stays in one split so test prompts are unseen.
    prompt_domain = (
        balanced.groupby("prompt_id")["domain"].agg(lambda s: s.iloc[0]).reset_index()
    )
    assignments = {}
    for domain, group in prompt_domain.groupby("domain"):
        ids = group["prompt_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise SystemExit(f"Domain {domain} has only {n} prompts; cannot split.")
        n_train = int(round(n * SPLIT_RATIOS[0]))
        n_val = int(round(n * SPLIT_RATIOS[1]))
        n_test = n - n_train - n_val
        # Keep every split non-empty even when rounding would swallow the test set.
        if n_test < 1 or n_val < 1 or n_train < 1:
            n_test = max(1, n // 7)
            n_val = max(1, n // 7)
            n_train = n - n_val - n_test
        for prompt_id in ids[:n_train]:
            assignments[prompt_id] = "train"
        for prompt_id in ids[n_train : n_train + n_val]:
            assignments[prompt_id] = "val"
        for prompt_id in ids[n_train + n_val :]:
            assignments[prompt_id] = "test"
    balanced["split"] = balanced["prompt_id"].map(assignments)
    balanced["balance_target_per_model"] = balanced["domain"].map(targets)
    return balanced.reset_index(drop=True)


def summarize(df: pd.DataFrame, filter_stats: dict) -> dict:
    by_model = df.groupby(["model", "split"]).size().unstack(fill_value=0)
    by_domain = df.groupby(["domain", "model"]).size().unstack(fill_value=0)
    by_source = df.groupby(["source", "model"]).size().unstack(fill_value=0)
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = set(df.loc[df["split"] == left, "prompt_id"])
        b = set(df.loc[df["split"] == right, "prompt_id"])
        overlap[f"{left}_x_{right}"] = len(a & b)
    return {
        "dataset": "openbmb/UltraFeedback",
        "license": "MIT",
        "models": list(MODELS),
        "families": FAMILY,
        "excluded_sources": list(EXCLUDED_SOURCES),
        "filter_stats": filter_stats,
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "split_prompt_overlap": overlap,
        "counts_by_model_split": by_model.to_dict(),
        "counts_by_domain_model": by_domain.to_dict(),
        "counts_by_source_model": by_source.to_dict(),
        "word_length_by_model": df.groupby("model")["response_words_full"]
        .describe()
        .round(2)
        .to_dict(),
        "storage_truncation_rate": float(df["truncated_for_storage"].mean()),
    }


def main() -> None:
    ensure_raw()
    rng = np.random.default_rng(SEED)
    raw, stats = collect()
    if raw.empty:
        raise SystemExit("No rows collected. Check that UltraFeedback JSONL files exist.")
    df = balance_and_split(raw, rng)
    if df["split"].isna().any():
        raise SystemExit("Some prompts were not assigned a split.")
    summary = summarize(df, stats)
    if any(summary["split_prompt_overlap"].values()):
        raise SystemExit(f"Prompt leakage across splits: {summary['split_prompt_overlap']}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "samples.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_rows", "n_prompts", "counts_by_model_split", "counts_by_domain_model", "split_prompt_overlap")}, indent=2))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
