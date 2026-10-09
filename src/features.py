"""Surface statistics used in RQ4. These are descriptive, not model inputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")
FENCE_RE = re.compile(r"```")
HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
BOLD_RE = re.compile(r"\*\*")


def row_features(text: str) -> dict:
    text = text or ""
    words = WORD_RE.findall(text)
    n_words = len(words)
    lowered = [w.lower() for w in words]
    unique = len(set(lowered))
    # Type-token ratio on a fixed prefix so longer answers are not punished.
    prefix = lowered[:200]
    ttr = (len(set(prefix)) / len(prefix)) if prefix else 0.0
    # Fixed window so short answers are not credited with a higher type-token ratio.
    ttr_50 = (len(set(lowered[:50])) / 50.0) if len(lowered) >= 50 else float("nan")
    letters = [ch for ch in text if ch.isalpha()]
    upper = sum(ch.isupper() for ch in letters)
    punct = sum(ch in ",.;:!?\"'`" for ch in text)
    return {
        "n_chars": len(text),
        "n_words": n_words,
        "n_sentences": max(len(SENT_RE.findall(text)), 1 if text.strip() else 0),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr_200": ttr,
        "ttr_50": ttr_50,
        "unique_types": unique,
        "punct_per_word": punct / n_words if n_words else 0.0,
        "exclaim": text.count("!"),
        "question_marks": text.count("?"),
        "upper_ratio": upper / len(letters) if letters else 0.0,
        "code_fences": len(FENCE_RE.findall(text)) // 2,
        "headers": len(HEADER_RE.findall(text)),
        "bullets": len(BULLET_RE.findall(text)),
        "bold_markers": len(BOLD_RE.findall(text)),
        "newline_per_word": text.count("\n") / n_words if n_words else 0.0,
        "markdown_hits": (
            (len(FENCE_RE.findall(text)) // 2)
            + len(HEADER_RE.findall(text))
            + len(BULLET_RE.findall(text))
            + len(BOLD_RE.findall(text))
        ),
    }


def feature_table(frame: pd.DataFrame, text_col: str = "response") -> pd.DataFrame:
    feats = pd.DataFrame([row_features(text) for text in frame[text_col].tolist()])
    feats["model"] = frame["model"].to_numpy()
    feats["domain"] = frame["domain"].to_numpy()
    return feats


def feature_summary(feats: pd.DataFrame) -> dict:
    columns = [
        "n_words",
        "avg_word_len",
        "ttr_200",
        "ttr_50",
        "punct_per_word",
        "upper_ratio",
        "code_fences",
        "headers",
        "bullets",
        "bold_markers",
        "markdown_hits",
        "newline_per_word",
        "exclaim",
    ]
    means = feats.groupby("model")[columns].mean().round(4)
    medians = feats.groupby("model")[["n_words", "ttr_50", "markdown_hits"]].median().round(4)
    return {"mean": means.to_dict(), "median_selected": medians.to_dict()}
