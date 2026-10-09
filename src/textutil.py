"""Tokenization, vocabulary, and the text views used by RQ2 and RQ4."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch

from src.config import (
    MAX_INSTR_TOKENS,
    MAX_LEN,
    MAX_RESP_TOKENS,
    MIN_FREQ,
    VOCAB_SIZE,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\s\w]")

PAD = "<pad>"
UNK = "<unk>"
SEP = "<sep>"


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class Vocab:
    def __init__(self, tokens: list[list[str]]):
        counts = Counter(tok for seq in tokens for tok in seq)
        most = [tok for tok, n in counts.most_common() if n >= MIN_FREQ]
        most = [tok for tok in most if tok not in {PAD, UNK, SEP}]
        most = most[: VOCAB_SIZE - 3]
        self.stoi = {PAD: 0, UNK: 1, SEP: 2}
        for tok in most:
            self.stoi[tok] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(tok, 1) for tok in tokenize(text)]
        if max_len is not None:
            ids = ids[:max_len]
        return ids


def encode_view(instruction: str, response: str, view: str, vocab: Vocab) -> list[int]:
    """Build token ids for one RQ2 view. Combined view reserves room for both sides."""
    if view == "output":
        return vocab.encode(response, MAX_LEN)
    if view == "input":
        return vocab.encode(instruction, MAX_LEN)
    if view == "both":
        left = vocab.encode(instruction, MAX_INSTR_TOKENS)
        right = vocab.encode(response, MAX_RESP_TOKENS)
        return (left + [2] + right)[:MAX_LEN]
    raise ValueError(view)


def pad_batch(seqs: list[list[int]], max_len: int = MAX_LEN) -> torch.Tensor:
    arr = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        n = min(len(seq), max_len)
        if n:
            arr[i, :n] = seq[:n]
    return torch.from_numpy(arr)


MARKDOWN_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1")
ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+)(?<!\*)\*(?!\*)")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def strip_markdown(text: str) -> str:
    text = MARKDOWN_FENCE_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = HEADER_RE.sub("", text)
    text = BOLD_RE.sub(r"\2", text)
    text = ITALIC_RE.sub(r"\1", text)
    text = BULLET_RE.sub("", text)
    return text


def force_n_words(text: str, n: int = 60) -> str:
    """Make every text exactly n words so length cannot separate classes."""
    words = (text or "").split()
    if not words:
        words = ["empty"]
    if len(words) >= n:
        return " ".join(words[:n])
    reps = (n + len(words) - 1) // len(words)
    return " ".join((words * reps)[:n])


def strip_punct_case(text: str) -> str:
    return PUNCT_RE.sub(" ", text or "").lower()


ABLATIONS = {
    "strip_markdown": strip_markdown,
    "fixed_length": lambda text: force_n_words(text, 60),
    "strip_punct": strip_punct_case,
    "strip_all": lambda text: force_n_words(strip_punct_case(strip_markdown(text)), 60),
}
