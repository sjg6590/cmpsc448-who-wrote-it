"""Shared constants. Every experiment uses SEED."""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
RAW_DIR = ROOT / "data" / "raw" / "ultrafeedback"

# One checkpoint model from each family in UltraFeedback (MIT license).
# Bard was not sampled on some FLAN subsets; those sources are dropped in prepare_data.
MODELS = (
    "bard",
    "falcon-40b-instruct",
    "gpt-4",
    "llama-2-70b-chat",
)

FAMILY = {
    "bard": "Bard",
    "falcon-40b-instruct": "Falcon",
    "gpt-4": "GPT",
    "llama-2-70b-chat": "Llama",
}

# Sources on which every selected model was actually queried.
# flan_v2_niv2, flan_v2_p3, and flan_v2_flan2021 contain zero Bard completions.
SOURCES = (
    "evol_instruct",
    "false_qa",
    "flan_v2_cot",
    "sharegpt",
    "truthful_qa",
    "ultrachat",
)

DOMAINS = ("code", "reasoning", "qa", "writing")

# Sources excluded so model identity is not predictable from prompt availability.
EXCLUDED_SOURCES = (
    "flan_v2_niv2",
    "flan_v2_p3",
    "flan_v2_flan2021",
)

MIN_RESPONSE_CHARS = 40
MAX_STORED_CHARS = 6000

SPLIT_RATIOS = (0.70, 0.15, 0.15)
# Cap each model × domain cell so question answering does not dwarf code,
# reasoning, and writing. The uncapped QA pool is several times larger.
MAX_PER_CELL = 700

MAX_LEN = 128
MAX_INSTR_TOKENS = 40
MAX_RESP_TOKENS = 87  # + 1 SEP token = 128
VOCAB_SIZE = 12000
MIN_FREQ = 2

EMBED_DIM = 128
CNN_FILTERS = 96
# Chapter 4's 1D example uses one filter width, 3, with zero padding.
CNN_KERNEL = 3
RNN_HIDDEN = 128

BATCH_SIZE = 64
# alpha_t in Chapter 9. Adam is the optimizer the lecture recommends.
LR = 1e-3
MAX_EPOCHS = 12
# Stop if validation accuracy does not improve. Chapter 9 says to read that curve.
# The LSTM was still improving at epoch 6, so the cap is 12 rather than 6.
PATIENCE = 3

# Word-count features for Naive Bayes and logistic regression (Chapter 1).
COUNT_MAX_FEATURES = 12000
# MAP smoothing: m virtual counts. Chapter 1 uses m to fix zero counts.
NB_M = 1.0
