# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 individual project. Team leader and sole member: **Shaun Gulati**.

The project trains a CNN and a bidirectional LSTM to predict which model wrote a response. The four models are Bard, Falcon-40B-Instruct, GPT-4, and Llama-2-70B-Chat, taken from the public [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset (MIT license). Research questions RQ1 and RQ2 are required; RQ3 and RQ4 are the extra-credit analyses.

Shaun is not presenting, so there are no slides. The write-up is `report/report.pdf`.

## What is in the repo

| Path | Role |
| --- | --- |
| `src/prepare_data.py` | Download UltraFeedback (if needed), filter, label domains, balance, split |
| `src/domains.py` | Rules that assign each prompt to code, reasoning, qa, or writing |
| `src/models.py` | Text CNN and BiLSTM |
| `src/train.py` | Seeded training loop, early stopping, metrics |
| `src/run_all.py` | RQ1–RQ4, TF-IDF baseline, figures |
| `src/features.py` | Length, lexical diversity, punctuation, markdown |
| `data/processed/samples.csv.gz` | The exact table the experiments used |
| `data/processed/summary.json` | Filter counts and class balance |
| `results/metrics.json` | Every reported test number |
| `results/figures/` | Confusion matrices, curves, RQ plots |
| `report/report.pdf` | Project report |

Raw UltraFeedback JSONL files are downloaded into `data/raw/ultrafeedback/` and are gitignored. The processed subset is committed so the report can be regenerated without another download.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.prepare_data   # optional if samples.csv.gz is already present
python -m src.run_all        # CPU, on the order of 30–45 minutes
cd report && pdflatex report.tex && pdflatex report.tex
```

`src/run_all.py` overwrites `results/`. Fixed seed: **42**. Device: CPU. Early stopping watches validation macro-F1 (patience 2, at most 6 epochs) and reloads the best epoch before the test set is scored.

## Dataset, in short

UltraFeedback queried a pool of models with prompts from ShareGPT, UltraChat, Evol-Instruct, FLAN, TruthfulQA, and FalseQA, four models per prompt. This project keeps one model from each of four families:

- Bard (Google)
- Falcon-40B-Instruct (TII)
- GPT-4 (OpenAI)
- Llama-2-70B-Chat (Meta)

FLAN subsets that contain no Bard completions (`flan_v2_niv2`, `flan_v2_p3`, `flan_v2_flan2021`) are dropped so a prompt's source cannot give Bard away. Each model is then downsampled to the same count inside each domain (at most 700). Splits are by prompt, 70/15/15, stratified by domain, so no prompt appears in more than one split. Details and the domain rules are in the report.

## Models

- **CNN.** Kim (2014) style: embedding size 128, convolutions of width 3, 4, and 5 with 96 filters, max-pool, dropout 0.5.
- **RNN.** One-layer bidirectional LSTM, hidden size 64 per direction, dropout 0.3, final states concatenated.
- **Baseline.** Word/bigram TF-IDF (20k features) and logistic regression.

Sequences are lowercased, truncated to 128 tokens, and the vocabulary (12k) is fit on the training split only. For the input+output view, the instruction is capped at 40 tokens and the response at 87 so the response is not truncated away.
