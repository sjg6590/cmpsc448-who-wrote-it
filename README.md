# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 individual project. Team leader and sole member: **Shaun Gulati**. I am not presenting.

The write-up is `report/report.pdf`. It uses the models from lecture: a CNN (convolution, ReLU, max-pooling) and a one-direction LSTM (the label comes from the last hidden state). Naive Bayes and logistic regression are the Chapter 1 baselines. The numbers are in `results/metrics.json`.

## What is in the repo

| Path | What it is |
| --- | --- |
| `src/prepare_data.py` | Download UltraFeedback if needed, filter, label tasks, balance, split |
| `src/domains.py` | Rules that mark a prompt as code, reasoning, qa, or writing |
| `src/models.py` | The CNN and the LSTM |
| `src/train.py` | Adam, negative log-likelihood, validation accuracy picks the epoch |
| `src/run_all.py` | RQ1 through RQ4 |
| `data/processed/samples.csv.gz` | The table the experiments used |
| `results/metrics.json` | The test numbers in the report |
| `results/figures/` | Plots |
| `report/report.pdf` | The report |

Raw UltraFeedback files go in `data/raw/ultrafeedback/` and are not committed.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.prepare_data   # skip this if samples.csv.gz is already there
python -m src.run_all        # CPU. Expect on the order of an hour.
cd report && pdflatex report.tex && pdflatex report.tex
```

Seed is 42. Batch size is 64. Learning rate is 0.001. At most 12 epochs. Training stops if validation accuracy does not improve for 3 epochs. The test set is scored once, with the best validation epoch.

## Data, short version

UltraFeedback is MIT licensed. I kept Bard, Falcon-40B-Instruct, GPT-4, and Llama-2-70B-Chat. I dropped three FLAN subsets that never queried Bard, so the prompt template cannot give Bard away. Each model has the same count inside each task (at most 700). The split is by prompt, 70/15/15. No prompt is in more than one split.

The main CNN uses one filter width, 3, as in the 1D convolution example from class. The main RNN is not bidirectional. A bidirectional LSTM and a CNN with widths 3, 4, and 5 are extra comparisons on the output-only task only.
