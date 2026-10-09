"""Training and evaluation loops shared by every research question."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, SEED
from src.textutil import Vocab, encode_view, pad_batch, tokenize


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextSplit(Dataset):
    def __init__(self, frame, view: str, vocab: Vocab, label_to_id: dict[str, int], response_col: str = "response"):
        self.frame = frame.reset_index(drop=True)
        self.view = view
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.response_col = response_col

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        ids = encode_view(row["instruction"], row[self.response_col], self.view, self.vocab)
        label = self.label_to_id[row["model"]]
        return ids, label


def _collate(batch):
    seqs, labels = zip(*batch)
    return pad_batch(list(seqs)), torch.tensor(labels, dtype=torch.long)


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=_collate,
    )


def build_vocab(frame, view: str, response_col: str = "response") -> Vocab:
    if view == "output":
        texts = frame[response_col].tolist()
    elif view == "input":
        texts = frame["instruction"].tolist()
    elif view == "both":
        texts = [f"{a}\n{b}" for a, b in zip(frame["instruction"], frame[response_col])]
    else:
        raise ValueError(view)
    return Vocab([tokenize(text) for text in texts])


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, golds = [], []
    for tokens, labels in loader:
        tokens = tokens.to(device)
        logits = model(tokens)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        golds.append(labels.numpy())
    return np.concatenate(golds), np.concatenate(preds)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, id_to_label: dict[int, str]) -> dict:
    labels = list(range(len(id_to_label)))
    names = [id_to_label[i] for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in names
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": names,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    id_to_label: dict[int, str],
    device: torch.device,
) -> tuple[nn.Module, list[dict]]:
    model.to(device)
    # Adam, Chapter 9. No weight decay: the lecture's update is w <- w - alpha * grad.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # CrossEntropyLoss is softmax followed by negative log-likelihood (Chapter 9).
    criterion = nn.CrossEntropyLoss()
    history = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        y_true, y_pred = predict(model, val_loader, device)
        val_metrics = metric_bundle(y_true, y_pred, id_to_label)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"  epoch {epoch}: loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}",
            flush=True,
        )
        # Chapter 1's reported number is accuracy, so that is what picks the epoch.
        score = val_metrics["accuracy"]
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  validation accuracy stopped improving at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in history:
        row["selected"] = row["epoch"] == best_epoch
    return model, history
