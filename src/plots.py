"""Figures for the report. All paths are created by the caller."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.autofmt_xdate(rotation=25)
    _save(fig, path)


def training_curves(histories: dict[str, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for name, rows in histories.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["train_loss"] for r in rows], marker="o", label=name)
        axes[1].plot(epochs, [r["val_macro_f1"] for r in rows], marker="o", label=name)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_title("Validation average F")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average F")
    axes[1].set_ylim(0, 1)
    axes[0].legend()
    axes[1].legend()
    _save(fig, path)


def grouped_bars(rows: list[dict], title: str, path: Path, ylabel: str = "Average F") -> None:
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    sns.barplot(data=frame, x="setting", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title)
    ax.axhline(0.25, color="gray", ls="--", lw=1, label="chance (4 classes)")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def feature_boxes(feats: pd.DataFrame, path: Path) -> None:
    melted = feats.melt(
        id_vars=["model"],
        value_vars=["n_words", "ttr_50", "punct_per_word", "markdown_hits"],
        var_name="feature",
        value_name="value",
    )
    # Cap word-count axis visually; raw values stay in the summary table.
    melted.loc[melted["feature"] == "n_words", "value"] = melted.loc[
        melted["feature"] == "n_words", "value"
    ].clip(upper=800)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    names = ["n_words", "ttr_50", "punct_per_word", "markdown_hits"]
    titles = [
        "Response length (words, clipped at 800)",
        "Lexical diversity (TTR on first 50 words)",
        "Punctuation marks per word",
        "Markdown hits (fences, headers, bullets, bold)",
    ]
    for ax, name, title in zip(axes.ravel(), names, titles):
        part = melted[melted["feature"] == name]
        sns.boxplot(data=part, x="model", y="value", ax=ax, fliersize=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def balance_heatmap(counts: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title("Examples by domain and model (after balancing)")
    ax.set_ylabel("Domain")
    _save(fig, path)
