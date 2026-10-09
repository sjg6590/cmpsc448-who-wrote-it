"""Required classifiers: a Kim-style CNN and a one-layer BiLSTM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CNN_DROPOUT,
    CNN_FILTERS,
    CNN_KERNELS,
    EMBED_DIM,
    RNN_DROPOUT,
    RNN_HIDDEN,
)


class TextCNN(nn.Module):
    """Convolution over word embeddings with max-pooling, following Kim (2014)."""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(EMBED_DIM, CNN_FILTERS, k) for k in CNN_KERNELS]
        )
        self.dropout = nn.Dropout(CNN_DROPOUT)
        self.fc = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, length]
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(embedded))
            pooled.append(F.max_pool1d(hidden, kernel_size=hidden.size(2)).squeeze(2))
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class TextRNN(nn.Module):
    """Bidirectional LSTM. The two final hidden states are concatenated."""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            RNN_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(RNN_DROPOUT)
        self.fc = nn.Linear(RNN_HIDDEN * 2, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        lengths = (token_ids != 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: [2, batch, hidden]
        joined = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.fc(self.dropout(joined))


def build_model(name: str, vocab_size: int, num_classes: int) -> nn.Module:
    if name == "cnn":
        return TextCNN(vocab_size, num_classes)
    if name == "rnn":
        return TextRNN(vocab_size, num_classes)
    raise ValueError(name)
