"""Sequence classifiers in the form from Chapter 4.

The CNN is a 1D convolution (width 3, zero padding), ReLU, max-pooling, then a
fully connected layer. The LSTM is many-to-one: the label is read from h_T,
the hidden state after the last real word.

PyTorch's nn.LSTM is the same recurrence as the lecture. With the lecture's
names, each step is

    i_t = sigmoid(x_t U^i + h_{t-1} W^i + b_i)
    f_t = sigmoid(x_t U^f + h_{t-1} W^f + b_f)
    o_t = sigmoid(x_t U^o + h_{t-1} W^o + b_o)
    q_t = tanh(x_t U^q + h_{t-1} W^q + b_q)
    p_t = f_t * p_{t-1} + i_t * q_t
    h_t = o_t * tanh(p_t)

U multiplies the input x_t and W multiplies the previous hidden state.
i, f, o are the input, forget, and output gates. q_t is the candidate.
p_t is the cell state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CNN_FILTERS, CNN_KERNEL, EMBED_DIM, RNN_HIDDEN


class TextCNN(nn.Module):
    """Chapter 4 text CNN: conv, ReLU, max-pool, fully connected."""

    def __init__(self, vocab_size: int, num_classes: int, kernel_sizes: tuple[int, ...] = (CNN_KERNEL,)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        # padding=kernel//2 is the zero padding on both ends from the 1D example.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(EMBED_DIM, CNN_FILTERS, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.fc = nn.Linear(CNN_FILTERS * len(kernel_sizes), num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, length]. Embedding gives one vector per word (Ch6).
        embedded = self.embedding(token_ids).transpose(1, 2)
        mask = token_ids.ne(0).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(embedded))
            # Do not let padding positions win the max-pool.
            if hidden.size(2) != mask.size(2):
                hidden = hidden[:, :, : mask.size(2)]
            hidden = hidden.masked_fill(~mask, -1e9)
            pooled.append(hidden.max(dim=2).values)
        return self.fc(torch.cat(pooled, dim=1))


class TextLSTM(nn.Module):
    """Unidirectional LSTM. Many-to-one: scores come from h_T only."""

    def __init__(self, vocab_size: int, num_classes: int, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            RNN_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(RNN_HIDDEN * directions, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        # Packing drops the pad steps so h_T is the last real word, not a pad.
        # The lecture does not discuss padding; this is just so variable-length
        # batches do not change the recurrence.
        lengths = token_ids.ne(0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        if self.bidirectional:
            final = torch.cat([hidden[0], hidden[1]], dim=1)
        else:
            final = hidden[-1]
        return self.fc(final)


def build_model(name: str, vocab_size: int, num_classes: int) -> nn.Module:
    if name == "cnn":
        return TextCNN(vocab_size, num_classes, kernel_sizes=(CNN_KERNEL,))
    if name == "cnn_multi":
        # Several filter widths. Chapter 4 only shows a width-3 filter.
        return TextCNN(vocab_size, num_classes, kernel_sizes=(3, 4, 5))
    if name == "lstm":
        return TextLSTM(vocab_size, num_classes, bidirectional=False)
    if name == "bilstm":
        # Chapter 4 says a one-direction RNN sees only one side. This is the
        # extra model that reads the sequence both ways.
        return TextLSTM(vocab_size, num_classes, bidirectional=True)
    raise ValueError(name)
