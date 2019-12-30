
from typing import Iterator, List, Dict

import torch
from torch import nn

class BasicLSTMClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 vocab_size : int) -> None:
        super().__init__()

        self.encoder = nn.LSTM(input_size=input_size, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional,
                               batch_first=True)

        self.hidden2label = torch.nn.Linear((2 if bidirectional else 1)*hidden_size,
                                            vocab_size)

    def forward(self,
                embeddings: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        hidden, _ = self.encoder(embeddings)
        hidden = hidden[range(embeddings.size(0)), mask.sum(dim=1)-1]
        logits = self.hidden2label(hidden)

        return logits
        



