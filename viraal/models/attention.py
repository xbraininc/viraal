
from typing import Iterator, List, Dict

import torch
from torch import nn

from allennlp.modules.attention import BilinearAttention

class AttentionClassifier(nn.Module):
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

        bidir_mul = 2 if bidirectional else 1

        self.attention = BilinearAttention(vector_dim=hidden_size*bidir_mul, matrix_dim=hidden_size*bidir_mul)

        self.output2label = torch.nn.Linear(2*bidir_mul*hidden_size,
                                            vocab_size)

    def forward(self,
                embeddings: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        # Notation: Batch (B), Max seq length (T), Hidden size (H), Number of labels (L)
        output, _ = self.encoder(embeddings) # output : (B,T,H)
        last_output = output[range(embeddings.size(0)), mask.sum(dim=1)-1] # last_output : (B,H)
        attention_weights = self.attention(last_output, output) # attention_weights : (B,T)
        context  = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1) 
        # context : (B,1,T)*(B,T,H) = (B,1,H) --squeeze(1)--> (B,H)
        
        concated = torch.cat([last_output, context], dim=1) # concated : (B,2*H) 
        logits = self.output2label(concated) # logits : (B,L)

        return logits
        



