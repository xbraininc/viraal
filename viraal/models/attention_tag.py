
from typing import Iterator, List, Dict

import torch
from torch import nn

from allennlp.modules.attention import BilinearAttention

class AttentionTagClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 tag_embedding_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 tag_vocab_size : int) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size=input_size, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional,
                               batch_first=True)

        bidir_mul = 2 if bidirectional else 1

        self.attention = BilinearAttention(vector_dim=hidden_size*bidir_mul, matrix_dim=hidden_size*bidir_mul)

        self.tag_embed = nn.Embedding(num_embeddings=tag_vocab_size, embedding_dim=tag_embedding_size)
        self.decoder = nn.LSTM(input_size=2*bidir_mul*hidden_size+tag_embedding_size, 
                               hidden_size=bidir_mul*hidden_size, 
                               num_layers=num_layers, 
                               bidirectional=False,
                               batch_first=True)

        self.output2tag = torch.nn.Linear(2*bidir_mul*hidden_size,
                                          tag_vocab_size)

    def forward(self,
                embeddings: torch.Tensor = None, #(B,T,H)
                mask: torch.Tensor = None, #(B,T)
                target_tag: torch.Tensor = None #(B,T)
                ) -> torch.Tensor:

        #### Intent
        # Notation: Batch (B), Max seq length (T), Hidden size (H), Number of labels (L), Number of tags (S)
        output, _ = self.encoder(embeddings) # output : (B,T,H)
        output = output*(mask.unsqueeze(2).float())
        last_output = output[range(embeddings.size(0)), mask.sum(dim=1)-1] # last_output : (B,H)
        attention_weights = self.attention(last_output, output)*(mask.float()) # attention_weights : (B,T)
        context  = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1) 
        # context : (B,1,T)*(B,T,H) = (B,1,H) --squeeze(1)--> (B,H)

        ### Tag

        context = context.unsqueeze(1)
        length = embeddings.size(1)
        decode = []
        hidden = None
        batch_size = embeddings.size(0)
        embedded_tag_zer= torch.zeros(batch_size,1,self.hidden_size).to(context.device)

        for i in range(length):  # Input_sequence Output_sequence -Because of the same length
            aligned = output[:,i,:].unsqueeze(1)  # (B,1,H)
            if target_tag is not None and i > 0:
                last_target_tag = target_tag.transpose(0, 1)[i-1].unsqueeze(1) #(B,1)
                last_target_mask = mask.transpose(0, 1)[i-1].unsqueeze(1) #(B,1)
                last_target_input =  (last_target_tag * mask.long()) + (x_input * (~mask).long())
            embedded_tag = self.tag_embed(last_target_input) if i>0 else embedded_tag_zer#(B,1,H)
            # x_input, context, aligned encoder hidden, hidden
            _, hidden = self.decoder(
                torch.cat((embedded_tag, context, aligned), dim=2), hidden)

            concated = torch.cat((hidden[0][0].squeeze(0), context.squeeze(1)), dim=1) # (B,2*H)
            score = self.output2tag(concated) # (B,H)
            decode.append(score)
            # next Context Vector to Attention Calculated by
            attention_weights = self.attention(hidden[0][0], output)*(mask.float()) # (B,H)
            context  = torch.bmm(attention_weights.unsqueeze(1), output)

            _, last_target_input = torch.max(score, 1) # (B,)
            last_target_input = last_target_input.unsqueeze(1) # (B,1)
        # Significant attention! time-step of column-wise concat After, reshape!!
        slot_logits = torch.stack(decode, dim=1)

        return slot_logits
        



