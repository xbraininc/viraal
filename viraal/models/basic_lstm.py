
from typing import Iterator, List, Dict

import torch
from torch import nn
import torch.nn.functional as F

from allennlp.models import Model

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary

class VATTextClassifier(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 word_embedder : BasicTextFieldEmbedder, 
                 vocab : Vocabulary,
                 vat_loss : VatLoss) -> None:

        self.encoder = nn.LSTM(input_size=word_embedder.get_output_dim(), 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional,
                               batch_first=True)
        self.vocab = vocab
        self.vat_loss = vat_loss

        self.hidden2label = torch.Linear(encoder.get_output_dim(), 
                                         vocab.get_vocab_size('labels'))

    def forward(self, 
                sentence: Dict[str, torch.Tensor] = None,
                label: torch.Tensor = None,
                labeled: List = None,
                embeddings: torch.Tensor = None,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        if embeddings is None and sentence is None:
            raise ValueError("forward call should be supplied with sentence or embeddings")
        if embeddings is not None and mask is None:
            raise ValueError("forward call should be supplied with mask along with embeddings")
        
        mask = mask or get_text_field_mask(sentence)
        embeddings = embeddings or self.token_embedding(sentence)

        hidden, _ = self.encoder(embeddings, mask)
        hidden = hidden[range(embeddings.size(0)), mask.sum(dim=1)-1]
        logits = self.hidden2label(hidden)
        
        if labeled is not None and any(labeled):
            ce = F.cross_entropy(logits[labeled], label[labeled], reduction='none')
            ce_loss = ce.mean()
        if self.training:
            ce_loss.backward()

        if self.vat_loss is not None:
            vat = self.vat_loss(self.forward_from_embeddings, logits, embeddings)
            vat_loss = vat.mean()
            if self.training:
                vat_loss.backward()

        return {
            'logits' : logits,
            'ce' : ce,
            'vat' : vat
        }
        



