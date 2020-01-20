from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class IobIntentDatasetReader(DatasetReader):
    """Reads an IOB file containging ATIS flight sentences in the following formatting:
      word1 word2 word3\tTAG1 TAG2 TAG3 INTENT
    
    Arguments:
        lazy : ``bool`` (optional, default=False)
            Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
            take longer per batch.  This also allows training with datasets that are too large to fit
            in memory.
    """
    def __init__(self,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

    @overrides
    def _read(self, file_path : str):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for idx, line in enumerate(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                text, labels = line.split('\t')
                labels = labels.split()

                intent = labels[-1]
            
                yield self.text_to_instance(text, intent, idx)

    @overrides
    def text_to_instance(self,
                         text : str,
                         intent : str = None,
                         idx : int = None) -> Instance:
        
        fields = {}

        tokenized_text = [Token(token) for token in text.split()]
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        text_field = TextField(tokenized_text, token_indexers)
        fields['sentence'] = text_field
        
        if intent is not None:
            fields['label'] = LabelField(intent)
        if idx is not None:
            fields['idx'] = MetadataField(idx)

        return Instance(fields)
                
