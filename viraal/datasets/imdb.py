from typing import Dict
import logging
import random

import os.path as osp
from glob import glob
import tarfile
from itertools import chain

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, MetadataField
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)

class ImdbDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 max_length: int = None) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_length = max_length

    @overrides
    def _read(self, file_path):
        pos_dir = osp.join(file_path, 'pos')
        neg_dir = osp.join(file_path, 'neg')

        path = list(chain(glob(osp.join(pos_dir,'*.txt')),
                     glob(osp.join(neg_dir,'*.txt'))))

        # max_instances = max_instances or len(path)

        # for p in self._rng.sample(path, max_instances):
        for i, p in enumerate(path):
            with open(p) as file:
                yield self.text_to_instance(file.read(), 'neg' if 'neg' in str(p) else 'pos', i, max_length=self.max_length)

    @overrides
    def text_to_instance(self, string: str, label: str = None, idx: int = None, max_length: int = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string.lower())[:max_length]
        fields['sentence'] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        if idx is not None:
            fields['idx'] = MetadataField(idx)
        return Instance(fields)