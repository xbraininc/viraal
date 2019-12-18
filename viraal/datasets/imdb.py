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
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)

class ImdbDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 max_instances: int = None,
                 local_seed: int = None) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_instances = max_instances
        self._local_seed = local_seed or 42
        self._rng = random.Random(self._local_seed)

    @overrides
    def _read(self, file_path):
        pos_dir = osp.join(file_path, 'pos')
        neg_dir = osp.join(file_path, 'neg')

        path = set(chain(glob(osp.join(pos_dir,'*.txt')),
                     glob(osp.join(neg_dir,'*.txt'))))

        for p in self._rng.sample(path, self._max_instances):
            with open(p) as file:
                yield self.text_to_instance(file.read(), 0 if 'neg' in str(p) else 1)

    @overrides
    def text_to_instance(self, string: str, label: int) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string)
        fields['sentence'] = TextField(tokens, self._token_indexers)
        fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)