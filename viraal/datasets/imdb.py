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
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        # self._local_seed = local_seed or 42
        # self._rng = random.Random(self._local_seed)

    @overrides
    def _read(self, file_path, max_instances=None):
        pos_dir = osp.join(file_path, 'pos')
        neg_dir = osp.join(file_path, 'neg')

        path = set(chain(glob(osp.join(pos_dir,'*.txt')),
                     glob(osp.join(neg_dir,'*.txt'))))

        # max_instances = max_instances or len(path)

        # for p in self._rng.sample(path, max_instances):
        for p in path:
            with open(p) as file:
                yield self.text_to_instance(file.read(), 'neg' if 'neg' in str(p) else 'pos')

    @overrides
    def text_to_instance(self, string: str, label: int) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string.lower())
        fields['sentence'] = TextField(tokens, self._token_indexers)
        fields['label'] = LabelField(label)
        return Instance(fields)