from viraal.datasets.imdb import ImdbDatasetReader
from allennlp.data import Vocabulary
import pytest

class TestImdbDatasetReader:
    def setup(self):
        self.reader = ImdbDatasetReader(max_instances=100)

    def test(self):
        train_instances = self.reader.read("/u/home/badr/Corpus/aclImdb/train")
        vocab = Vocabulary.from_instances(train_instances)
        print("Done")
