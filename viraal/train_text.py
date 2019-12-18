import glob
import os

import hydra
import ometrics
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from ray.tune import Trainable

from hydra.utils import instantiate

from viraal.datasets.imdb import ImdbDatasetReader
from viraal.datasets.unlabel_iterator import UnlabelIterator

class TrainText(Trainable):
    def _setup(self, config):
        self.dataset_reader = instantiate(config.dataset.reader)
        self.iterator = instantiate(config.training.iterator)

        self.train_instances = self.dataset_reader.read(config.dataset.splits.train)
        self.val_instances = self.dataset._reader.read(config.dataset.splits.val)

        self.vocab = Vocabulary.from_instances(self.train_instances)
        self.iterator.index_with(self.vocab)

        token_embedding = Embedding.from_params(self.vocab, config.dataset.embedding)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        vat_loss = 

        self.model = instantiate(config.model, word_embedder=self.word_embeddings, vocab=self.vocab, vat_loss=vat_loss)
        self.optimizer = instantiate(config.training.optimizer, self.model.params())

        
    def _train(self):
        
        self.model.train()
        for batch in self.iterator(self.train_instances):
            output = self.model(**batch)
            self.metrics.update(output, batch)
            self.optimizer.step()
        
        self.metrics.log()
        self.metrics.upload()

    def _save(self, checkpoint_prefix):
        checkpoint = {
            'model' : os.path.join(checkpoint_prefix, f'model_{self.iteration}.th'),
            'vocab' : os.path.join(checkpoint_prefix, 'model.vocab')
        }
        torch.save(self.model, checkpoint['model'])
        self.vocab.save_to_files(checkpoint['vocab'])
        return checkpoint

    def _restore(self, checkpoint):
        self.model = torch.load(checkpoint['model'])
        self.vocab.from_files(checkpoint['vocab'])
