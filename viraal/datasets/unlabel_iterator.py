import logging
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.fields import MetadataField
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

def unlabel_instances(instances : List[Instance],
                      label_key : str,
                      labeled_part : float,
                      label_min : int,
                      local_seed : int):
    """
    Simulates unlabeled data by adding a field 'labeled' to the instances 
    in the list and setting a 'labeled_part' proportion of them to true while 
    keeping at least a minimum of 'label_min' of each label
    
    Arguments:
        instances {List[Instance]} -- List of instances to add labeled field to
        label_key {str} -- the key corresponding to the label in the instance
        labeled_part {float} -- the proportion of instances that are considered labeled
        label_min {int} -- the minimum instances to keep per label
        local_seed {int} -- local seed used to generate the labeled/unlabeled partition
    """
    rng = random.Random(local_seed) #We create a local rng to prevent the order of execution from changing the partition
    instances = rng.sample(instances, len(instances)) #We locally shuffle the instances 

    label_count = {} # We calculate a label count
    for instance in instances:
        label = instance.fields[label_key].label
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1

    label_to_keep = {} # We then calculate the number instances to keep per label
    for label in label_count:
        label_to_keep[label] = min(label_count[label],
                                    max(label_min,
                                        round(label_count[label]*labeled_part)))

    for instance in instances: # We set 'labeled' accordingly
        label = instance.fields[label_key].label
        if label_to_keep[label] > 0:
            instance.add_field('labeled', MetadataField(True))
            label_to_keep[label] -= 1
        else:
            instance.add_field('labeled', MetadataField(False))
            
class UnlabelIterator(DataIterator):
    """
    An iterator that adds an 'unlabeled' field to simulate unlabeled instances on a labeled dataset
    
    Parameters
    ----------
        label_key : str
            The key corresponding to the label in the instances
        labeled_part : float, optional (default=0.0)
            The proportion of instances to simulate as unlabeled (between 0 and 1)
        label_min : int, optional (default = 1)
            Include at least label_min examples of each label in total
        local_seed : int, optional (default = None)
            local_seed to use to generate the labeled/unlabeled partition
        batch_size : int, optional, (default = 32)
            The size of each batch of instances yielded when calling the iterator.
        instances_per_epoch : int, optional, (default = None)
            See :class:`BasicIterator`.
        max_instances_in_memory : int, optional, (default = None)
            See :class:`BasicIterator`.
        maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
            See :class:`BasicIterator`.
        skip_smaller_batches : bool, optional, (default = False)
            When the number of data samples is not dividable by `batch_size`,
            some batches might be smaller than `batch_size`.
            If set to `True`, those smaller batches will be discarded.
    """
    def __init__(self, 
                 label_key : str,
                 labeled_part : float,
                 label_min : int = 0,
                 local_seed : int = None,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False,
            ) -> None:

        super().__init__(
            cache_instances=cache_instances,
            track_epoch=track_epoch,
            batch_size=batch_size,
            instances_per_epoch=instances_per_epoch,
            max_instances_in_memory=max_instances_in_memory,
            maximum_samples_per_batch=maximum_samples_per_batch,
        )
        self._label_key = label_key
        self._labeled_part = labeled_part
        self._label_min = label_min
        self._local_seed = local_seed or random.randint(0,1e6)

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        logger.info('Using unlabeling local seed : %s', self._local_seed)
        for instance_list in self._memory_sized_lists(instances):
            
            unlabel_instances(instance_list, self._label_key, self._labeled_part, self._label_min, self._local_seed)

            if shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(
                    batch_instances, excess
                ):
                    batch = Batch(possibly_smaller_batches)
                    yield batch
            if excess:
                yield Batch(excess)
