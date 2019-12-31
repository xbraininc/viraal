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

class UnlabelInstances:
    """
        Simulates unlabeled data by adding a field 'labeled' to the instances 
        in the list and setting a 'labeled_part' proportion of them to true while 
        keeping at least a minimum of 'label_min' of each label
        
        Arguments:
            label_key {str} -- the key corresponding to the label in the instance
            labeled_part {float} -- the proportion of instances that are considered labeled
            label_min {int} -- the minimum instances to keep per label
            local_seed {int} -- local seed used to generate the labeled/unlabeled partition
    """
    def __init__(self,
                 label_key : str,
                 labeled_part : float,
                 label_min : int):
        self.label_key=label_key
        self.labeled_part=labeled_part
        self.label_min=label_min
        # self.local_seed=local_seed

    def __call__(self, instances : List[Instance]):
        """
        Arguments:
            instances {List[Instance]} -- List of instances to add labeled field to unlabel
        """
        # rng = random.Random(self.local_seed) #We create a local rng to prevent the order of execution from changing the partition
        instances_shuf = random.sample(instances, len(instances)) #We locally shuffle the instances 

        label_count = {} # We calculate a label count
        for instance in instances_shuf:
            label = instance.fields[self.label_key].label
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1

        label_to_keep = {} # We then calculate the number instances_shuf to keep per label
        for label in label_count:
            label_to_keep[label] = min(label_count[label],
                                        max(self.label_min,
                                            round(label_count[label]*self.labeled_part)))

        for instance in instances_shuf: # We set 'labeled' accordingly
            label = instance.fields[self.label_key].label
            if label_to_keep[label] > 0:
                instance.add_field('labeled', MetadataField(True))
                label_to_keep[label] -= 1
            else:
                instance.add_field('labeled', MetadataField(False))
