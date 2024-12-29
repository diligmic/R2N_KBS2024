from typing import Dict, Iterable, List, Tuple, Union
import ns_lib as ns
from ns_lib.logic.commons import Atom, FOL, RuleGroundings
import tensorflow as tf
from keras.metrics import Metric
from keras.losses import Loss
import abc
import numpy as np


DomainName = str
ConstantName = str
PredicateName = str

FormulaSignature = str
Label = int
Tensor = Union[np.array, tf.Tensor]
#Queries = Union[List[Atom], List[List[Atom]]]
#Labels = Union[List[Label], List[List[Label]]]


IntId = np.array
AtomId = int
SingleIdFeature = Tensor
FloatFeatures = Tensor
ConstantFeatures = Union[SingleIdFeature, FloatFeatures]
ConstantTuples = Union[List[List[IntId]], Tensor]
AtomIds = Union[List[List[AtomId]], Tensor] # one atom for each tuple above
AtomTuples = Union[List[List[AtomId]], Tensor] # a list of groundings (lists of AtomId) for each formula
# QueriesIds = Union[List[AtomId], List[List[AtomId]], Tensor]


def _from_strings_to_tensors(fol, serializer,
                             queries, labels, engine, ragged,
                             constants_features=None, deterministic=True):

    # Symbolic step
    facts_tuple = tuple(fol.facts)
    queries_tuple = tuple(ns.utils.to_flat(queries))
    if engine is not None:
        if len(queries_tuple) > 100000:
            print('Start grounder on facts:', len(facts_tuple),
                  'queries:', len(queries_tuple), flush=True)
        ground_formulas: Dict[str, RuleGroundings] = engine.ground(
            facts_tuple, queries_tuple, deterministic=deterministic)
        rules = engine.rules
    else:
        ground_formulas = {}
        rules = []
    (domain_to_global, predicate_tuples, groundings, queries) = (
        serializer.serialize(queries=queries,
                             rule_groundings=ground_formulas))

    # CONSTANTS
    input_domains_tf: Dict[DomainName, ConstantFeatures] = {}
    for d in fol.domains:
        if constants_features is not None and d.name in constants_features:
            global_features = constants_features[d.name]
            global_indices = domain_to_global[d.name]
            input_features = tf.gather(global_features, global_indices, axis=0)
            input_domains_tf[d.name] = input_features
        else:
            input_domains_tf[d.name] = tf.constant(domain_to_global[d.name],
                                                   dtype=tf.int32)
    # Creating the input dictionaries (atoms as tuples of domains,
    # atoms as dense ids, formulas as tuples of atoms)
    # Dict[predicate_name, List[Tuple[constants_ids]]]
    input_atoms_tuples_tf: Dict[PredicateName, ConstantTuples] = {
        name:tf.constant(tuples, dtype=tf.int32) if len(tuples) > 0 else
             tf.zeros(shape=(0, fol.name2predicate[name].arity), dtype=tf.int32)
        for name,tuples in predicate_tuples.items()}

    # Dict[formula_id, List[Tuple[atom_ids]]]
    input_formulas_tf: Dict[FormulaSignature, (AtomTuples, AtomTuples)] = {}
    for rule in rules:
        ai = len(rule.body)
        ao = len(rule.head)
        if rule.name in groundings and len(groundings[rule.name]) > 0:
            # adding batch dimension
            input_formulas_tf[rule.name] = (
                tf.constant(groundings[rule.name][0], dtype=tf.int32),
                tf.constant(groundings[rule.name][1], dtype=tf.int32))
        else:
            # empty tensor
            input_formulas_tf[rule.name] = (
                tf.zeros(shape=[0, ai], dtype=tf.int32),
                tf.zeros(shape=[0, ao], dtype=tf.int32))
    # TODO check how to understand if it is a good tensor or need to be ragged.
    if ragged:
        queries = tf.ragged.constant(queries, dtype=tf.int32)
        labels =  tf.ragged.constant(labels, dtype=tf.float32)
    else:
        queries = tf.constant(queries, dtype=tf.int32)
        labels = tf.constant(labels, dtype=tf.float32)

    return (input_domains_tf, input_atoms_tuples_tf,
            input_formulas_tf, queries), labels


class Dataset():

    def __init__(self,
                 queries: List[List[Union[Atom, str, Tuple]]],
                 labels: List[List[Label]],
                 constants_features: Dict[DomainName, Tensor]=None,
                 format: str='functional'):

        self.constants_features = constants_features
        self.queries = [[a if isinstance(a,Tuple) else a.toTuple() if isinstance(a,Atom) else Atom(s=a, format=format).toTuple() for a in q] for q in queries]
        self.labels = labels
        assert len(queries) == len(labels)
        self.format = format


    def __getitem__(self, item):
        queries,labels =  self.queries[item], self.labels[item]
        return queries, labels


    def __len__(self):
        return len(self.queries)


    def _get_batch(self, i, b):

        return self.queries[b*i:b*(i+1)], self.labels[b*i:b*(i+1)], 


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 fol: FOL,
                 serializer,
                 engine=None,
                 deterministic=True,
                 batch_size=None,
                 ragged: bool=False,
                 name= "None"):

        self.dataset = dataset
        self.deterministic = deterministic
        self.fol = fol
        self.engine = engine
        self.serializer = serializer
        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)
        self.name = name

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))


    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):

        queries, labels = self.dataset[b*i:b*(i+1)]
        constants_features = self.dataset.constants_features

        ((X_domains_data,
          A_predicates_data,
          A_rules_data, Q), y) = _from_strings_to_tensors(
            fol=self.fol,
            serializer=self.serializer,
            queries=queries,
            labels=labels,
            engine=self.engine,
            ragged=self.ragged,
            constants_features=constants_features,
            deterministic=self.deterministic)

        return (X_domains_data, A_predicates_data, A_rules_data, Q), y



class DataGeneratorTensor(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 serializer,
                 engine=None,
                 batch_size=None,
                 ragged: bool=False):

        self.dataset = dataset
        self.engine = engine
        self.serializer = serializer
        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))

    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):

        queries, labels = self.dataset[b*i:b*(i+1)]

        lenghts = [len(x) for x in queries]
        flat_queries = tf.constant(ns.utils.to_flat(queries), dtype=tf.int32)
        if self.engine is not None:
            ground_formulas = self.engine.ground(
                tf.constant(self.dataset.facts, dtype=tf.int32),
                flat_queries)
            rules = self.engine.rules
        else:
            ground_formulas = {}
            rules = []

        (input_domains_tf, input_atoms_tuples_tf,
         input_formulas_tf, flat_queries) = self.serializer.serialize(
            queries=flat_queries, rule_groundings=ground_formulas)

        if self.dataset.constants_features is not None:
            for d in self.dataset.domains:
                global_features = self.dataset.constants_features[d.name]
                local_indices = input_domains_tf[d.name]
                global_to_input = tf.gather(global_features, local_indices, axis=0)
                input_domains_tf[d.name] = global_to_input

        if self.ragged:
            queries = tf.RaggedTensor.from_row_lengths(values = flat_queries, row_lengths = lenghts)
            labels = tf.ragged.constant(labels, dtype=tf.float32)
        else:
            queries = tf.constant(queries, dtype=tf.int32)
            labels = tf.constant(labels, dtype=tf.float32)

        return (input_domains_tf, input_atoms_tuples_tf,
                input_formulas_tf, queries), labels


class DataGeneratorTensorFast(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 batch_size=None,
                 ragged: bool=False):

        self.dataset = dataset

        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))

    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):

        q,l =  self.dataset[b*i:b*(i+1)]

        return tf.ragged.constant(q), tf.ragged.constant(l)
