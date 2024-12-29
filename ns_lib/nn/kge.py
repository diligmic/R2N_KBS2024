import math

import keras.layers
import tensorflow as tf
from keras.layers import Layer
import abc
from typing import List, Tuple
from collections import OrderedDict
from ns_lib.logic.commons import Predicate, FOL, Domain
from abc import ABCMeta, abstractmethod


class AtomEmbeddingLayer(Layer):
    def __init__(self, embedder_class,
                 atom_embedding_size,
                 predicates: List[Predicate],
                 regularization=0.0,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__()
        self.atom_embedding_size = atom_embedding_size
        self.regularization = regularization
        self.regularization_n3 = 0.0
        self.predicates = predicates
        self.embedders = {}
        self.embedder_class = embedder_class

        for predicate in self.predicates:
            self.embedders[predicate.name] = self.embedder_class(
                atom_embedding_size=self.atom_embedding_size,
                regularization=regularization,
                dropout_rate=dropout_rate,
                **kwargs)
            # This shoud not be needed.
            #for l in self.embedders[predicate.name].losses:
            #    self.add_loss(l)

    def create_tuples(self, constants_embeddings,
                      tuples_tensor, predicate_domains):
        tuple_features_per_predicate = []
        for i,domain in enumerate(predicate_domains):
            constants = tf.gather(constants_embeddings[domain.name],
                                  tf.cast(tuples_tensor[..., i], tf.int32),
                                  axis=0)
            tuple_features_per_predicate.append(constants)
        # Returns shape (batch_size, predicate_arity, constant_embedding_size)
        return tf.stack(tuple_features_per_predicate, axis=-2)

    def call(self, inputs, **kwargs):
        # This is a GNN-like interface, constants embeddings (dictionary with
        # domain as keys) and atoms (dictionary with predicates as keys and
        # tuples of constants as values).
        constants_embeddings, predicate_to_constant_tuples = inputs

        # Now we embed the tuples in input to the predicates by stacking their
        # constants embeddings.
        tuple_features = {}
        for predicate in self.predicates:
            if predicate.name not in predicate_to_constant_tuples:
                continue

            def GetZeros(size: int):
                def Fn():
                    arity: int = len(predicate.domains)
                    return  tf.zeros([0, arity, size], dtype=tf.float32)
                return Fn
            def GetTuples():
                return self.create_tuples(
                    constants_embeddings=constants_embeddings,
                    tuples_tensor=predicate_to_constant_tuples[predicate.name],
                    predicate_domains=predicate.domains)
            tuple_features[predicate.name] = tf.cond(
                tf.size(predicate_to_constant_tuples[predicate.name]) == 0,
                GetZeros(self.embedders[predicate.name].input_size()),
                GetTuples)

        # Now we embed the tuples (per predicate) using the dynamically defined atom_embedders
        # Each element has shape (B,atom_emb_size).
        predicate_atoms2embeddings = []  # do not use list comprehansion, causes out os scope errors
        for predicate in self.predicates:
            embeddings = self.embedders[predicate.name](
                tuple_features[predicate.name])
            predicate_atoms2embeddings.append(embeddings)

        # Put all the atoms of all the predicates in a unique dense tensor
        embeddings = tf.concat(predicate_atoms2embeddings, axis=0)

        # Specify the last dimension size for the following layers
        embeddings = tf.reshape(embeddings, [-1, self.atom_embedding_size])
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 * tf.reduce_sum(
                abs_embeddings * abs_embeddings * abs_embeddings))
        return embeddings

############################################
# KGE=Layer interface.
class KGELayer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def input_size(self):
        pass

    @property
    @abstractmethod
    def output_size(self):
        pass

############################################
class DistMult(KGELayer, Layer):
    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self.atom_embedding_size = atom_embedding_size

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        p_embeddings, c_embeddings = inputs
        p_embeddings = self.dropout_layer(p_embeddings)
        c_embeddings = self.dropout_layer(c_embeddings)
        embeddings = p_embeddings * tf.reduce_prod(c_embeddings, axis=1)
        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(embeddings))
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 *
                          tf.nn.l2_loss(abs_embeddings))
        return embeddings

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.reduce_sum(inputs, axis=-1)
            outputs = tf.nn.sigmoid(outputs)
            return outputs

        return __internal__

###########################################
class TransE(KGELayer, Layer):

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            # outputs = 1.0 - 2.0 * (tf.nn.sigmoid(tf.reduce_mean(tf.square(inputs), axis=-1)) - 0.5)
            outputs = -tf.reduce_mean(tf.square(inputs), axis=-1)
            return outputs
        return __internal__


    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.atom_embedding_size = atom_embedding_size

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        p_embeddings, c_embeddings = inputs
        p_embeddings = self.dropout_layer(p_embeddings)
        c_embeddings = self.dropout_layer(c_embeddings)

        #head = tf.squeeze(tf.gather(params=c_embeddings, indices=[0], axis=1),axis=1)
        #tail = tf.squeeze(tf.gather(params=c_embeddings, indices=[1], axis=1), axis=1)
        head = c_embeddings[..., 0, :]
        tail = c_embeddings[..., 1, :]

        embeddings = p_embeddings + head - tail

        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(embeddings))
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 *
                          tf.nn.l2_loss(abs_embeddings))
        return embeddings

###########################################
class ModE(KGELayer, Layer):

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.exp(-tf.norm(inputs, axis=-1))
            return outputs
        return __internal__

    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self.atom_embedding_size = atom_embedding_size

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        p_embeddings, c_embeddings = inputs
        p_embeddings = self.dropout_layer(p_embeddings)
        c_embeddings = self.dropout_layer(c_embeddings)

        #head = tf.squeeze(tf.gather(params=c_embeddings, indices=[0], axis=1),axis=1)
        #tail = tf.squeeze(tf.gather(params=c_embeddings, indices=[1], axis=1), axis=1)
        head = c_embeddings[..., 0, :]
        tail = c_embeddings[..., 1, :]

        embeddings = p_embeddings * head - tail

        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(embeddings))
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 *
                          tf.nn.l2_loss(abs_embeddings))
        return embeddings


class ComplEx(KGELayer, Layer):

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.reduce_sum(inputs, axis=-1)
            outputs = tf.nn.sigmoid(outputs)
            return outputs
        return __internal__

    def __init__(self, atom_embedding_size,
                 regularization=0.0, regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.atom_embedding_size = atom_embedding_size
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    def input_size(self):
        return 2 * self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        p_embeddings, c_embeddings = inputs
        p_embeddings = self.dropout_layer(p_embeddings)
        c_embeddings = self.dropout_layer(c_embeddings)
        Rr, Ri = tf.split(p_embeddings, 2, axis=1)

        tf.debugging.assert_equal(tf.shape(Ri)[-1], tf.shape(Rr)[-1])

        head = c_embeddings[..., 0, :]
        tail = c_embeddings[..., 1, :]

        h_r, h_i = tf.split(head, 2, axis=1)
        t_r, t_i = tf.split(tail, 2, axis=1)

        e1 = h_r * t_r * Rr
        e2 = h_i * t_i * Rr
        e3 = h_r * t_i * Ri
        e4 = h_i * t_r * Ri
        embeddings = e1 + e2 + e3 - e4
        if self.regularization > 0.0:
             self.add_loss(self.regularization * (tf.nn.l2_loss(Rr) + tf.nn.l2_loss(Ri)))
        if self.regularization_n3 > 0.0:
            abs_head = tf.math.abs(head)
            abs_tail = tf.math.abs(tail)
            abs_R = tf.math.abs(tf.concat([Rr,Ri], axis=0))
            self.add_loss(self.regularization_n3 * (
                tf.reduce_sum(abs_head * abs_head * abs_head) +
                tf.reduce_sum(abs_tail * abs_tail * abs_tail) +
                tf.reduce_sum(abs_R * abs_R * abs_R)))
        return embeddings


class RotatE(KGELayer, Layer):
    """`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_ (RotatE), which defines each relation as a rotation from the source entity to the target entity in the complex vector space.
    https://openreview.net/forum?id=HkgEQnRqYQ
    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
    """
    # Class attributes.
    epsilon = 0.5
    margin = 6.0  # also called gamma

    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        assert atom_embedding_size > 0
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        if dropout_rate > 0.0:
            self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        else:
            self.dropout_layer = tf.identity

        self.atom_embedding_size = atom_embedding_size
        self.embedding_range = (self.margin + self.epsilon) / (
            atom_embedding_size)
        self.norm_factor = math.pi / self.embedding_range

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = cls.margin - tf.reduce_sum(inputs, axis=-1)  # B,1
            outputs = tf.sigmoid(outputs)
            return outputs
        return __internal__

    def input_size(self):
        return 2 * self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs):
        """Calculating the score of triples.
        The formula for calculating the score is :math:`\margin - \|h \circ r - t\|`
        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
        Returns:
            embeddings: The output embeddings (B, atom_embedding_size)
        """
        p_embeddings, c_embeddings = inputs
        p_embeddings = self.dropout_layer(p_embeddings)
        c_embeddings = self.dropout_layer(c_embeddings)

        head = c_embeddings[..., 0, :]
        tail = c_embeddings[..., 1, :]

        if tf.shape(head)[0] == 0 or tf.shape(tail)[0] == 0:
            return tf.zeros([0, self.atom_embedding_size])

        re_head, im_head = tf.split(head, 2, axis=-1)
        re_tail, im_tail = tf.split(tail, 2, axis=-1)

        phase_relation = p_embeddings * self.norm_factor
        re_relation = tf.math.cos(phase_relation)
        im_relation = tf.math.sin(phase_relation)
        #hadamard = tf.multiply(tf.complex(re_head, im_head),
        #                       tf.complex(re_relation, im_relation))
        #complex_tail = tf.complex(re_tail, im_tail)
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
        # embeddings = re_score - im_score  # (B,atom_emb_size)
        embeddings = tf.math.sqrt(tf.maximum(
            1e-9,
            re_score * re_score + im_score * im_score))  # (B,atom_emb_size)
        # embeddings = hadamard - complex_tail
        #if self.regularization > 0.0:
        #    self.add_loss(self.regularization * tf.nn.l2_loss(p_embeddings))
        #if self.regularization_n3 > 0.0:
        #    self.add_loss(self.regularization_n3 * tf.nn.l2_loss(embeddings))
        return embeddings

############################################
class Tucker(KGELayer, Layer):
    def __init__(self, atom_embedding_size: int,
                 relation_embedding_size: int=None,
                 regularization: float=0.0,
                 dropout_rate: float=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.atom_embedding_size = atom_embedding_size
        self.relation_embedding_size = (atom_embedding_size
                                        if relation_embedding_size is None else
                                        relation_embedding_size)
        init = tf.initializers.GlorotUniform()
        self.W = tf.Variable(
            initial_value=init(shape=[self.atom_embedding_size,
                                      self.atom_embedding_size,
                                      self.relation_embedding_size]),
            name='Tucker')

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return 1

    def call(self, inputs, **kwargs):
        p_embeddings, c_embeddings = inputs
        p_embeddings = self.dropout_layer(p_embeddings)
        c_embeddings = self.dropout_layer(c_embeddings)

        head = c_embeddings[..., 0, :]  # BE
        tail = c_embeddings[..., 1, :]  # BE
        W = self.dropout_layer(self.W)  # EER
        W1 = tf.einsum('bi,itr->btr', head, W)   # BER
        W2 = tf.einsum('bi,bir->br', tail, W1)   # BR
        embeddings = tf.expand_dims(tf.einsum('i,i->b', p_embeddings, W2), axis=-1)  # B1
        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(self.W))
        return embeddings

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.nn.sigmoid(outputs)
            return outputs

        return __internal__


####################################
def KGEFactory(name: str,
               atom_embedding_size: int,
               regularization: float,
               dropout_rate: float,
               relation_embedding_size: int=None):

  relation_embedding_size = (relation_embedding_size
                             if relation_embedding_size is not None
                             else atom_embedding_size)
  if name.casefold() == 'complex':
    return ComplEx(atom_embedding_size, regularization, dropout_rate), ComplEx.output_layer()

  elif name.casefold() == 'distmult':
    return DistMult(atom_embedding_size, regularization, dropout_rate), \
           DistMult.output_layer()

  elif name.casefold() == 'tucker':
    return Tucker(atom_embedding_size, relation_embedding_size,
                  regularization, dropout_rate), Tucker.output_layer()

  elif name.casefold() == 'transe':
    return TransE(atom_embedding_size, regularization, dropout_rate), TransE.output_layer()

  elif name.casefold() == 'rotate':
    return RotatE(atom_embedding_size, regularization, dropout_rate), RotatE.output_layer()

  elif name.casefold() == 'mode':
    return ModE(atom_embedding_size, regularization, dropout_rate), ModE.output_layer()

  else:
    print('Unknown KGE', name, flush=True)
    return None
