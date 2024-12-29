from keras.layers import Dense, Embedding, Layer
from keras.models import Sequential
from keras.regularizers import L2
from typing import Dict, List, Tuple
import tensorflow as tf
import tensorflow_probability as tfp

from ns_lib.logic.commons import Domain

class ConstantEmbeddings(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain],
                 constant_embedding_sizes_per_domain: Dict[str, int],
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        for domain in domains:
            if has_features:
                # This should be replaced with the actual embedder,
                # make this a factory function call.
                self.embedder[domain.name] = Sequential([
                    Dense(constant_embedding_sizes_per_domain[domain.name],
                          kernel_regularizer=L2(regularization))])
            else:
                self.embedder[domain.name] = Embedding(
                    len(domain.constants),
                    constant_embedding_sizes_per_domain[domain.name],
                    embeddings_regularizer=L2(regularization))

    # domain_inputs is Dict domain->tensor of idx
    def call(self, domain_inputs: Dict[str, tf.Tensor], **kwargs):
        domain_features = {}
        for domain in self.domains:
            domain_features[domain.name] = self.embedder[domain.name](
                domain_inputs[domain.name])
        return domain_features

class PredicateEmbeddings(Layer):
    """Calls the predicate embedders."""
    def __init__(self, predicates: List[str],
                 predicate_embedding_size: int,
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        #self.predicates = predicates
        # self.predicate2index = {p:i for i,p in enumerate(predicates)}
        #self.table = tf.lookup.StaticHashTable(
        #    tf.lookup.KeyValueTensorInitializer(self.predicate2index.keys(),
        #                                        self.predicate2index.values()),
        #    default_value=-1)
        #self.regularization = regularization
        if has_features:
            # This should be replaced with the actual embedder,
            # make this a factory function call.
            self.embedder = Sequential([
                Dense(len(predicates), kernel_regularizer=L2(regularization))])
        else:
            self.embedder = Embedding(len(predicates), predicate_embedding_size,
                                      embeddings_regularizer=L2(regularization))

    # Inputs is tensor of predicate idx.
    # Output is tensor of embeddings of each predicate.
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.embedder(inputs)  #BE

class AdaptiveConstantEmbeddings(Layer):
    def __init__(self, domains: List[Domain],
                 constant_embedder: ConstantEmbeddings,
                 constant_embedding_size: int,
                 num_adaptive_constants: int,
                 dot_product: bool=False):
        super().__init__()
        self.adaptive_constant2relevance = {}
        self.constant_embedder = constant_embedder
        self.num_adaptive_constants = num_adaptive_constants
        self.domains = domains
        self.dot_product = dot_product
        for domain in domains:
            if not dot_product:
                self.adaptive_constant2relevance[domain.name] = (
                    Sequential([
                        Embedding(num_adaptive_constants, constant_embedding_size),
                    ]))
            else:
                self.adaptive_constant2relevance[domain.name] = (
                    Sequential([
                        Embedding(num_adaptive_constants, constant_embedding_size),
                        # Dense(len(domain.constants), activation=None),
                    ]))

    def call(self, domain_inputs, **kwargs):
        # TODO: check with Mike / Beppe
        # emb_inputs = self.constant_embedder(domain_inputs)

        constant_ranges = {domain.name:tf.range(0, len(domain.constants))  #CE
                            for domain in self.domains}
        constant_embeddings = self.constant_embedder(constant_ranges)

        embedder_outputs = {}
        for domain in self.domains:
            # Only constants that are outside of domain bounds are
            # adaptive and should be embedded.
            # We zero out non-adaptive constants, which will be anyway
            # discarded later by the caller.
            embedder_inputs = tf.math.maximum(
                0, domain_inputs[domain.name] - len(domain.constants))

            # TODO: check with Mike / Beppe
            if self.dot_product:
                adaptive_emb = self.adaptive_constant2relevance[domain.name](embedder_inputs)
                emb = constant_embeddings[domain.name]  #CE
                constant2relevance = tf.tensordot(adaptive_emb, tf.transpose(emb), axes=1)
                max_value = tf.reduce_max(constant2relevance, axis=-1, keepdims=True)
                mask = tf.cast(tf.equal(constant2relevance, max_value), adaptive_emb.dtype)
                embedder_outputs[domain.name] = tf.linalg.matmul(mask, emb)

            else:
                # Computes the distribution over the constants.
                logits = self.adaptive_constant2relevance[domain.name](
                    embedder_inputs)
                dist = tfp.distributions.Logistic(logits, 1.0)
                sample = tf.sigmoid(dist.sample())  # CC
                embedder_outputs[domain.name] = tf.linalg.matmul(
                    sample, constant_embeddings[domain.name])  #CE
                #deb = tf.math.argmax(embedder_outputs[domain.name], axis=-1)
                #tf.print('Selected constants domain', domain.name, deb)

        return embedder_outputs

class DomainWiseMLP(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain], constant_embedding_sizes_per_domain):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        for domain in domains:
            m = Sequential()
            m.add(Dense(constant_embedding_sizes_per_domain[domain.name],activation = "relu"))
            self.embedder[domain.name] = m

    def call(self, domain_inputs, **kwargs):
        domain_features = {}
        for d in self.domains:
            domain_name = d.name
            domain_features[domain_name] = self.embedder[domain_name](domain_inputs[domain_name])
        return domain_features

class ExplicitDomainEmbedders(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of the single domains."""
    def __init__(self, domains: List[Domain], embedders):
        super().__init__()
        self.embedder = embedders
        self.domains = domains

    def call(self, domain_inputs, **kwargs):
        domain_features = {}
        for domain in self.domains:
                domain_features[domain.name] = self.embedder[domain.name](domain_inputs[domain.name])
        return domain_features
