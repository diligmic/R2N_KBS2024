from keras import Model, regularizers, models
from keras.layers import Dense, Dropout, Lambda, Layer
import ns_lib as ns
import tensorflow as tf
from ns_lib.nn.constant_embedding import *
from ns_lib.nn.reasoning import *
from ns_lib.nn.kge import KGEFactory, KGELayer
from ns_lib.logic import FOL, Rule
from ns_lib.logic.semantics import GodelTNorm
from typing import Dict, List, Union
import tensorflow_probability as tfp


class KGEModel(Model):

    def __init__(self, fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 num_adaptive_constants: int=0,
                 dot_product: bool=False):
        super().__init__()
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
        self.predicate_embedder = PredicateEmbeddings(
            fol.predicates,
            predicate_embedding_size,
            regularization=kge_regularization,
            has_features=False)
        self.constant_embedder = ConstantEmbeddings(
            domains=fol.domains,
            constant_embedding_sizes_per_domain={
                domain.name: constant_embedding_size
                for domain in fol.domains},
            regularization=kge_regularization,
            has_features=False)
        self.dot_product = dot_product
        if num_adaptive_constants > 0:
            self.adaptive_constant_embedder = AdaptiveConstantEmbeddings(
                domains=fol.domains,
                constant_embedder=self.constant_embedder,
                constant_embedding_size=constant_embedding_size,
                num_adaptive_constants=num_adaptive_constants,
                dot_product=dot_product)
        else:
            self.adaptive_constant_embedder = None

        self.kge_embedder, self.output_layer = KGEFactory(
            name=kge,
            atom_embedding_size=kge_atom_embedding_size,
            relation_embedding_size=kge_atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate)
        assert self.kge_embedder is not None


    def create_triplets(self,
                        constant_embeddings: Dict[str, tf.Tensor],
                        predicate_embeddings: tf.Tensor,
                        A_predicates: Dict[str, tf.Tensor]):
        predicate_embeddings_per_triplets = []
        for p,indices in A_predicates.items():
            idx = self.fol.name2predicate_idx[p]
            p_embeddings = tf.expand_dims(predicate_embeddings[idx], axis=0)  # 1E
            predicate_embeddings_per_triplets.append(
                tf.repeat(p_embeddings, tf.shape(indices)[0], axis=0)  #PE
            )
        predicate_embeddings_per_triplets = tf.concat(predicate_embeddings_per_triplets,
                                                      axis=0)

        constant_embeddings_for_triplets = []
        for p,constant_idx in A_predicates.items():
            constant_idx = tf.cast(constant_idx, tf.int32)
            predicate = self.fol.name2predicate[p]
            one_predicate_constant_embeddings = []
            for i,domain in enumerate(predicate.domains):
                constants = tf.gather(constant_embeddings[domain.name],
                                      constant_idx[..., i], axis=0)
                one_predicate_constant_embeddings.append(constants)
            # shape (predicate_batch_size, predicate_arity, constant_embedding_size)
            one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings,
                                                         axis=-2)
            constant_embeddings_for_triplets.append(one_predicate_constant_embeddings)
        constant_embeddings_for_triplets = tf.concat(constant_embeddings_for_triplets,
                                                     axis=0)
        tf.debugging.assert_equal(tf.shape(predicate_embeddings_per_triplets)[0],
                                  tf.shape(constant_embeddings_for_triplets)[0])
        # Shape TE, T2E with T number of triplets.
        return predicate_embeddings_per_triplets, constant_embeddings_for_triplets

    def call(self, inputs):
        # X_domains type is Dict[str, inputs]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        (X_domains, A_predicates) = inputs
        if self.adaptive_constant_embedder is not None:
            X_domains_fixed_mask = {
                name:tf.where(x < len(self.fol.name2domain[name].constants),
                              True, False) for name,x in X_domains.items()}
            X_domains_fixed = {
                name:tf.where(X_domains_fixed_mask[name], x, 0)
                for name,x in X_domains.items()}
            constant_embeddings_fixed = self.constant_embedder(X_domains_fixed)
            constant_embeddings_adaptive = self.adaptive_constant_embedder(
                X_domains)
            constant_embeddings = {
                name:tf.where(
                    # Expand dim to broadcast to the embeddings size.
                    tf.expand_dims(X_domains_fixed_mask[name], axis=-1),
                    constant_embeddings_fixed[name],
                    constant_embeddings_adaptive[name])
                for name in X_domains.keys()}
            tf.print('EMB', {
                name:tf.boolean_mask(
                    constant_embeddings_adaptive[name],
                    tf.math.logical_not(
                        X_domains_fixed_mask[name]))
                for name in X_domains.keys()})
        else:
            constant_embeddings = self.constant_embedder(X_domains)

        predicate_embeddings = self.predicate_embedder(self.predicate_index_tensor)
        # Shape TE, T2E with T number of triplets.
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates)

        # Shape TE
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        # Shape T1
        atom_outputs = tf.expand_dims(self.output_layer(atom_embeddings), -1)
        return atom_outputs, atom_embeddings


class CollectiveModel(Model):

    def __init__(self,
                 fol: FOL,
                 rules: List[Rule],
                 *,  # all named after this point
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 # same model for all reasoning depths
                 reasoner_atom_embedding_size: int,
                 reasoner_formula_hidden_embedding_size: int,
                 reasoner_regularization: float,
                 reasoner_single_model: bool,
                 reasoner_dropout_rate: float,
                 reasoner_depth: int,
                 aggregation_type: str,
                 signed: bool,
                 temperature: float,
                 model_name: str,
                 resnet: bool,
                 embedding_resnet: bool,
                 filter_num_heads: int,
                 filter_activity_regularization: float,
                 num_adaptive_constants: int,
                 dot_product: bool,
                 cdcr_use_positional_embeddings: bool,
                 cdcr_num_formulas: int,
                 r2n_prediction_type: str):
        super().__init__()
        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from
        # self.reasoner_depth during multi-stage learning (like if
        # pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.embedding_resnet = embedding_resnet
        self.logic = GodelTNorm()

        self.kge_model = KGEModel(fol, kge,
                                  kge_regularization,
                                  constant_embedding_size,
                                  predicate_embedding_size,
                                  kge_atom_embedding_size,
                                  kge_dropout_rate,
                                  num_adaptive_constants,
                                  dot_product)
        self.model_name = model_name

        # REASONING LAYER
        self.reasoning = None
        if reasoner_depth > 0 and len(rules) > 0:
            if self.embedding_resnet:
                self.embedding_resnet_weight = Sequential([
                    #Dense(16, activation='relu'),
                    #Dropout(0.2),
                    Dense(1,
                          #kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                          activation='sigmoid')])

            self.reasoning = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
                  continue

              if model_name == 'dcr':
                  self.reasoning.append(DCRReasoningLayer(
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'cdcr':
                  self.reasoning.append(ClusteredDCRReasoningLayer(
                      num_formulas=cdcr_num_formulas,
                      use_positional_embeddings=cdcr_use_positional_embeddings,
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'r2n':
                  def SumAndSigmoidOutput(x):
                      return tf.nn.sigmoid(tf.reduce_sum(x, axis=-1))
                  output_layer = (Lambda(SumAndSigmoidOutput, name='output_layer')
                                  if kge == 'rotate' else self.kge_model.output_layer)
                  self.reasoning.append(R2NReasoningLayer(
                      rules=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      atom_embedding_size=reasoner_atom_embedding_size,
                      prediction_type=r2n_prediction_type,
                      aggregation_type=aggregation_type,
                      output_layer=output_layer,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'sbr':
                  self.reasoning.append(SBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type))

              elif model_name == 'gsbr':
                  self.reasoning.append(GatedSBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'rnm':
                  self.reasoning.append(RNMReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'dsl':
                  self.reasoning.append(DeepStocklogLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              else:
                  assert False, 'Unknown model name %s' % model_name

        self._explain_mode = False

    def explain_mode(self, mode=True):
        self._explain_mode = mode

    def call(self, inputs, *args, **kwargs):
        if self._explain_mode:
            # No explanations are posible when reasoning is disabled.
            assert self.reasoning is not None
            # Check that we are using an explainable model.
            assert self.model_name == 'dcr' or self.model_name == 'cdcr'

        # X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        #              e.g. mapping predicate_name -> tensor [num_groundings, arity]
        #                   with constant indices for each grounding.
        (X_domains, A_predicates, A_rules, Q) = inputs
        concept_output, concept_embeddings = self.kge_model((X_domains, A_predicates))
        task_output = tf.identity(concept_output)

        explanations = None
        if self.reasoning is not None:
            atom_embeddings = tf.identity(concept_embeddings)
            for i in range(self.enabled_reasoner_depth):
                if self._explain_mode and i == self.enabled_reasoner_depth - 1:
                    explanations = self.reasoning[i].explain(
                        [task_output, atom_embeddings, A_rules])
                task_output, atom_embeddings = self.reasoning[i]([
                    task_output, atom_embeddings, A_rules])
            if self.embedding_resnet:
                # In this case we need to recompute the output from the updated embeddings.
                w = tf.clip_by_value(self.embedding_resnet_weight(tf.concat([concept_embeddings, atom_embeddings], axis=-1)), 1e-9, 1.0 - 1e-7)
                tf.print('embedding_resnet_weight', tf.reduce_mean(w))
                atom_embeddings = (1.0 - w) * tf.stop_gradient(concept_embeddings) + w * atom_embeddings
                task_output = tf.expand_dims(self.kge_model.output_layer(atom_embeddings), axis=-1)

        task_output = tf.gather(params=tf.squeeze(task_output, -1), indices=Q)
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),
                                   indices=Q)
        if self.resnet and self.reasoning is not None:
            task_output = self.logic.disj_pair(task_output, concept_output)

        if self._explain_mode:
            return concept_output, task_output, explanations
        else:
            return concept_output, task_output
