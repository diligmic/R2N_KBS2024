import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Layer, Dense, Dropout, LeakyReLU
import ns_lib as ns
from typing import List,Tuple
from ns_lib.logic.commons import Rule
from ns_lib.logic.semantics import *
from ns_lib.nn.concepts import *


##################################################
# Base class with common functionalities for all reasoning layers.
class ReasoningLayer(Layer):
    def __init__(self):

        super().__init__()

    def _merge_clique_data_by_atom(self,
                                   clique_data, num_atoms,
                                   grounding_indices, max_num_atoms,
                                   aggregation_type):
        grounding_indices = tf.expand_dims(grounding_indices, -1)
        if aggregation_type == 'max':
            base = tf.float32.min * tf.ones(
                shape=[max_num_atoms, tf.shape(clique_data)[-1]])
            return tf.tensor_scatter_nd_max(
                base, grounding_indices, clique_data)

        elif aggregation_type == 'softmax':
            base_max = tf.float32.min * tf.ones(
                shape=[max_num_atoms, tf.shape(clique_data)[-1]])
            aggregated_by_max = tf.tensor_scatter_nd_max(
                base_max, grounding_indices, clique_data)
            base_sum = tf.zeros(
                shape=[max_num_atoms, tf.shape(clique_data)[-1]])
            aggregated_by_sum = tf.tensor_scatter_nd_max(
                base_sum, grounding_indices, clique_data)
            return tf.divide_no_nan(aggregated_by_max, aggregated_by_sum)

        elif aggregation_type == 'sum':
            base = tf.zeros([max_num_atoms, tf.shape(clique_data)[-1]])
            return tf.tensor_scatter_nd_add(
                base, grounding_indices, clique_data)

        elif aggregation_type == 'mean':
            base = tf.zeros([max_num_atoms, tf.shape(clique_data)[-1]])
            aggregated_sum = tf.tensor_scatter_nd_add(
                base, grounding_indices, clique_data)
            num_groundings = tf.shape(grounding_indices)[0]
            ones_for_avg = tf.ones(shape=[num_groundings, num_atoms, 1])
            base_count = tf.zeros([max_num_atoms, 1])
            count = tf.tensor_scatter_nd_add(base_count, grounding_indices,
                                             ones_for_avg)
            return tf.math.divide_no_nan(aggregated_sum, count)

        else:
            raise Exception('Unknown aggregation method: %s' % aggregation_type)

    def _formula_aggregation(self, data_all_formulas, aggregation_type: str):
        if aggregation_type == 'mean':
            return tf.reduce_mean(data_all_formulas, axis=0)
        elif self.aggregation_type == 'max':
            return tf.reduce_max(data_all_formulas, axis=0)
        elif self.aggregation_type == 'sum':
            return tf.reduce_sum(data_all_formulas, axis=0)
        else:
            raise Exception('Unknown aggregation method: %s' % aggregation_type)
            assert False, 'Unknown aggregation method'  # blocking error

    def _gumbel(self, logits, temperature: float,
                make_prob: bool=False, hard: bool=False):
        assert temperature > 0.
        coolness = 1.0 / temperature
        dist = tfp.distributions.Logistic(logits * coolness, coolness)
        logits_sample = dist.sample()
        sample = tf.sigmoid(logits_sample) if make_prob else logits_sample
        if hard:
            argmax_index = tf.math.argmax(sample, axis=-1)
            argmax_mask = tf.one_hot(argmax_index, tf.shape(sample)[-1],
                                     axis=-1)
            sample = tf.where(argmax_mask == 1.0, 1.0, 0.0)
        return sample

#################################################
class R2NReasoningLayer(ReasoningLayer):

    def __init__(self, rules: List[Rule],
                 formula_hidden_size: int,
                 atom_embedding_size: int,
                 output_layer: tf.keras.Model,
                 aggregation_type: str='max',
                 prediction_type: str='full',
                 regularization: float=0.0,
                 dropout_rate: float=0.0):

        super().__init__()

        self.rules = rules
        self.prediction_type = prediction_type  # can be wither "full" or "head"
        assert self.prediction_type == 'full' or self.prediction_type == 'head'

        self.regularizer = (tf.keras.regularizers.l2(regularization)
                            if regularization > 0.0 else None)

        self.dropout_layer = (Dropout(dropout_rate)
                              if dropout_rate > 0.0 else None)
        self.output_layer = output_layer

        self.rule_embedders = {}
        for rule in rules:
            self.rule_embedders[rule.name] =  tf.keras.Sequential()

            if formula_hidden_size > 0:
                self.rule_embedders[rule.name].add(Dense(
                    formula_hidden_size, activation='relu',
                    kernel_regularizer=self.regularizer))
                if dropout_rate > 0.0:
                    self.rule_embedders[rule.name].add(Dropout(dropout_rate))

            num_predicted_atoms = (len(rule.head) + len(rule.body)
                                   if self.prediction_type == "full"
                                   else len(rule.head))
            output_embedding_size = atom_embedding_size * num_predicted_atoms
            self.rule_embedders[rule.name].add(Dense(
                output_embedding_size,
                activation=None if formula_hidden_size > 0 else tf.nn.relu,
                use_bias=True,
                kernel_regularizer=self.regularizer))

            if dropout_rate > 0.0:
                self.rule_embedders[rule.name].add(Dropout(dropout_rate))

        self.aggregation_type = aggregation_type

    def call(self, inputs):
        input_concepts, input_atom_embeddings, formula_to_atom_tuples = inputs
        num_atoms = tf.shape(input_atom_embeddings)[0]
        input_atom_embedding_size = tf.shape(input_atom_embeddings)[-1]

        atom_embeddings_all_formulas = []

        for rule in self.rules:
            assert rule.name in formula_to_atom_tuples, (
                '%s missing in rules %s' % (
                    rule.name, list(formula_to_atom_tuples.keys())))
            (A_in, A_out) = formula_to_atom_tuples[rule.name]
            if self.prediction_type == 'full':
                A = tf.concat((A_in, A_out), axis=1)
                A_in = A
                A_out = A
                num_atoms_out = len(rule.head) + len(rule.body)
                num_atoms_in = len(rule.head) + len(rule.body)
            else:
                num_atoms_out = len(rule.head)
                num_atoms_in = len(rule.body)

            block_input = tf.gather(params=input_atom_embeddings, indices=A_in)
            block_input = tf.reshape(  # Concat the atoms of the body
                block_input,
                [tf.shape(block_input)[0],
                 num_atoms_in * input_atom_embedding_size])

            clique_embeddings = self.rule_embedders[rule.name](block_input)
            clique_embeddings = tf.stack(tf.split(clique_embeddings,
                                                  num_or_size_splits=num_atoms_out,
                                                  axis=-1), axis=1)

            atom_embeddings_per_one_formula = (
                self._merge_clique_data_by_atom(
                    clique_embeddings,
                    num_atoms_out,
                    A_out,
                    num_atoms, self.aggregation_type))

            atom_embeddings_all_formulas.append(atom_embeddings_per_one_formula)

        atom_embeddings_all_formulas = tf.stack(atom_embeddings_all_formulas, 0)

        atom_embeddings = self._formula_aggregation(
            atom_embeddings_all_formulas, self.aggregation_type)

        predictions = tf.expand_dims(self.output_layer(atom_embeddings), axis=-1)
        return predictions, atom_embeddings


###############################################
class DCRReasoningLayer(ReasoningLayer):

    def __init__(self,
                 templates: List[Rule],
                 formula_hidden_size: int,
                 aggregation_type: str='max',
                 temperature: float=1.0,
                 signed: bool=True,
                 filter_num_heads: int=1,
                 filter_activity_regularization: float=0.0,
                 regularization: float=0.0,
                 dropout_rate: float=0.0,
                 logic: Logic=GodelTNorm()):

        super().__init__()

        self.templates = templates
        self.signed = signed
        self.temperature = temperature
        self._explain_mode = False
        self.regularizer = (tf.keras.regularizers.l2(regularization)
                            if regularization > 0.0 else None)

        self.dropout_layer = (Dropout(dropout_rate)
                              if dropout_rate > 0.0 else None)

        self.rule_embedders = {}
        self.logic = logic

        for rule in templates:
            #self.rule_embedders[rule.name] = ConceptReasoningLayerUnfiltered(
            #    rule, formula_hidden_size, logic=self.logic)
            self.rule_embedders[rule.name] = ConceptReasoningLayer(
                rule, formula_hidden_size,
                signed=self.signed,
                logic=self.logic,
                filter_num_heads=filter_num_heads,
                filter_activity_regularization=filter_activity_regularization)

        self.aggregation_type = aggregation_type

    # Takes concepts, atom embeddings, grounding structures and returns the
    # predictions.
    def call(self, inputs):
        input_concepts, input_atom_embeddings, formula_to_atom_tuples = inputs
        num_atoms = tf.shape(input_atom_embeddings)[0]
        input_atom_embedding_size = tf.shape(input_atom_embeddings)[-1]

        predictions_all_formulas = []
        if self._explain_mode:
            # Dict[rule_name:str, Tuple[x, c, preds, sign_attn, filter_attn]]
            rule2explain_info = {}

        for rule in self.templates:
            assert rule.name in formula_to_atom_tuples, (
                '%s missing in rules %s' % (
                    rule.name, list(formula_to_atom_tuples.keys())))
            (A_in, A_out) = formula_to_atom_tuples[rule.name]
            num_atoms_out = len(rule.head)

            atom_embeddings = tf.gather(params=input_atom_embeddings,
                                        indices=A_in)
            concepts = tf.gather(params=input_concepts, indices=A_in)

            if self._explain_mode:
                rule2explain_info[rule.name] = self.rule_embedders[rule.name](
                    [concepts, atom_embeddings], return_explain_info=True)
                rule2explain_info[rule.name]['relational_info'] = [A_in, A_out]

            head_predictions = self.rule_embedders[rule.name](
                [concepts, atom_embeddings], constrastive_task_loss_weight=0.1)
            head_predictions = tf.stack(
                tf.split(head_predictions, num_or_size_splits=num_atoms_out,
                         axis=-1), axis=1)

            predictions_per_one_formula = (
                self._merge_clique_data_by_atom(
                    head_predictions,
                    num_atoms_out,
                    A_out,
                    num_atoms, self.aggregation_type))

            predictions_all_formulas.append(predictions_per_one_formula)

        predictions_all_formulas = tf.stack(predictions_all_formulas, 0)
        predictions = self._formula_aggregation(
            predictions_all_formulas, self.aggregation_type)

        if self._explain_mode:
            return predictions, input_atom_embeddings, rule2explain_info
        else:
            return predictions, input_atom_embeddings

    def explain(self, inputs):
        self._explain_mode = True
        _, _, rule2explain_info = self(inputs)

        rule2explain_info_aggregated = {}
        for r,explain_info in rule2explain_info.items():
            indices_bodies, indices_head = explain_info['relational_info']
            del explain_info['relational_info']

            y_preds = explain_info['preds']
            idx_max_grounding = {}
            for idx_grounding, h in enumerate(indices_head):
                id_head = h[0].numpy()
                if id_head not in idx_max_grounding:
                    idx_max_grounding[id_head] = idx_grounding
                elif y_preds[idx_max_grounding[id_head]][0] < y_preds[idx_grounding][0]:
                    idx_max_grounding[id_head] = idx_grounding

            idx_max_grounding = tf.expand_dims(
                tf.constant(list(idx_max_grounding.values()), dtype=tf.int32),
                axis=-1)
            explain_info = {k:tf.gather_nd(v, idx_max_grounding) for k,v in explain_info.items()}
            rule2explain_info_aggregated[r] = explain_info

        explanations = {r:self.rule_embedders[r].explain(explain_info)
                        for r, explain_info in rule2explain_info_aggregated.items()}
        self._explain_mode = False
        return explanations


###############################################
class ClusteredDCRReasoningLayer(DCRReasoningLayer):

    def __init__(self,
                 templates: List[Rule],
                 num_formulas: int,
                 formula_hidden_size: int,
                 use_positional_embeddings: bool=True,
                 aggregation_type: str='max',
                 temperature: float=1.0,
                 signed: bool=True,
                 filter_num_heads: int=1,
                 filter_activity_regularization: float=0.0,
                 regularization: float=0.0,
                 dropout_rate: float=0.0,
                 constrastive_task_loss_weight: float=0.1,
                 logic: Logic=GodelTNorm()):

        super().__init__(templates=templates,
                         formula_hidden_size=formula_hidden_size,
                         aggregation_type=aggregation_type,
                         temperature=temperature,
                         signed=signed,
                         filter_num_heads=filter_num_heads,
                         filter_activity_regularization=filter_activity_regularization,
                         regularization=regularization,
                         dropout_rate=dropout_rate,
                         logic=logic)
        self.num_formulas = num_formulas
        self.formula_hidden_size = formula_hidden_size
        self.use_positional_embeddings = use_positional_embeddings
        self.constrastive_task_loss_weight = constrastive_task_loss_weight

        self.positional_embeddings = {}
        self.formula_scorer = {}
        self.formula_embedders = {}
        self.rule_embedders = {}
        filter_activity_regularizer = (
            tf.keras.regularizers.l2(filter_activity_regularization)
            if filter_activity_regularization > 0.0 else None)
        for rule in templates:
            num_body_atoms = len(rule.body)
            self.positional_embeddings[rule.name] = tf.random.uniform(
                [1, num_body_atoms, formula_hidden_size])
            self.formula_scorer[rule.name] = Dense(1, activation=None)
            formula_embedder_dim = num_formulas * formula_hidden_size
            if not use_positional_embeddings:
                formula_embedder_dim *= num_body_atoms
            self.formula_embedders[rule.name] = tf.keras.Sequential([
                    Dense(formula_embedder_dim, activation=None),
                    LeakyReLU(),
                    Dense(formula_embedder_dim, activation=None,
                          activity_regularizer=filter_activity_regularizer)])
            #self.rule_embedders[rule.name] = ConceptReasoningLayerUnfiltered(
            #    rule, formula_hidden_size, logic=self.logic)
            self.rule_embedders[rule.name] = ConceptReasoningLayer(
                rule, formula_hidden_size,
                logic=self.logic,
                signed=self.signed,
                filter_num_heads=filter_num_heads,
                filter_activity_regularization=filter_activity_regularization)

    # Takes concepts, atom embeddings, grounding structures and returns the
    # predictions.
    def call(self, inputs):
        input_concepts, input_atom_embeddings, formula_to_atom_tuples = inputs
        num_atoms = tf.shape(input_atom_embeddings)[0]
        input_atom_embedding_size = tf.shape(input_atom_embeddings)[-1]

        predictions_all_formulas = []
        if self._explain_mode:
            # Dict[rule_name:str, Tuple[x, c, preds, sign_attn, filter_attn]]
            rule2explain_info = {}
        for rule in self.templates:
            assert rule.name in formula_to_atom_tuples, (
                '%s missing in rules %s' % (
                    rule.name, list(formula_to_atom_tuples.keys())))
            (A_in, A_out) = formula_to_atom_tuples[rule.name]
            num_atoms_out = len(rule.head)
            num_body_atoms = len(rule.body)

            atom_embeddings = tf.gather(params=input_atom_embeddings,
                                        indices=A_in)
            concepts = tf.gather(params=input_concepts, indices=A_in)

            # Shape (batch_size, num_body_atoms * atom_embedding_size)
            atom_embeddings = tf.reshape(atom_embeddings,
                                         (tf.shape(atom_embeddings)[0], -1))

            # if self.use_positional_embeddings:
            # Shape (batch_size, num_formulas * formula_hidden_size)
            # else:
            # Shape (batch_size,
            #        num_formulas * num_body_atoms * formula_hidden_size)
            formula_embeddings = self.formula_embedders[rule.name](
                atom_embeddings)

            # if self.use_positional_embeddings:
            # Shape (batch_size, num_formulas, formula_hidden_size)
            # else:
            # Shape (batch_size, num_formulas,
            #        num_body_atoms * formula_hidden_size)
            formula_embeddings = tf.reshape(
                formula_embeddings,
                shape=(tf.shape(formula_embeddings)[0], self.num_formulas, -1))

            # Shape (batch_size, num_formulas)
            formula_logits = tf.squeeze(self.formula_scorer[rule.name](
                formula_embeddings), axis=-1)
            if self.temperature > 0.0:
                formula_logits = self._gumbel(formula_logits, self.temperature)

            # Shape (batch_size, num_formulas)
            formula_prob = tf.nn.softmax(formula_logits)
            # Shape (batch_size,)
            formula_selected = tf.math.argmax(formula_logits, axis=-1)

            # if self.use_positional_embeddings:
            # Shape (batch_size, formula_hidden_size)
            # else:
            # Shape (batch_size, num_body_atoms * formula_hidden_size)
            formula_embeddings = tf.gather_nd(
                formula_embeddings, tf.stack(
                    [tf.range(tf.shape(formula_embeddings)[0],
                              dtype=formula_selected.dtype),
                     formula_selected], axis=1))

            if self.use_positional_embeddings:
                # Shape (batch_size, num_body_atoms, formula_hidden_size)
                formula_embeddings = tf.tile(
                    tf.expand_dims(formula_embeddings, axis=1),
                    multiples=(1, num_body_atoms, 1))

                # Shape (batch_size, num_body_atoms, formula_hidden_size)
                positional_embeddings = tf.tile(
                    self.positional_embeddings[rule.name],
                    multiples=(tf.shape(formula_embeddings)[0], 1, 1))
                # Shape (batch_size, num_body_atoms, formula_hidden_size)
                formula_embeddings += positional_embeddings
            else:
                # Shape (batch_size, num_body_atoms, formula_hidden_size)
                formula_embeddings = tf.reshape(
                    formula_embeddings,
                    shape=(-1, num_body_atoms, self.formula_hidden_size))

            rule_embedders = self.rule_embedders[rule.name]
            if self._explain_mode:
                rule2explain_info[rule.name] = rule_embedders(
                    [concepts, formula_embeddings], return_explain_info=True)
                rule2explain_info[rule.name]['relational_info'] = [A_in, A_out]

            # Shape (batch_size, num_atoms_out)
            head_predictions = rule_embedders(
                [concepts, formula_embeddings],
                constrastive_task_loss_weight=self.constrastive_task_loss_weight)
            # Shape (batch_size, num_atoms_out, 1)
            head_predictions = tf.expand_dims(head_predictions, axis=-1)

            predictions_per_one_formula = (
                self._merge_clique_data_by_atom(
                    head_predictions,
                    num_atoms_out,
                    A_out,
                    num_atoms, self.aggregation_type))

            predictions_all_formulas.append(predictions_per_one_formula)

        predictions_all_formulas = tf.stack(predictions_all_formulas, 0)

        predictions = self._formula_aggregation(
            predictions_all_formulas, self.aggregation_type)

        if self._explain_mode:
            return predictions, input_atom_embeddings, rule2explain_info
        else:
            return predictions, input_atom_embeddings


##################################################
class SBRReasoningLayer(ReasoningLayer):

    def __init__(self,
                 rules: List[Rule],
                 aggregation_type: str='max',
                 logic: Logic=GodelTNorm()):

        super().__init__()

        self.rules = rules
        self.logic = logic
        self.aggregation_type = aggregation_type

    # TODO: add explain mode.
    def call(self, inputs):
        input_atom_predictions, input_atom_embeddings, formula_to_atom_tuples = inputs
        num_atoms = tf.shape(input_atom_embeddings)[0]
        input_atom_embedding_size = tf.shape(input_atom_embeddings)[-1]

        predictions_all_formulas = []

        for rule in self.rules:
            assert rule.name in formula_to_atom_tuples, (
                '%s missing in rules %s' % (
                    rule.name, list(formula_to_atom_tuples.keys())))
            (A_in, A_out) = formula_to_atom_tuples[rule.name]

            # Shape [num_atoms, body_len, emb_size]
            atom_embeddings = tf.gather(params=input_atom_embeddings, indices=A_in)
            # Shape [num_atoms, body_len]
            atom_predictions = tf.squeeze(tf.gather(
                params=input_atom_predictions, indices=A_in), axis=-1)
            # Shape [num_atoms, 1]
            head_predictions = self.logic.conj(atom_predictions, axis=-1)

            num_atoms_out = len(rule.head)
            # Shape [num_atoms, head_len, 1]
            head_predictions = tf.stack(tf.split(head_predictions,
                                                 num_or_size_splits=num_atoms_out,
                                                 axis=-1), axis=1)
            predictions_per_one_formula = self._merge_clique_data_by_atom(
                head_predictions, num_atoms_out, A_out, num_atoms,
                self.aggregation_type)

            predictions_all_formulas.append(predictions_per_one_formula)

        predictions_all_formulas = tf.stack(predictions_all_formulas, 0)
        predictions = self._formula_aggregation(
            predictions_all_formulas, self.aggregation_type)

        return predictions, input_atom_embeddings


##################################################
# Base class for GatedSBR models.
class _GatedSBRReasoningLayerBase(SBRReasoningLayer):

    def __init__(self, rules: List[Rule],
                 aggregation_type: str='max',
                 regularization: float=0.0,
                 #dropout_rate: float=0.0,
                 logic: Logic=GodelTNorm(),
                 # use a per_grounding gate vs a gate per rule.
                 per_grounding_gate: bool=True):

        super().__init__(rules, aggregation_type, logic)

        self.regularizer = (tf.keras.regularizers.l2(regularization)
                            if regularization > 0.0 else None)

        #self.dropout_layer = (Dropout(dropout_rate)
        #                      if dropout_rate > 0.0 else None)
        self.per_grounding_gate = per_grounding_gate
        if per_grounding_gate:
            self.rule_gates = {rule.name: tf.keras.Sequential(
                [tf.keras.layers.Dense(1,
                                       kernel_regularizer=self.regularizer,
                                       activation='sigmoid')])
                               for rule in rules}
        else:
            self.rule_gates = {rule.name:
                               tf.Variable(tf.zeros([1]),
                                           name='rule_weight(%s)' % rule.name,
                                           dtype=tf.float32)
                               for rule in rules}


    # TODO: add explain mode.
    def call(self, inputs):
        input_atom_predictions, input_atom_embeddings, formula_to_atom_tuples = inputs
        num_atoms = tf.shape(input_atom_embeddings)[0]
        input_atom_embedding_size = tf.shape(input_atom_embeddings)[-1]

        predictions_all_formulas = []

        for rule in self.rules:
            assert rule.name in formula_to_atom_tuples, (
                '%s missing in rules %s' % (
                    rule.name, list(formula_to_atom_tuples.keys())))
            (A_in, A_out) = formula_to_atom_tuples[rule.name]

            # Shape [num_atoms, body_len, emb_size]
            atom_embeddings = tf.gather(params=input_atom_embeddings, indices=A_in)
            # Shape [num_atoms, body_len]
            atom_predictions = tf.squeeze(tf.gather(
                params=input_atom_predictions, indices=A_in), axis=-1)
            # Shape [num_atoms, 1]
            head_predictions = self.logic.conj(atom_predictions, axis=-1)

            if self.per_grounding_gate:
                # Shape [num_atoms, body_len]
                atom_embeddings_shape = tf.shape(atom_embeddings)
                gate_inputs = tf.reshape(
                    atom_embeddings, (atom_embeddings_shape[0], -1))
                rule_weights = self.rule_gates[rule.name](gate_inputs)
            else:
                rule_weights = tf.nn.sigmoid(self.rule_gates[rule.name])
            head_predictions = rule_weights * head_predictions
            num_atoms_out = len(rule.head)
            head_predictions = tf.stack(tf.split(head_predictions,
                                                 num_or_size_splits=num_atoms_out,
                                                 axis=-1), axis=1)
            predictions_per_one_formula = self._merge_clique_data_by_atom(
                head_predictions, num_atoms_out, A_out, num_atoms,
                self.aggregation_type)

            predictions_all_formulas.append(predictions_per_one_formula)

        predictions_all_formulas = tf.stack(predictions_all_formulas, axis=0)
        predictions = self._formula_aggregation(
            predictions_all_formulas, self.aggregation_type)

        return predictions, input_atom_embeddings

##################################################
class GatedSBRReasoningLayer(_GatedSBRReasoningLayerBase):

    def __init__(self, rules: List[Rule],
                 aggregation_type: str='max',
                 regularization: float=0.0,
                 logic: Logic=GodelTNorm()):

        super().__init__(rules, aggregation_type, regularization, logic,
                         per_grounding_gate=True)

##################################################
class RNMReasoningLayer(_GatedSBRReasoningLayerBase):

    def __init__(self, rules: List[Rule],
                 aggregation_type: str='max',
                 regularization: float=0.0,
                 #dropout_rate: float=0.0,
                 logic: Logic=GodelTNorm()):

        super().__init__(rules, aggregation_type, regularization, logic,
                         per_grounding_gate=False)


##################################################
class GradientDescentUpdate(tf.Module):
    # It is a not-in-place version (var is a tf.Tensor and not a tf.Variable)
    # of the optimizer in https://www.tensorflow.org/guide/core/optimizers_core
    def __init__(self, learning_rate=1e-3):
            # Initialize parameters
        super().__init__()
        self.learning_rate = learning_rate

    def apply_gradients(self, grad, var):
            return var - self.learning_rate * grad

class DeepLogicModelReasoner(Layer):

    def __init__(self, semantic_loss, output_layer, update_model):

        super().__init__()
        self.semantic_loss = semantic_loss
        self.update_model = update_model
        self.output_layer = output_layer


    def call(self, inputs):
        atom_embeddings, A_rules = inputs
        with tf.GradientTape() as tape:
            tape.watch(atom_embeddings)
            predictions = self.output_layer(atom_embeddings)
            loss= self.semantic_loss([predictions, A_rules])
        grads = tape.gradient(loss, atom_embeddings)
        atom_embeddings = self.update_model.apply_gradients(
            grads, atom_embeddings)
        return atom_embeddings

##################################################
class DeepStocklogLayer(SBRReasoningLayer):

    def __init__(self, rules: List[Rule],
                 aggregation_type: str='max',
                 regularization: float=0.0,
                 #dropout_rate: float=0.0,
                 logic: Logic=GodelTNorm()):

        super().__init__(rules, aggregation_type, logic)

        self.regularizer = (tf.keras.regularizers.l2(regularization)
                            if regularization > 0.0 else None)

        self.rule_weights = tf.Variable(
            #tf.random_normal([len(rules)], stddev=0.35),
            tf.ones([len(rules)]),
            # [1.0 for i in range(len(rules))],
            name='rule_weights',
            dtype=tf.float32)
        self.rule2weight_index = {rule.name:i for i,rule in enumerate(rules)}


    # TODO: add explain mode.
    def call(self, inputs):
        input_atom_predictions, input_atom_embeddings, formula_to_atom_tuples = inputs
        num_atoms = tf.shape(input_atom_embeddings)[0]
        input_atom_embedding_size = tf.shape(input_atom_embeddings)[-1]

        predictions_all_formulas = []

        rule_weights = tf.nn.softmax(self.rule_weights)
        tf.print(rule_weights)
        for rule in self.rules:
            weight = rule_weights[self.rule2weight_index[rule.name]]
            assert rule.name in formula_to_atom_tuples, (
                '%s missing in rules %s' % (
                    rule.name, list(formula_to_atom_tuples.keys())))
            (A_in, A_out) = formula_to_atom_tuples[rule.name]

            # Shape [num_atoms, body_len, emb_size]
            atom_embeddings = tf.gather(params=input_atom_embeddings, indices=A_in)
            # Shape [num_atoms, body_len]
            atom_predictions = tf.squeeze(tf.gather(
                params=input_atom_predictions, indices=A_in), axis=-1)
            # Shape [num_atoms, 1]
            head_predictions = self.logic.conj(atom_predictions, axis=-1)

            head_predictions = weight * head_predictions
            num_atoms_out = len(rule.head)
            head_predictions = tf.stack(tf.split(head_predictions,
                                                 num_or_size_splits=num_atoms_out,
                                                 axis=-1), axis=1)
            predictions_per_one_formula = self._merge_clique_data_by_atom(
                head_predictions, num_atoms_out, A_out, num_atoms,
                self.aggregation_type)

            predictions_all_formulas.append(predictions_per_one_formula)

        predictions_all_formulas = tf.stack(predictions_all_formulas, axis=0)
        predictions = self._formula_aggregation(
            predictions_all_formulas, self.aggregation_type)

        return predictions, input_atom_embeddings
