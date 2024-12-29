import keras
import tensorflow as tf
import numpy as np
from collections import Counter
from keras.layers import Layer, Dense, Dropout, LeakyReLU
from ns_lib.logic.commons import Rule
from ns_lib.logic.semantics import GodelTNorm, Logic
from typing import Dict

class ConceptReasoningLayer(Layer):
    def __init__(self, rule: Rule, emb_size: int,
                 logic: Logic = GodelTNorm(),
                 filter_num_heads: int=1,
                 filter_activity_regularization: float=0.0,
                 signed: bool=True):
        super().__init__()
        self.rule = rule
        self.emb_size = emb_size
        self.n_classes = len(rule.head)
        self.logic = logic
        self.filter_num_heads = filter_num_heads
        filter_activity_regularizer = (
            tf.keras.regularizers.l2(filter_activity_regularization)
            if filter_activity_regularization > 0.0 else None)

        if self.filter_num_heads > 1:
            self.filter_nn = []
            for h in range(self.filter_num_heads):
                self.filter_nn.append(keras.Sequential([
                    Dense(emb_size, activation=None, name=('filter_%d_1' % h)),
                    LeakyReLU(),
                    Dense(self.n_classes, activation=tf.nn.softmax,
                          name=('filter_%d_2' % h),
                          activity_regularizer=filter_activity_regularizer)]))
        else:
            self.filter_nn = keras.Sequential([
                    Dense(emb_size, activation=None, name='filter_1'),
                    LeakyReLU(),
                    Dense(self.n_classes, activation=tf.nn.sigmoid, name='filter_2',
                          activity_regularizer=filter_activity_regularizer)])

        self.signed = signed
        if self.signed:  # Keep only positive concepts.
            self.sign_nn = keras.Sequential([
                Dense(emb_size, activation=None, name='sign_1'),
                LeakyReLU(),
                Dense(self.n_classes, activation=None, name='sign_2')])
        else:
            self.sign_nn = None

    def call(self, inputs,
             return_explain_info: bool=False,
             constrastive_task_loss_weight: float=0.0):
        # Change x to be a sigle embedding and make the sign and filter
        # function to expand to the nun_body_atoms.
        # x.shape = (batch_size, num_body_atoms, embedding_size)
        # c.shape = (batch_size, num_body_atoms, 1) TO CHECK
        c, x = inputs
        values = tf.tile(c, multiples=(1, 1, self.n_classes))

        # compute attention scores to build logic sentence, each attention
        # score will represent whether the concept should be active or not
        # in the logic sentence.
        if self.signed:
            sign_attn = tf.nn.sigmoid(self.sign_nn(x))
        else:  # Keep only positive concepts.
            sign_attn = tf.ones_like(values, dtype=tf.float32)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        # Attention scores to identify only relevant concepts for a class
        # Shape filter_attn (batch_size, num_body_atoms, 1)
        if self.filter_num_heads > 1:
            filter_attns = []
            for h in range(self.filter_num_heads):
                filter_attn_h = self.filter_nn[h](x)
                filter_attns.append(filter_attn_h)
            filter_attn = tf.reduce_max(tf.stack(filter_attns, axis=0),
                                        axis=0, keepdims=False)
        else:
            filter_attn = self.filter_nn(x)
        # tf.print('FILTER_ATTN', tf.squeeze(filter_attn)[:10, ...])

        # filtered implemented as "or(not a, b)", corresponding to "a -> b"
        filtered_values = self.logic.disj_pair(self.logic.neg(filter_attn),
                                               sign_terms)
        # filtered_values = self.logic.imply_pair(filter_attn, sign_terms)
        # tf.print('FILTER_VALUES', filtered_values, summarize=3)

        # generate minterm
        preds = tf.squeeze(self.logic.conj(filtered_values, axis=1), axis=1)
        # tf.print('PREDS', preds.shape, preds, summarize=3)

        if constrastive_task_loss_weight > 0.0:
            num_body_atoms = tf.shape(filter_attn)[1]
            # Count the number of selected concepts for each entry.
            # Shape (batch_size,)
            filter_attn_count = tf.squeeze(tf.reduce_sum(tf.cast(filter_attn > 0.5, tf.int32),
                                                         axis=1, keepdims=False), axis=-1)
            # Repeat each filter entry as many times there are selected concepts.
            # Shape (num_filter_attn_gt_0.5, num_body_atoms, 1)
            filter_attn_contr = tf.repeat(filter_attn, repeats=filter_attn_count, axis=0)
            # Select the indices of the selected concepts.
            # Shape (num_filter_attn_gt_0.5,)
            filter_attr_indices = tf.reshape(tf.where(filter_attn > 0.5)[:, 1], shape=(-1,))
            # Repeat each sign entry as many times there are selected concepts.
            # Shape (num_filter_attn_gt_0.5, num_body_atoms, 1)
            sign_terms_contr = tf.repeat(sign_terms, repeats=filter_attn_count, axis=0)
            # Create a mask for the selected concept to flip in each input.
            # Shape (num_filter_attn_gt_0.5, num_body_atoms, 1)
            sign_terms_contr_mask = tf.expand_dims(tf.one_hot(filter_attr_indices,
                                                              depth=num_body_atoms,
                                                              dtype=tf.int32),
                                                   axis=-1)
            # Use the mask to flip exactly one selected concept for each input.
            # e.g. this is selective intervention on one selected concept.
            sign_terms_contr = tf.where(sign_terms_contr_mask == 1,
                                        self.logic.neg(sign_terms_contr), sign_terms_contr)
            # Recompute the outputs after the flips.
            filtered_values_contr = self.logic.disj_pair(self.logic.neg(filter_attn_contr),
                                                         sign_terms_contr)
            preds_contr = tf.squeeze(self.logic.conj(filtered_values_contr, axis=1), axis=1)
            # Repeat the task preds to shape it like the contrastive outputs.
            baseline_preds_contr = tf.repeat(preds, repeats=filter_attn_count, axis=0)
            # Compute the loss by forcing task preds to be flipped as well when performing
            # the interventions on the concepts.
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss = (constrastive_task_loss_weight /
                    tf.cast(num_body_atoms, dtype=tf.float32) *
                    tf.reduce_sum(bce(preds_contr, self.logic.neg(
                        baseline_preds_contr))))
            # TODO: Should we make the baseline preds crisp to make the bce be used
            # in a more standard way? Or should we just use mse here?
                        # tf.where(baseline_preds_contr > 0.5, 1.0, 0.0)))))  # crispify the baseline.
            self.add_loss(loss)
            self.add_metric(loss, name='constrastive_loss')

        if return_explain_info:
            return {'x': x, 'c': c, 'preds': preds, 'sign_attn': sign_attn, 'filter_attn': filter_attn}
        else:
            return preds

    def explain(self, explain_info: Dict[str, tf.Tensor]):
        x_unused, c, y_preds, sign_attn_mask, filter_attn_mask = \
            explain_info['x'], explain_info['c'], explain_info['preds'], \
            explain_info['sign_attn'], explain_info['filter_attn']
        mode = 'global'
        c = tf.cast(tf.squeeze(c), tf.float32)

        concept_names = [str(a) for a in self.rule.body]
        class_names = [str(a) for a in self.rule.head]

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(y_preds)):
            prediction = y_preds[sample_idx] > 0.5
            # Select 1 and flatten.
            active_classes = np.argwhere(prediction).reshape(-1)
            if len(active_classes) == 0:
                # print('Skipping ', sample_idx, 'PREDS', y_preds[sample_idx])
                # If no class is predicted, then we cannot extract any
                # explanation.
                explanations.append(
                    {'class': -1, 'explanation': '', 'attention': [], })
                continue

            # Otherwise we can extract an explanation for each active class!
            # e.g. we expain positive classifications.
            #print('A', sample_idx, active_classes)
            for target_class in active_classes:
                minterm = []
                for concept_idx in range(len(concept_names)):
                    c_pred = c[sample_idx, concept_idx]
                    sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]
                    filter_attn = filter_attn_mask[sample_idx, concept_idx, target_class]

                    # A concept is relevant <-> the filter attention score is
                    # lower than the concept probability.
                    sign_terms = self.logic.iff_pair(sign_attn, c_pred)

                    # If we are here we know that the prediction is positive, then all of
                    # self.logic.disj_pair(self.logic.neg(filter_attn), sign_terms) > 0.5
                    # Here, we only care of entries for which the attention is on, the others
                    # got assigned the neutral element and had no effect.
                    if filter_attn > 0.5:
                        if sign_attn >= 0.5:
                            # Concept is relevant and the sign is positive.
                            if mode == 'exact':
                                minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                            else:
                                minterm.append(f'{concept_names[concept_idx]}')
                        else:
                            # Concept is relevant and the sign is negative.
                            if mode == 'exact':
                                minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                            else:
                                minterm.append(f'~{concept_names[concept_idx]}')

                # add explanation to list
                target_class_name = class_names[target_class]
                minterm = ' & '.join(minterm)
                all_class_explanations[target_class_name].append(minterm)
                explanations.append({
                    'sample-id': sample_idx,
                    'class': target_class_name,
                    'explanation': minterm,
                })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class ConceptReasoningLayerUnfiltered(Layer):
    def __init__(self, rule: Rule, emb_size: int,
                 logic: Logic = GodelTNorm()):
        super().__init__()
        self.rule = rule
        self.emb_size = emb_size
        self.n_classes = len(rule.head)
        self.logic = logic
        self.sign_nn = keras.Sequential([
            Dense(emb_size, activation=None, name='sign_1'),
            LeakyReLU(),
            Dense(self.n_classes, activation=None, name='sign_2')])
        self._verbose = False

    def call(self, inputs, return_explain_info: bool=False):
        # Change x to be a sigle embedding and make the sign and filter
        # function to expand to the nun_body_atoms.
        # x.shape = (batch_size, num_body_atoms, embedding_size)
        # c.shape = (batch_size, num_body_atoms, 1) TO CHECK
        c, x = inputs
        values = tf.tile(c, multiples=(1, 1, self.n_classes))

        # compute attention scores to build logic sentence, each attention
        # score will represent whether the concept should be active or not
        # in the logic sentence.
        sign_attn = tf.nn.sigmoid(self.sign_nn(x))
        sign_terms = self.logic.iff_pair(sign_attn, values)
        if self._verbose:
            tf.print('SIGN', tf.concat([sign_attn, values, sign_terms], axis=-1), summarize=10)

        # generate minterm
        preds = tf.squeeze(self.logic.conj(sign_terms, axis=1), axis=1)
        # tf.print('PREDS', preds.shape, preds, summarize=3)

        if return_explain_info:
            return {'x': x, 'c': c, 'preds': preds, 'sign_attn': sign_attn}
        else:
            return preds

    def explain(self, explain_info: Dict[str, tf.Tensor]):
        x_unused, c, y_preds, sign_attn_mask = \
            explain_info['x'], explain_info['c'], explain_info['preds'], \
            explain_info['sign_attn']
        mode = 'global'
        c = tf.cast(tf.squeeze(c), tf.float32)

        concept_names = [str(a) for a in self.rule.body]
        class_names = [str(a) for a in self.rule.head]

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(y_preds)):
            prediction = y_preds[sample_idx] > 0.5
            # Select 1 and flatten.
            active_classes = np.argwhere(prediction).reshape(-1)
            if len(active_classes) == 0:
                # print('Skipping ', sample_idx, 'PREDS', y_preds[sample_idx])
                # If no class is predicted, then we cannot extract any
                # explanation.
                explanations.append(
                    {'class': -1, 'explanation': '', 'attention': [], })
                continue

            # Otherwise we can extract an explanation for each active class!
            # e.g. we expain positive classifications.
            #print('A', sample_idx, active_classes)
            for target_class in active_classes:
                minterm = []
                for concept_idx in range(len(concept_names)):
                    c_pred = c[sample_idx, concept_idx]
                    sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]

                    # A concept is relevant <-> the filter attention score is
                    # lower than the concept probability.
                    sign_terms = self.logic.iff_pair(sign_attn, c_pred)

                    if sign_attn >= 0.5:
                        # Concept is relevant and the sign is positive.
                        if mode == 'exact':
                            minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                        else:
                            minterm.append(f'{concept_names[concept_idx]}')
                    else:
                        # Concept is relevant and the sign is negative.
                        if mode == 'exact':
                            minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                        else:
                            minterm.append(f'~{concept_names[concept_idx]}')

                # add explanation to list
                target_class_name = class_names[target_class]
                minterm = ' & '.join(minterm)
                all_class_explanations[target_class_name].append(minterm)
                explanations.append({
                    'sample-id': sample_idx,
                    'class': target_class_name,
                    'explanation': minterm,
                })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations

