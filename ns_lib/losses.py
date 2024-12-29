from keras.losses import BinaryCrossentropy, Loss, binary_crossentropy
import tensorflow as tf
from ns_lib.logic.commons import RuleGroundings, Rule
from ns_lib.logic import RuleGroundings
from typing import List
from keras.layers import Layer

class WeightedBinaryCrossEntropy(Loss):

    def __init__(self, weight_0 = 1, weight_1=1):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=None)
        self.weight_0 = weight_0
        self.weight_1 = weight_1

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true.to_tensor(0), [-1])
        y_pred = tf.reshape(y_pred.to_tensor(0), [-1])
        s = binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(tf.where(y_true>0.5, self.weight_1 * s, self.weight_0*s))



class SemanticBasedRegularizer(Layer):

    def __init__(self, rules: List[Rule],  semiring, learn_weights = False, weight = 1.):
        super().__init__()
        self.rule_weights = {}
        self.rules = rules
        for r in self.rules:
            if learn_weights:
                w = self.add_weight(shape = (), initializer = tf.zeros_initializer(), trainable=True)
            else:
                w = r.weight
            self.rule_weights[r.name] = w
        self.semiring = semiring
        self.weight = weight


    def call(self, inputs,*args, **kwargs):

            loss = tf.constant(0.)
            predictions, formula_to_atom_tuples = inputs
            for rule in self.rules:
                (A_in, A_out) = formula_to_atom_tuples[rule.name]

                body = tf.gather(params=predictions, indices=A_in)
                head = tf.gather(params=predictions, indices=A_out)
                values = self.semiring.implies(x=self.semiring.conj_n(body,axis=-1),
                                               y=self.semiring.disj_n(head, axis=-1))
                loss += self.rule_weights[rule.name] * (1 - tf.reduce_mean(values))
            return self.weight * loss





