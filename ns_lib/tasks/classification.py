import tensorflow as tf


class MutuallyExclusiveClassification():


    def __init__(self, competitive_groups, labels):
        self.competitive_groups = competitive_groups
        self.labels = labels

    def loss(self):

        def __inner_loss(y_true, y_pred):
            del y_true
            logits = tf.gather(params=y_pred, indices = self.competitive_groups)
            return tf.losses.categorical_crossentropy(y_true= self.labels,
                                                      y_pred = logits)
        return __inner_loss


    def metric(self):

        def __inner_loss(y_true, y_pred):
            del y_true
            logits = tf.gather(params=y_pred, indices = self.competitive_groups)
            return tf.metrics.categorical_accuracy(y_true= self.labels,
                                                      y_pred = logits)
        return __inner_loss


class BinaryClassification():


    def __init__(self, atoms, labels):
        self.atoms = atoms
        self.labels = labels

    def loss(self):

        def __inner_loss(y_true, y_pred):
            del y_true
            logits = tf.gather(params=y_pred, indices = self.atoms)
            return tf.losses.binary_crossentropy(y_true= self.labels,
                                                      y_pred = logits)
        return __inner_loss


    def metric(self):

        def __inner_loss(y_true, y_pred):
            del y_true
            logits = tf.gather(params=y_pred, indices = self.atoms)
            return tf.metrics.binary_accuracy(y_true= self.labels,
                                              y_pred = logits)
        return __inner_loss



