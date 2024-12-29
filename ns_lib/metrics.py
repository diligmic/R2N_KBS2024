import tensorflow as tf
from keras.metrics import Metric

class MRRMetric():

    def __init__(self):
        self.__name__ = "mrr"
        self.name = "mrr"

    def __call__(self, y_true, y_pred):
        # if isinstance(y_pred, tf.RaggedTensor):
        #     y_pred = y_pred.to_tensor(default_value=0.)
        # if isinstance(y_true, tf.RaggedTensor):
        #     y_true = y_true.to_tensor(default_value=0)
        ranks_list = 1 + tf.argsort(tf.argsort(y_pred, direction="DESCENDING", axis=-1, stable=True), axis=-1, stable=True)
        rank_target = tf.gather_nd(ranks_list, tf.where(y_true > 0))
        rank_target = tf.cast(rank_target, tf.float32)
        mrr = tf.reduce_mean(1.0 / rank_target)

        return mrr
