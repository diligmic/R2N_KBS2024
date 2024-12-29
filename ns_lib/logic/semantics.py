import tensorflow as tf
import abc


class Logic:
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, a, axis=-1):
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a, axis=-1):
        raise NotImplementedError

    def conj_pair(self, a, b):
        raise NotImplementedError

    def disj_pair(self, a, b):
        raise NotImplementedError

    def iff_pair(self, a, b):
        raise NotImplementedError

    # Implements modus ponens for the selected semantics.
    def imply_pair(self, a, b):
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        raise NotImplementedError


class ProductTNorm(Logic):
    def __init__(self):
        super().__init__()
        self.current_truth = tf.constant(1)
        self.current_false = tf.constant(0)

    def update(self):
        pass

    def conj(self, a, axis=-1):
        return tf.reduce_prod(a, axis=axis, keepdims=True)

    def conj_pair(self, a, b):
        return a * b

    def disj(self, a, axis=-1):
        return 1 - tf.reduce_prod(1 - a, axis=axis, keepdims=True)

    def disj_pair(self, a, b):
        return a + b - a * b

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b),
                              self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1 - a

    def imply_pair(self, a, b):
        a = tf.minimum(a, 1e-6)
        b = tf.minimum(b, 1e-6)
        return tf.where(a > b, 1.0, tf.math.divide_no_nan(b, a))

    def predict_proba(self, a):
        return a.squeeze(-1)

class GodelTNorm(Logic):
    def __init__(self):
        super().__init__()
        self.current_truth = tf.constant(1)
        self.current_false = tf.constant(0)

    def update(self):
        pass

    def conj(self, a, axis=-1):
        return tf.reduce_min(a, axis=axis, keepdims=True)

    def disj(self, a, axis=-1):
        return tf.reduce_min(a, axis=axis, keepdims=True)

    def conj_pair(self, a, b):
        return tf.minimum(a, b)

    def disj_pair(self, a, b):
        return tf.maximum(a, b)

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b),
                              self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1.0 - a

    def imply_pair(self, a, b):
        return tf.where(a > b, 1.0, b)

    def predict_proba(self, a):
        return a.squeeze(-1)


class SumProductSemiring(Logic):
    def __init__(self):
        super().__init__()
        self.current_truth = tf.constant(1)
        self.current_false = tf.constant(0)

    def update(self):
        pass

    def conj(self, a, axis=-1):
        return tf.reduce_prod(a, axis=axis, keepdims=True)

    def conj_pair(self, a, b):
        return a * b

    def disj(self, a, axis=-1):
        return tf.reduce_sum(a, axis=axis, keepdims=True)

    def disj_pair(self, a, b):
        return a + b

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b),
                              self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1 - a

    def imply_pair(self, a, b):
        a = tf.minimum(a, 1e-6)
        b = tf.minimum(b, 1e-6)
        return tf.where(a > b, 1.0, tf.math.divide_no_nan(b, a))

    def predict_proba(self, a):
        return a.squeeze(-1)

#
# class LukaTNorm(Semiring):
#
#     def conj_n(self, x, axis):
#         return tf.maximum(tf.reduce_sum(x - 1., axis=axis) + 1., 0.)
#
#     def conj(self, x, y):
#         return tf.maximum( x + y - 1., 0.)
#
#     def disj_n(self, x, axis):
#         x = tf.math.reduce_sum(x, axis=axis)
#         x = tf.minimum(1., x)
#         return x
#
#     def disj(self, x, y):
#         return tf.minimum(1., x + y)
#
#     def disj_scatter(self, base, update, indices):
#         indices = tf.expand_axiss(indices, -1)
#         add = tf.tensor_scatter_nd_add(base, indices, update)
#         res = tf.minimum(1., add)
#         return res
#
#     def implies(self, x ,y):
#         "x -> y"
#         t = tf.minimum(1., 1 - x + y)
#         return t
#
#
# class Max(Semiring):
#
#     def conj_n(self, x, axis):
#         return tf.reduce_min(x, axis=axis)
#
#     def conj(self, x, y):
#         return tf.minimum(x,y)
#
#     def disj_n(self, x, axis):
#         x = tf.reduce_max(x, axis=axis)
#         return x
#
#     def disj(self, x, y):
#         return tf.maximum(x, y)
#
#     def disj_scatter(self, base, update, indices):
#         indices = tf.expand_axiss(indices, -1)
#         res = tf.tensor_scatter_nd_max(base, indices, update)
#         return res
#
#
# class ProductTNorm(Semiring):
#
#     def conj_n(self, x, axis):
#         return tf.reduce_prod(x, axis=axis)
#
#     def conj(self, x, y):
#         return x * y
#
#     def disj_n(self, x, axis):
#         x = 1 - x
#         x = tf.math.reduce_prod(x, axis=axis)
#         x = 1 - x
#         return x
#
#     def disj(self, x, y):
#         return 1 - (1-x)*(1-y)
#
#     def disj_scatter(self, base, update, indices):
#         embeddings = 1.0 - update
#         embeddings = tf.where(embeddings < eps, embeddings + eps, embeddings)
#         log_clique_embeddings = tf.math.log(embeddings)
#
#         indices = tf.expand_axiss(indices, -1)
#         aggregated_log_sum = tf.tensor_scatter_nd_add(base, indices,
#                                                       log_clique_embeddings)
#         aggregated_product = tf.math.exp(aggregated_log_sum)
#         res = 1.0 - aggregated_product
#         return res
#
#
#
#
# class LogSumProduct(Semiring):
#
#     def conj_n(self, x, axis):
#         return tf.reduce_sum(x, axis=axis)
#
#     def conj(self, x, y):
#         return x + y
#
#     def disj_n(self, x, axis):
#         x = tf.math.exp(x)
#         x = tf.reduce_sum(x, axis=axis)
#         x = tf.math.log(x)
#         return x
#
#     def disj(self, x, y):
#         x = tf.math.exp(x)
#         y = tf.math.exp(y)
#         a = x + y
#         a = tf.log(a)
#         return a
#
#     def disj_scatter(self, base, update, indices):
#         indices = tf.expand_axiss(indices, -1)
#         update = tf.math.exp(update)
#         aggregated_sum = tf.tensor_scatter_nd_add(base, indices, update)
#         res = tf.math.log(aggregated_sum)
#         return res
