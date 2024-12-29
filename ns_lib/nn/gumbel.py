import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Any, Dict, List, Union, Tuple
import tensorflow_probability as tfp

"""Gumbel-Softmax layer.
"""
class GumbelSoftmax(Layer):

    def __init__(self, temperature: float, axis: int=-1, hard: bool=True,
                 **kwargs) -> None:
        """Initialization method.

        Args:
            axis: Axis to perform the softmax operation.
            temperature: Gumbel-Softmax temperature parameter.

        """

        super().__init__(**kwargs)
        self.temperature = temperature
        self.axis = axis
        self.hard = hard
        assert self.temperature > 0.0

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            The Gumbel-Softmax sample.

        """

        x = inputs + self._gumbel_distribution(tf.shape(inputs))
        x = tf.nn.softmax(x / self.temperature, self.axis)
        if self.hard:
            argmax_mask = tf.one_hot(tf.argmax(x, self.axis, tf.int32),
                                     tf.shape(x)[self.axis], axis=self.axis)
            x = tf.where(argmax_mask == 1.0, x, 0.0)
        return x

    def _gumbel_distribution(self, input_shape: Tuple[int, ...]) -> tf.Tensor:
        """Samples a tensor from a Gumbel distribution.

        Args:
            input_shape: Shape of tensor to be sampled.

        Returns:
           (tf.Tensor): An input_shape tensor sampled from a Gumbel distribution.

        """

        uniform_dist = tf.random.uniform(input_shape, 0.0, 1.0)
        gumbel_dist = -tf.math.log(1e-6 - tf.math.log(uniform_dist + 1e-6))
        return gumbel_dist

    def get_config(self) -> Dict[str, Any]:
        """Gets the configuration of the layer for further serialization.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config = {"axis": self.axis}
        config.update(super().get_config())
        return config


# Take a generic layer and gumbelifies it.
class MakeGumbel(Layer):

    # The passed layer must fold a shape (..., N) into (..., 1) like for example
    # dense layer.
    def __init__(self, layer, temperature, **kwargs):
        super().__init__(**kwargs)
        self.parameter_bernoulli = layer
        assert temperature > 0.0
        self.coolness = 1. / temperature

    def call(self, inputs, *args, **kwargs):
        logits = self.parameter_bernoulli(inputs)
        dist = tfp.distributions.Logistic(logits * self.coolness, self.coolness)
        concept_output = tf.sigmoid(dist.sample())
        concept_output = tf.squeeze(concept_output, axis=-1)
        return concept_output

"""Sampling of the element of a tensor element-wise layer."""
class SamplingLayer(Layer):

    def __init__(self, minval=0.0, maxval=1.0, **kwargs) -> None:
        """Initialization method.
        """

        super().__init__(**kwargs)
        self.maxval = maxval
        self.minval = minval

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.where(inputs > tf.random.uniform(tf.shape(inputs), minval=minval, maxval=maxval),
                     inputs, 0.0)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Gets the configuration of the layer for further serialization.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config = {"maxval": self.maxval,
                  "minval": self.minval}
        config.update(super().get_config())
        return config
