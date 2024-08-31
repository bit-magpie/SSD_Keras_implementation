import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf


class Normalize(Layer):
    """Performs L2 normalization on the input tensor
    Args
        gamma_init: Default scaling parameter.
    Input shape
        4D tensor with shape:
        `(samples, rows, cols, channels)`
    Output shape
        The scaled tensor with same shape as input.

    """

    def __init__(self, gamma_init=20, **kwargs):
        self.axis = 3
        self.gamma_init = gamma_init
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        gamma = self.gamma_init * np.ones(shape)
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self._trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = tf.math.l2_normalize(x, self.axis)
        output *= self.gamma
        return output

    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
