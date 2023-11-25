from typing import Union, Optional, Any, Callable
import tensorflow as tf
from tensorflow import keras
import numpy as np

Layer = keras.layers.Layer


class BasicLayer(Layer):
    """
    Customize basic layer: create a layer of x@w+b.
    :param output_dim(int): the dimension of output
    :param activation(str): the type of activation function, 
        default to None--do not enable Activation function, choose from:'relu','sigmoid','tanh'
    :param normalization(bool): whether to normalize the data before activation function
    """

    def __init__(
        self,
        output_dim: int,
        activation: str = None,
        use_bias: bool = True,
        trainable: bool = True,
        normalization: bool = False,
        **kwargs,
    ):
        super().__init__(trainable=trainable, **kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.normalization = normalization
        if activation is not None:
            assert isinstance(activation, str)
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.output_dim],
            trainable=self.trainable,
            initializer=tf.keras.initializers.RandomNormal(0, 0.3))
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[int(self.output_dim)],
                trainable=self._trainable,
                initializer=tf.keras.initializers.Constant(0.1))
        self.built = True

    def call(self, inputs: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        out = tf.matmul(inputs, self.kernel) + self.bias

        if self.activation is not None:
            if self.activation == 'relu':
                out = tf.nn.relu(out)
            elif self.activation == 'sigmoid':
                out = tf.nn.sigmoid(out)
            elif self.activation == 'tanh':
                out = tf.nn.tanh(out)

        if self.normalization:
            norm_layer = tf.keras.layers.LayerNormalization()
            out = norm_layer(out)

        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super().get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
