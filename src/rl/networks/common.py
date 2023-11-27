import operator
from rl.layers.base import BasicLayer
import numpy as np
from tensorflow import keras
import tensorflow as tf
from typing import Any, Optional, Union, List, Iterator, Sequence


class MLBasicLayer(keras.Model):
    """
    Customize MLP network from Basiclayer

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param input_dim (int): dimension of the input vector.
    :param output_dim (int): dimension of the output vector. If set to 0, there is no final
        linear layer.
    :param hidden_sizes (list[int]): num and dimension of hidden layers, excluding input and
        output layers.
    :param norm_layers (bool): whether to add normalization layers between hidden layers,
        default to false.
    :param norm_layers_pos (list[int]): the position of normalized layers after hidden layers.
    :param norm_args (Tuple[epsilon: float, beta: float, gamma: float] |
        Sequence[Tuple[epsilon: float, beta: float, gamma: float]): normalization layer parameters.
    :param activation (Type[keras.layers.Layer] | List[keras.layers.Layer]):
        type and location of Activation function, default to Relu.
    :param softmax (int): does the output pass through the softmax layer.
    :param flatten (int): does the output pass through the flatten layer.

    """

    def __init__(
        self,
        input_shape: Optional[int] = None,
        output_dim: int = 0,
        hidden_sizes: List[int] = [],
        activations: Optional[Union[str, List[str]]] = None,
        norm_layers: bool = False,
        norm_layers_pos: Optional[List[int]] = None,
        softmax: bool = False,
        trainable: bool = True,
        name: Any | None = None,
        dtype: Any | None = None,
        dynamic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self._norm_layers = norm_layers
        self._softmax = softmax
        self._trainable = trainable

        layers = []
        if input_shape is not None:
            layers.append(keras.Input(shape=input_shape))

        if activations is None or operator.eq(activations, []):
            activations_list = [None] * len(hidden_sizes)
        elif isinstance(activations, list):
            assert len(activations) == len(
                hidden_sizes
            ), "Activations list must have same length of hidden_sizes list"
            activations_list = activations
        elif isinstance(activations, str):
            activations_list = [activations] * (len(hidden_sizes))
        else:
            raise TypeError("Activations has a wrong input!")

        if len(hidden_sizes) == 0:
            if output_dim == 0:
                raise AttributeError(
                    "output_dim and hidden_sizes can not equal zero at the same time"
                )
            else:
                layers.append(BasicLayer(output_dim, trainable=self._trainable))
        else:
            if self._norm_layers == True:
                if len(norm_layers_pos) > len(hidden_sizes):
                    raise RuntimeError(
                        "norm_layers must have length no longer than hidden_sizes"
                    )
                else:
                    pos_list = norm_layers_pos
                    for i in range(0, len(hidden_sizes)):
                        if i in pos_list:
                            layers.append(
                                BasicLayer(
                                    hidden_sizes[i],
                                    activation=activations_list[i],
                                    normalization=True,
                                    trainable=self._trainable,
                                )
                            )
                        else:
                            layers.append(
                                BasicLayer(
                                    hidden_sizes[i],
                                    activation=activations_list[i],
                                    normalization=False,
                                    trainable=self._trainable,
                                )
                            )
            else:
                for i in range(0, len(hidden_sizes)):
                    layers.append(
                        BasicLayer(
                            hidden_sizes[i],
                            activation=activations_list[i],
                            trainable=self._trainable,
                        )
                    )

            if output_dim != 0:
                layers.append(BasicLayer(output_dim, trainable=self._trainable))

        for i in list(reversed(layers)):
            if isinstance(i, BasicLayer):
                self._output_dim = i.output_dim
                break

        if self._softmax:
            layers.append(keras.layers.Softmax())

        self.model = keras.Sequential(layers)

    def summary(self) -> None:
        self.model.summary()

    # def info(self) -> None:
    #     info = "输入维度: {:d}\n输出维度: {:d}"
    #     print(info.format(self._input_dim, self._output_dim))

    def call(self, input: np.ndarray | tf.Tensor) -> tf.Tensor:
        if isinstance(input, np.ndarray):
            input = tf.convert_to_tensor(input)
        output = self.model(input)
        return output


if __name__ == "__main__":
    a = MLBasicLayer((10,), 5, [20, 50, 30], None)
    a.summary()
