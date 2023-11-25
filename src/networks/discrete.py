from src.networks import MLBasicLayer
import tensorflow as tf
import numpy as np
from tensorflow import keras
from typing import List, Optional, Sequence, Union


class Actor(keras.Model):
    """
    Simple actor network will create an actor operated in discrete 
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single layer).
    :param trainable: whether network parameters can be trained.

    """

    def __init__(self,
                 preprosses_net: keras.Model,
                 action_shape: Sequence[int],
                 hidden_sizes: Sequence[int] = [],
                 activations: Optional[Union[str, List[str]]] = None,
                 norm_layers: bool = False,
                 norm_layers_pos: Optional[List[int]] = None,
                 softmax: bool = False,
                 trainable: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preprosses_net = preprosses_net
        self._output_dim = np.prod(action_shape)
        self._input_dim = getattr(preprosses_net, '_output_dim')
        self._softmax = softmax
        self._trainable = trainable
        self.backnet = MLBasicLayer(self._input_dim,
                                    self._output_dim,
                                    hidden_sizes,
                                    activations=activations,
                                    norm_layers=norm_layers,
                                    norm_layers_pos=norm_layers_pos,
                                    trainable=self._trainable)

    def summary(self):
        self._preprosses_net.summary()
        self.backnet.summary()

    def call(self, obs: np.ndarray | tf.Tensor) -> tf.Tensor:
        logits = self._preprosses_net(obs)
        out = self.backnet(logits)
        if self._softmax:
            out = tf.nn.softmax(out, axis=-1)
        return out


class Critic(keras.Model):
    """
    Simple critic network. Will create a q value operated in discrete 
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int output_dim: the output dimension of Critic network. Default to 1.
    """

    def __init__(self,
                 preprosses_net: keras.Model,
                 hidden_sizes: Sequence[int] = [],
                 activations: Optional[Union[str, List[str]]] = None,
                 norm_layers: bool = False,
                 norm_layers_pos: Optional[List[int]] = None,
                 softmax: bool = False,
                 qcritic: bool = False,
                 trainable: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._preprosses_net = preprosses_net
        self._input_dim = getattr(preprosses_net, '_output_dim')
        self._output_dim = 1
        self._qcritic = qcritic
        self._trainable = trainable
        self.backnet = MLBasicLayer(self._input_dim,
                                    self._output_dim,
                                    hidden_sizes,
                                    activations=activations,
                                    norm_layers=norm_layers,
                                    norm_layers_pos=norm_layers_pos,
                                    softmax=softmax,
                                    trainable=self._trainable)

    def summary(self):
        self._preprosses_net.summary()
        self.backnet.summary()

    def call(self,
             obs: np.ndarray | tf.Tensor,
             action: Optional[tf.Tensor] = None) -> tf.Tensor:
        if self._qcritic:
            if action is None:
                raise AttributeError('you should input a tensor of action')
            else:
                logits = self._preprosses_net(obs)
                action = tf.reshape(action, [-1])
                input = tf.concat([obs, action])
                output = self.backnet(input)
        else:
            logits = self._preprosses_net(obs)
            output = self.backnet(logits)

        return output
