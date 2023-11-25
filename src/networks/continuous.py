from typing import Optional, List, Tuple, Union, Sequence
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
from src.networks.common import MLBasicLayer

SIGMA_MIN = -2
SIGMA_MAX = 20


class Actor(keras.Model):
    """
    Simple actor network will create an actor operated in continuous 
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
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
                 max_action: float = 1.0,
                 trainable: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preprosses_net = preprosses_net
        self._output_dim = np.prod(action_shape)
        self._input_dim = getattr(preprosses_net, '_output_dim')
        self._max_action = max_action
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

    def call(self, obs: np.ndarray | tf.Tensor) -> tf.Tensor:
        logits = self._preprosses_net(obs)
        logits = self.backnet(logits)
        out = tf.nn.tanh(logits) * self._max_action
        return out


class Critic(keras.Model):
    """
    Simple critic network. Will create an critic operated in continuous 
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.
    """

    def __init__(self,
                 preprosses_net: keras.Model,
                 hidden_sizes: Sequence[int] = [],
                 activations: Optional[Union[str, List[str]]] = None,
                 norm_layers: bool = False,
                 norm_layers_pos: Optional[List[int]] = None,
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
                                    trainable=self._trainable)

    def summary(self):
        self._preprosses_net.summary()
        self.backnet.summary()

    def call(self,
             obs: np.ndarray | tf.Tensor,
             action: Optional[tf.Tensor] = None) -> tf.Tensor:
        if self._qcritic:
            assert action is not None
            logits = self._preprosses_net(obs)
            action = tf.reshape(action, [])
            input = tf.concat([obs, action])
            output = self.backnet(input)
        else:
            logits = self._preprosses_net(obs)
            output = self.backnet(logits)

        return output


class ActorProb(keras.Model):
    """
    Simple actor network (output with a united gauss probability distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.
    """

    def __init__(self,
                 preprosses_net: keras.Model,
                 action_shape: Sequence[int],
                 hidden_sizes: Sequence[int] = [],
                 activations: Optional[Union[str, List[str]]] = None,
                 norm_layers: bool = False,
                 norm_layers_pos: Optional[List[int]] = None,
                 softmax: bool = False,
                 max_action: float = 1.0,
                 trainable: bool = True,
                 unbounded: bool = False,
                 conditioned_sigma: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preprosses_net = preprosses_net
        self._output_dim = np.prod(action_shape)
        self._input_dim = getattr(preprosses_net, '_output_dim')
        self._trainable = trainable
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn(
                "Note that max_action input will be discarded when unbounded is True."
            )
            max_action = 1.0
        self.mu = MLBasicLayer(self._input_dim,
                               self._output_dim,
                               hidden_sizes,
                               activations=activations,
                               norm_layers=norm_layers,
                               norm_layers_pos=norm_layers_pos,
                               softmax=softmax,
                               trainable=self._trainable)
        self._c_sigma = conditioned_sigma
        if self._c_sigma:  # 用于SAC
            self.sigma = MLBasicLayer(self._input_dim,
                                      self._output_dim,
                                      hidden_sizes,
                                      activations=activations,
                                      norm_layers=norm_layers,
                                      norm_layers_pos=norm_layers_pos,
                                      softmax=softmax,
                                      trainable=self._trainable)  # sigma为可训练网络
        else:
            self.sigma_variable = tf.Variable(
                tf.zeros([self._output_dim]),
                trainable=self._trainable)  # sigma为可训练张量
        self._max_action = max_action
        self._unbounded = unbounded

    def call(self, obs: np.ndarray | tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits = self._preprosses_net(obs)
        # 动作的中值
        mu = self.mu(logits)
        mu = tf.squeeze(mu, axis=0)
        # 将动作中值限定在最大动作范围内
        if not self._unbounded:
            mu = tf.nn.tanh(mu) * self._max_action
        if self._c_sigma:
            sigma = tf.exp(
                tf.clip_by_value(self.sigma(logits), SIGMA_MIN, SIGMA_MAX))
        else:
            sigma = tf.exp(self.sigma_variable)

        return (mu, sigma)
