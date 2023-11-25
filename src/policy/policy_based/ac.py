from src.data.buffer import ReplayBuffer
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from tensorflow import keras
from src.data.data_stru import Flow
from src.policy import BasePolicy
from typing import Any, Optional, Union

tfd = tfp.distributions
tfb = tfp.bijectors


class ACPolicy(BasePolicy):
    """
    Implementation of actor-critic algorithm.
    """

    def __init__(
            self,
            evn: Any,
            actor: keras.Model,  # 策略网络
            critic: keras.Model,  # 价值网络
            optim1: tf.optimizers,  # 动作网络优化器
            optim2: tf.optimizers,  # 策略网络优化器       
            discount_factor: float = 0.9,  # 回报折扣系数
            action_scaling: bool = True,  # 动作范围重映射到环境动作空间
            action_bound_method: str = 'clip',  # 动作截断
            **kwargs: Any) -> None:
        super().__init__(action_scaling=action_scaling,
                         action_bound_method=action_bound_method,
                         env=evn,
                         **kwargs)
        self._actor = actor
        self._critic = critic
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._optim1 = optim1
        self._optim2 = optim2

    def process_fn(self,
                   flow: Union[Flow, ReplayBuffer],
                   index: int | None = None) -> float:
        """
        Compute the TD-error.
        """
        if isinstance(flow, ReplayBuffer):
            assert index is not None
            flow['get'] = np.zeros_like(index, dtype=np.float64)
            for i in range(len(flow['get'])):
                if i == 0:
                    flow['get'][len(flow['get']) - i -
                                1] = flow['rew'][len(flow['get']) - i - 1]
                else:
                    flow['get'][len(flow['get'])-i-1] = flow['rew'][len(flow['get'])-i-1] + \
                        self._gamma * flow['get'][len(flow['get'])-i]
            flow['advantage'] = flow['get'] - self._critic(flow['obs'])
        # td_error = flow['rew']+self._gamma * \
        #     self._critic(flow['obs_next'])-self._critic(flow['obs'])
        # return np.float32(td_error)

        return flow

    def forward(
        self,
        obs: Union[np.ndarray, tf.Tensor],
    ) -> float:
        """
        Compute action over the given obs data.
        """
        logits = self._actor(obs)
        # print(logits)
        if isinstance(logits, tuple):  # 应对连续动作
            dist = tfp.distributions.Normal(*logits)
            act = dist.sample()
            act = tf.reshape(act, (-1, ))
            if self._action_scaling:
                act = self.map_action(act)
        else:  # 应对离散动作
            act = logits

        return act

    def learn(
        self,
        flow: Flow,
        step: int,
    ) -> None:
        # with tf.GradientTape() as tape1:
        #     loss1 = td_error*self._critic(flow['obs'], flow['act'])
        # grads1 = tape1.gradient(loss1, self._critic.trainable_variables)

        # if self._action_type=="discrete":
        #     with tf.GradientTape() as tape2:
        #         loss2 = -(self._gamma**step)*td_error*tf.math.log(self._actor(flow['obs']))
        #     grads2 = tape2.gradient(loss2, self._actor.trainable_variables)
        # else:
        #     with tf.GradientTape() as tape2:
        #         loss2 = -(self._gamma**step)*td_error*tfp.distributions.Normal(*self._actor(flow['obs'])).log_prob(flow['act'])
        #     grads2 = tape2.gradient(loss2, self._actor.trainable_variables)

        # self._optim1.apply_gradients(
        #     (zip(grads1, self._critic.trainable_variables)))
        # self._optim2.apply_gradients(
        #     (zip(grads2, self._actor.trainable_variables)))
        huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM)
        with tf.GradientTape() as tape1:
            action_log_probs = tf.math.log(self._actor(flow['act']))
            actor_loss = -tf.math.reduce_sum(
                action_log_probs * flow['advantage'])
            critic_loss = huber_loss(flow['get'], self._critic(flow['obs']))
            loss = actor_loss + critic_loss
        grads = tape1.gradient(loss, self._critic.trainable_variables)

    def update(
            self,
            flow: Union[Flow, ReplayBuffer],
            step: Optional[int] = None) -> None:  # 可以应用与online和offline trainer
        if isinstance(flow, Flow):
            assert step is not None
            td_error = self.process_fn(flow)
            self.learn(td_error, flow, step)
        else:
            for step in range(len(flow)):
                self.update(flow[step], step)
