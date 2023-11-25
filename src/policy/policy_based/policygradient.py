import gymnasium as gym
from typing import Any, Dict, Union, Optional
from src.policy import BasePolicy
from src.data.data_stru import Flow
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class PolicyGradient(BasePolicy):
    """
    Implementation of policygradient policy.
    """

    def __init__(
            self,
            env: Any,
            actor: keras.Model,  # 策略网络
            actor_optim: tf.optimizers.Optimizer,  # actor优化器
            reward_normalization: bool = False,
            discount_factor: float = 0.9,  # 回报折扣系数
            action_scaling: bool = True,  # 动作范围重映射到环境动作空间
            action_bound_method: str = 'clip',  # 动作截断
            base_line: bool = False,  # 是否带有基线
            # 基线价值函数（需要与动作无关，所以使用价值函数）
        critic: Optional[keras.Model] = None,
            # critic优化器
            critic_optim: Optional[tf.optimizers.Optimizer] = None,
            **kwargs: Any) -> None:
        super().__init__(action_scaling=action_scaling,
                         action_bound_method=action_bound_method,
                         env=env,
                         **kwargs)
        self._actor = actor
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._actor_optim = actor_optim
        self._rew_norm = reward_normalization
        self.eps = np.finfo(np.float32).eps
        self._update_method = 'offpolicy'
        self._base_line = base_line
        if self._base_line is True:
            assert critic is not None and critic_optim is not None
            self._critic = critic
            self._critic_optim = critic_optim

    def forward(
        self,
        obs: Union[np.ndarray, tf.Tensor],
    ) -> tf.Tensor:
        """
        Compute action over the given obs data.
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        if obs.ndim == 1:
            # 将state转化为(batch,None)的矩阵
            obs = tf.expand_dims(obs, 0)
        # 连续动作
        if self._action_type == 'continuous':
            logits = self._actor(obs)
            if isinstance(logits, tuple):  # 连续动作概率分布
                dist = tfp.distributions.Normal(*logits)
                act = dist.sample()
            else:
                act = logits  # 连续动作值
            if self._action_scaling:
                act = self.map_action(act)
        # 离散动作
        elif self._action_type == 'discrete':
            logits = self._actor(obs)
            act = tf.random.categorical(logits, 1)[0, 0]
            act = act.numpy()
        return act

    def learn(self, input_data: list[Flow], **kwargs: Any) -> Dict[str, Any]:
        if self._base_line is False:
            for flow in input_data:
                if self._action_type == 'continuous':
                    pass
                elif self._action_type == 'discrete':
                    with tf.GradientTape() as tape:
                        all_logits = self._actor(flow['obs'])
                        all_probs = tf.nn.log_softmax(all_logits, axis=1)
                        all_act_probs = tf.reduce_sum(
                            all_probs *
                            tf.one_hot(flow['act'], depth=all_logits.shape[1]),
                            axis=1)
                        actor_loss = - \
                            tf.reduce_mean(flow['returns'][0]*all_act_probs)
                    grads = tape.gradient(actor_loss,
                                          self._actor.trainable_variables)
                    self._actor_optim.apply_gradients(
                        zip(grads, self._actor.trainable_variables))
        else:
            for flow in input_data:
                if self._action_type == 'continuous':
                    pass
                elif self._action_type == 'discrete':
                    with tf.GradientTape(persistent=True) as tape:
                        all_logits = self._actor(flow['obs'])
                        all_probs = tf.nn.log_softmax(all_logits, axis=1)
                        all_act_probs = tf.reduce_sum(
                            all_probs *
                            tf.one_hot(flow['act'], depth=all_logits.shape[1]),
                            axis=1)
                        actor_loss = - \
                            tf.reduce_mean(flow['returns']*all_act_probs-self._critic(flow['obs']))
                        critic_loss = tf.reduce_mean(
                            (flow['returns'] - self._critic(flow['obs'])**2))
                    grads1 = tape.gradient(actor_loss,
                                           self._actor.trainable_variables)
                    grads2 = tape.gradient(critic_loss,
                                           self._critic.trainable_variables)
                    self._actor_optim.apply_gradients(
                        zip(grads1, self._actor.trainable_variables))
                    self._critic_optim.apply_gradients(
                        zip(grads2, self._critic.trainable_variables))
                    del tape


# if __name__=="__main__":
# env = gym.make('MountainCarContinuous-v0',render_mode="human")
# actor = tf.keras.Sequential([tf.keras.layers.Dense(10,'relu')])
# actor_optim = tf.keras.optimizers.Adam(0.1)
# a = Reinforce(env,actor,actor_optim,wz=1)

# flow=[
#         Flow({'act': np.array([[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]),
#             'obs': np.array([4, 5, 6, 7, 8, 9]),
#             'terminated':np.array([0, 0, 0, 0, 0, 1]),
#             'truncated':np.array([0, 0, 0, 0, 0, 0]),
#             'rew':np.array([0.2, 0.3, 0.1, 0.6, 0.8, 0])
#             }),

#         Flow({'act': np.array([[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]),
#             'obs': np.array([4, 5, 6, 7, 8, 9]),
#             'terminated':np.array([0, 0, 0, 0, 0, 1]),
#             'truncated':np.array([0, 0, 0, 0, 0, 0]),
#             'rew':np.array([0.2, 0.3, 0.1, 0.6, 0.8, 0])
#             }),
#         ]

# b = a.compute_episodic_return(flow, a._gamma)
# print(b)
