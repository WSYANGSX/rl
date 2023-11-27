from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary, Box
from rl.data.buffer import ReplayBuffer
import tensorflow as tf
from numba import njit
import numpy as np
from rl.data.data_stru import Flow
from tensorflow import keras
from abc import ABC, abstractmethod


class BasePolicy(ABC, keras.Model):
    """The base class for any RL policy.

    It comes into several classes of policies. All of the policy classes must inherit
    :class:`policy.BasePolicy`.

    A policy class typically has the following parts:

    * :meth:`policy.BasePolicy.__init__`: initialize the policy, including 
        coping the target network and so on;
    * :meth:`policy.BasePolicy.forward`: compute action with given observation;
    * :meth:`policy.BasePolicy.process_fn`: pre-process data from the replay buffer;
    * :meth:`policy.BasePolicy.learn`: update policy with a given batch of data.
    * :meth:`policy.BasePolicy.post_process_fn`: update the replay buffer from the learning process
        (e.g., prioritized replay buffer needs to update the weight);
    * :meth:`policy.BasePolicy.update`: the main interface for training,  
        i.e., `process_fn -> learn -> post_process_fn`.

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation "obs" (may be a ``numpy.ndarray``, a ``torch.Tensor``,  \
        ), hidden state "state" (for RNN usage), and other information \
        "info" provided by the environment.
    2. Output: some "logits", the next hidden state "state", and the intermediate \
    result during policy forwarding procedure "policy". The "logits" could be a tuple \
    instead of a ``torch.Tensor``. It depends on how the policy process the network \
    output. For example, in PPO, the return of the network might be \
    ``(mu, sigma), state`` for Gaussian policy. The "policy" can be a Flow of \
    tf.Tensor or other things, which will be stored in the replay buffer, and can \
    be accessed in the policy update process (e.g. in "policy.learn()", the \
    "flow.policy" is what you need).

    Since :class:`policy.BasePolicy` inherits ``torch.nn.Module``, you can
    use :class:`policy.BasePolicy` almost the same as ``keras.Model``,
    for instance, loading and saving the model.
    """
    agent_id = []

    def __init__(self,
                 env: Any,
                 action_scaling: bool = False,
                 action_bound_method: str = "",
                 discount_factor: float = 0.9,
                 **kwargs: Any) -> None:
        super().__init__()
        self._env = env
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        if isinstance(self._action_space,
                      (Discrete, MultiDiscrete, MultiBinary)):
            self._action_type = "discrete"
        elif isinstance(self._action_space, Box):
            self._action_type = "continuous"
        self._updating = False
        self._action_scaling = action_scaling
        # can be one of ("clip", "tanh", ""), empty string means no bounding
        assert action_bound_method in ("", "clip", "tanh")
        self._action_bound_method = action_bound_method
        self._gamma = discount_factor
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                self.__dict__[key] = value

    def set_agent_id(self, agent_id: int) -> None:
        assert agent_id not in BasePolicy.agent_id, f"agent_id {agent_id} has already exited!"
        self._agent_id = agent_id
        BasePolicy.agent_id.append(agent_id)

    def exploration_noise(self, act: Union[np.ndarray, tf.Tensor],
                          flow: Flow) -> Union[np.ndarray, tf.Tensor]:
        """
        Modify the action from policy.forward with exploration noise.

        :param act: tf.Tensor or numpy.ndarray which is the action output by
            policy.forward.

        :param batch: the input flow for policy.forward, kept for advanced usage.

        :return: action in the same form of input "act" but with added exploration
            noise.
        """
        pass

    def soft_update(self, tgt: keras.Model, : keras.Model,
                    tau: float) -> None:
        """Softly update the parameters of target module towards the parameters of source module."""
        pass

    @abstractmethod
    def forward(
        self,
        flow: Flow,
    ) -> Union[np.ndarray, tf.Tensor]:
        """
        Compute action over the given flow data.

        :return: np.ndarray or tf.Tensor of action.
        """
        pass

    def map_action(
            self, act: Union[np.ndarray,
                             tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
        """
        Map raw network output to action range in env.action_space
        """
        if self._action_bound_method == "clip":
            act = np.clip(act, -1.0, 1.0)
        elif self._action_bound_method == "tanh":
            act = np.tanh(act)
        if self._action_scaling:
            assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                "action scaling only accepts raw action range = [-1, 1]"
            low, high = self._action_space.low, self._action_space.high
            act = low + (high - low) * (act + 1.0) / 2.0
        return act

    # def map_action_inverse(self,act:Union[np.ndarray,tf.Tensor])->Union[np.ndarray,tf.Tensor]:
    #     """Inverse operation to :meth:`policy.BasePolicy.map_action`.

    #     This function is called in :meth:`Collector.collect` for
    #     random initial steps. It scales [action_space.low, action_space.high] to
    #     the value ranges of policy.forward.

    #     :param act: a data batch, list or numpy.ndarray which is the action taken
    #         by gym.spaces.Box.sample().

    #     :return: action remapped.
    #     """
    #     if isinstance(self.action_space, gym.spaces.Box):
    #         act = to_numpy(act)
    #         if isinstance(act, np.ndarray):
    #             if self.action_scaling:
    #                 low, high = self.action_space.low, self.action_space.high
    #                 scale = high - low
    #                 eps = np.finfo(np.float32).eps.item()
    #                 scale[scale < eps] += eps
    #                 act = (act - low) * 2.0 / scale - 1.0
    #             if self.action_bound_method == "tanh":
    #                 act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0  # type: ignore
    #     return act

    def pre_process(
        self,
        input_data: Union[Flow, List[Flow], Tuple[Flow, np.ndarray]],
        funs: List[Callable[[Union[Flow, List[Flow], Tuple[Flow, np.ndarray]]],
                            Flow]] | None = None,
    ) -> Flow:
        """
        前处理函数, 调用前处理方法, 对从Replaybuffer中采集的数据进行前处理.
        前处理方法包括:
        1.针对蒙特卡洛类算法一幕回报计算函数--compute_eposide_return();
        2.针对时序差分算法计算多部回报--compute_nstep_return();
        3....
        """
        if funs is None or funs == []:
            flow = input_data
        else:
            flow = input_data
            for fun in funs:
                flow = fun(flow)
        return flow

    def post_process(
        self,
        input_data: Union[Flow, List[Flow], Tuple[Flow, np.ndarray]],
        funs: List[Callable[[Union[Flow, List[Flow], Tuple[Flow, np.ndarray]]],
                            Flow]] | None = None,
    ) -> Flow:
        """
        后处理函数, 调用后处理方法, 对从Replaybuffer中采集的数据进行后处理.
        """
        if funs is None or funs == []:
            flow = input_data
        else:
            flow = input_data
            for fun in funs:
                flow = fun(flow)
        return flow

    @abstractmethod
    def learn(self, flow: Flow) -> Dict[str, Any]:
        """
        Update policy with a given flow of data.

        :return: A dict, including the data needed to be logged (e.g., loss).
        """
        pass

    def buffer_update(self,
                      sample_method: str,
                      pre_process_funs: List[Callable[
                          [Union[Flow, List[Flow], Tuple[Flow, np.ndarray]]],
                          Flow]] | None = None,
                      post_process_funs: List[Callable[
                          [Union[Flow, List[Flow], Tuple[Flow, np.ndarray]]],
                          Flow]] | None = None,
                      buffer: Optional[ReplayBuffer] = None,
                      sample_size: Optional[int] = None,
                      indices: Optional[Union[int, List[int], Tuple[int],
                                              np.ndarray, slice]] = None,
                      eposide_num: Optional[int] = None,
                      **kwargs: Any) -> Dict[str, Any]:
        """
        从Replaybuffer中数据接进行更新, 适用于在线的offpolicy更新和offline更新.
        param: sample_method(str):对buffer中数据取样方式, 可选:random_sample(随机取样), sample(按索引取样)和eposide_sample(按回合取样);
        param: flow_size(int), random_sample时设置取样大小, 默认为None;
        param: indices(int, List[int], Tuple[int], np.ndarray, slice), sample时设置取样索引, 默认为None;
        param: eposide_num(int), eposide_sample时设置取样回合数, 默认为None;
        """
        if buffer is None or len(buffer) == 0:
            return {}

        if sample_method == 'random_sample':
            assert sample_size is not None
            flow, indices = buffer.random_sample(sample_size)
            self._updating = True
            flow = self.pre_process(funs=pre_process_funs,
                                    input_data=(flow, indices))  # 前处理函数未确定
            result = self.learn(flow, **kwargs)
            self.post_process(funs=post_process_funs,
                              input_data=(flow, indices))

        elif sample_method == 'sample':
            assert indices is not None
            flow = buffer.sample(indices)
            self._updating = True
            flow = self.pre_process(funs=pre_process_funs, input_data=flow)
            result = self.learn(flow, **kwargs)
            self.post_process(funs=post_process_funs, input_data=flow)

        elif sample_method == 'eposide_sample':
            assert eposide_num is not None
            flow_list = buffer.eposide_sample(eposide_num)
            self._updating = True
            flow = self.pre_process(funs=pre_process_funs,
                                    input_data=flow_list)
            result = self.learn(flow, **kwargs)
            self.post_process(funs=post_process_funs, input_data=flow_list)

        self._updating = False
        return result

    def env_update(self,
                   env_output: Flow,
                   pre_process_funs: List[Callable[[Flow], Flow]]
                   | None = None,
                   post_process_funs: List[Callable[[Flow], Flow]]
                   | None = None,
                   **kwargs: Any) -> Dict[str, Any]:
        """
        从环境输出数据直接进行,适用于在线单步更新.
        """
        self._updating = True
        flow = self.pre_process(funs=pre_process_funs, input_data=env_output)
        result = self.learn(flow, **kwargs)
        self.post_process(funs=post_process_funs, input_data=env_output)
        self._updating = False
        return result

    @staticmethod
    def compute_episodic_return(
        input_data: List[Flow],
        gamma: float = 0.99,
    ) -> List[Flow]:
        """
        蒙特卡洛回报计算函数.
        """
        # 增量表达
        for data in input_data:
            data['returns'] = np.zeros_like(data['rew'], dtype=np.float64)
            for i in range(len(data['returns'])):
                if i == 0:
                    data['returns'][len(data['returns']) - i -
                                    1] = data['rew'][len(data['rew']) - i - 1]
                else:
                    data['returns'][len(data['returns'])-i-1] = data['rew'][len(data['rew'])-i-1] + \
                        gamma * data['returns'][len(data['returns'])-i]
        return input_data

    @staticmethod
    def return_normalization(input_data: List[Flow]) -> List[Flow]:
        for data in input_data:
            data['returns'] = (data['returns']-np.mean(data['returns'])) / \
                (np.var(data['returns'])+np.finfo(np.float32).eps)**0.5
        return input_data

    @staticmethod
    def compute_nstep_return(
        flow: Flow,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], tf.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Flow:
        """
        计算时序差分回报.
        """
        pass


#     def _compile(self) -> None:
#         f64 = np.array([0, 1], dtype=np.float64)
#         f32 = np.array([0, 1], dtype=np.float32)
#         b = np.array([False, True], dtype=np.bool_)
#         i64 = np.array([[0, 1]], dtype=np.int64)
#         _gae_return(f64, f64, f64, b, 0.1, 0.1)
#         _gae_return(f32, f32, f64, b, 0.1, 0.1)
#         _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)

# @njit
# def _gae_return(
#     v_s: np.ndarray,
#     v_s_: np.ndarray,
#     rew: np.ndarray,
#     end_flag: np.ndarray,
#     gamma: float,
#     gae_lambda: float,
# ) -> np.ndarray:
#     returns = np.zeros(rew.shape)
#     delta = rew + v_s_ * gamma - v_s
#     discount = (1.0 - end_flag) * (gamma * gae_lambda)
#     gae = 0.0
#     for i in range(len(rew) - 1, -1, -1):
#         gae = delta[i] + discount[i] * gae
#         returns[i] = gae
#     return returns

# @njit
# def _nstep_return(
#     rew: np.ndarray,
#     end_flag: np.ndarray,
#     target_q: np.ndarray,
#     indices: np.ndarray,
#     gamma: float,
#     n_step: int,
# ) -> np.ndarray:
#     gamma_buffer = np.ones(n_step + 1)
#     for i in range(1, n_step + 1):
#         gamma_buffer[i] = gamma_buffer[i - 1] * gamma
#     target_shape = target_q.shape
#     bsz = target_shape[0]
#     # change target_q to 2d array
#     target_q = target_q.reshape(bsz, -1)
#     returns = np.zeros(target_q.shape)
#     gammas = np.full(indices[0].shape, n_step)
#     for n in range(n_step - 1, -1, -1):
#         now = indices[n]
#         gammas[end_flag[now] > 0] = n + 1
#         returns[end_flag[now] > 0] = 0.0
#         returns = rew[now].reshape(bsz, 1) + gamma * returns
#     target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
#     return target_q.reshape(target_shape)

if __name__ == "__main__":
    a = Flow({'rew': np.array([1, 2, 3, 4, 5, 6])})
    b = BasePolicy.compute_episodic_return([a])
    print(b)
    c = BasePolicy.return_normalization(b)
    print(c)
