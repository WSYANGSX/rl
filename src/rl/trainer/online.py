import time
import numpy as np
from rl.data.collector import Collector
from rl.policy import BasePolicy
from rl.data.buffer import ReplayBuffer
from rl.data.data_stru import Flow
from rl.trainer.base import Trainer
from typing import Any, Callable, Union, List, Tuple


class Online(Trainer):
    """
    在线训练方法,分为onpolicy和offpolicy.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        collector: Collector,
        policy: BasePolicy,
        eposide: int = 10000,
        rollout_times: int = 1,
    ) -> None:
        super().__init__(eposide=eposide, collector=collector)
        self._buffer = buffer
        self._rollout_times = rollout_times
        self._policy = policy
        self._method = self._policy._update_method
        self._env = self._policy._env

    def train(
        self,
        method: str,
        pre_process_funs: Callable[
            [Union[Flow, List[Flow], Tuple[Flow, np.ndarray]]], Flow
        ]
        | None = None,
        post_process_funs: Callable[
            [Union[Flow, List[Flow], Tuple[Flow, np.ndarray]]], Flow
        ]
        | None = None,
    ) -> None:
        if method == "onpolicy":
            for epo in range(self._eposide):
                while 1:
                    action = self._policy.forward(self._env._current_obs)
                    env_output = self._evn.step(action)
                    self._collector.collect(env_output)
                    self._policy.env_update(
                        env_output=env_output,
                        pre_process_funs=pre_process_funs,
                        post_process_funs=post_process_funs,
                    )

                    if env_output["terminated"] or env_output["truncated"]:
                        break

        if method == "offpolicy":
            for epo in range(self._eposide):
                while 1:
                    action = self._policy.forward(self._env._current_obs)
                    env_output = self._env.step(action)
                    self._collector.collect(env_output)
                    if env_output["terminated"] or env_output["truncated"]:
                        self._env.reset(seed=int(time.time()))
                        break

                self._policy.buffer_update(
                    sample_method="eposide_sample",
                    pre_process_funs=pre_process_funs,
                    post_process_funs=post_process_funs,
                    buffer=self._buffer,
                    eposide_num=1,
                )
                self._buffer.reset()
