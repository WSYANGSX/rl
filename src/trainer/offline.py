from src.data.collector import Collector
from typing import Any
from src.policy import BasePolicy
from src.data.buffer import ReplayBuffer
from src.trainer.base import Trainer


class Offline(Trainer):
    """
    离线训练方法.
    """

    def __init__(
        self,
        env: Any,
        buffer: ReplayBuffer,
        collector: Collector,
        act_policy: BasePolicy,
        tg_policy: BasePolicy,
        eposide: int = 10000,
        rollout_times: int = 1,
    ) -> None:
        super().__init__(eposide=eposide, env=env, collector=collector)
        self._buffer = buffer
        self._rollout_times = rollout_times
        self._act_policy = act_policy
        self._tg_policy = tg_policy

    def train(self) -> None:
        # 采用初始策略对环境进行采样
        for _ in range(self._rollout_times):
            action = self._act_policy.forward(self._evn._current_obs)
            env_output = self._evn.step(action)
            self._collector.collect(env_output)
            if env_output["terminated"] or env_output["truncated"]:
                break

        self._act_policy.update()
