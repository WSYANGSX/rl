import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import Any
import time
from src.data.data_stru import Flow
import tensorflow as tf


class MountainCar:

    id_list = []

    def __init__(self, xml_file: str, id: int, **kwargs) -> None:
        assert id not in MountainCar.id_list, f'Evn_id has already existed!'
        self._env_id = id
        MountainCar.id_list.append(self._env_id)
        self._env = gym.make(xml_file, **kwargs)
        self.init_state = self.reset(int(time.time()))
        self._current_obs = self.init_state[0]

    def set_id(self, env_id: int):
        """
        用于多环境训练同步训练
        """
        self._env_id = env_id

    def step(self, action: ActType) -> Flow:
        observation, reward, terminated, truncated, info = self._env.step(
            action)
        info_dict = {
            'obs': self._current_obs,
            'act': action,
            'obs_next': observation,
            'rew': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info
        }
        self._current_obs = observation
        return Flow(info_dict)

    def reset(self, seed) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        self._current_obs = obs
        return obs, info

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    def close(self) -> None:
        self._env.close()


# test
if __name__ == "__main__":
    ant1 = MountainCar(xml_file='MountainCarContinuous-v0',
                       id=1,
                       render_mode="human")
    print(ant1._current_obs)
    # # print(ant1.init_obs_info)
    # for _ in range(100):
    #     action = ant1.action_space.sample()
    #     ans = ant1.step(action)
    #     print(ans)

    #     if ans["terminated"] or ans["truncated"]:
    #         observation, info = ant1.reset(int(time.time()))

    # ant1.close()
