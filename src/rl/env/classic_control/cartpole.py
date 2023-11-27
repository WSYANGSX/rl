from typing import Any
import gymnasium as gym
from rl.data.data_stru import Flow
from gymnasium.core import ActType, ObsType


class CartPole:
    id_list = []

    def __init__(
        self,
        xml_file: str = "CartPole-v0",
        seed: int = 23,
        id: int | None = None,
        **kwargs,
    ) -> None:
        if id is not None:
            self.set_id(id)
        self._env = gym.make(xml_file, **kwargs)
        self.init_state = self.reset(seed=seed)
        self._current_obs = self.init_state[0]

    def set_id(self, id: int):
        """
        用于多环境训练同步训练
        """
        assert id not in CartPole.id_list, f"Evn_id has already existed!"
        CartPole.id_list.append(id)
        self._id = id

    def step(self, action: ActType) -> Flow:
        observation, reward, terminated, truncated, info = self._env.step(action)
        info_dict = {
            "obs": self._current_obs,
            "act": action,
            "obs_next": observation,
            "rew": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        self._current_obs = observation
        return Flow(info_dict)

    def reset(self, seed) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self._env.reset()
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
    agent = CartPole(xml_file="CartPole-v0", render_mode="human")
    print(agent.action_space)
    # for _ in range(100):
    #     action = agent.action_space.sample()
    #     ans = agent.step(action)
    #     # print(ans.obs.ndim)

    #     if ans["terminated"] or ans["truncated"]:
    #         observation, info = agent.reset(int(time.time()))

    agent.close()
