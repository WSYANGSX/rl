import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import Any
import time
from src.data.data_stru import Flow


class AntEnv:
    """
    参数 xml_file <str>: Path to a MuJoCo model;
    参数 ctrl_cost_weight(可选) <float>: Weight for ctrl_cost term (see section on reward);
    参数 use_contact_forces(可选) <bool>: If true, it extends the observation space by adding contact forces 
                                        (see Observation Space section) and includes contact_cost to the 
                                        reward function (see Rewards section);
    参数 contact_cost_weight(可选) <float>: Weight for contact_cost term (see section on reward);
    参数 healthy_reward(可选) <float>: Constant reward given if the ant is “healthy” after timestep;
    参数 terminate_when_unhealthy(可选) <bool>: If true, issue a done signal if the z-coordinate of 
                                               the torso is no longer in the healthy_z_range;
    参数 healthy_z_range(可选) <tuple>: The ant is considered healthy if the z-coordinate of the 
                                       torso is in this range;
    参数 contact_force_range(可选) <tuple>: Contact forces are clipped to this range in the
                                           computation of contact_cost;
    参数 reset_noise_scale(可选) <float>: Scale of random perturbations of initial position 
                                         and velocity (see section on Starting State);
    参数 exclude_current_positions_from_observation(可选) <bool>: Whether or not to omit the x- and y-coordinates from observations.
                                                                 Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies.
    参数 render_mode(可选) <str>: "human"、"rgb_array"、"ansi"、"rgb_array_list"
    """

    id_list = []

    def __init__(self,
                 xml_file: str = 'Ant-v4',
                 id: int | None = None,
                 seed: int = 23,
                 **kwargs) -> None:
        if id is not None:
            self.set_id(id)
        self._env = gym.make(xml_file, **kwargs)
        self.init_state = self.reset(seed)
        self._current_obs = self.init_state[0]

    def set_id(self, id: int):
        """
        用于多环境训练同步训练
        """
        assert id not in AntEnv.id_list, f'Evn_id has already existed!'
        AntEnv.id_list.append(id)
        self._id = id

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
    agent = AntEnv(xml_file='Ant-v4', render_mode="human")
    for _ in range(100):
        action = agent.action_space.sample()
        ans = agent.step(action)
        # print(ans.obs.ndim)

        if ans["terminated"] or ans["truncated"]:
            observation, info = agent.reset(int(time.time()))

    agent.close()
#     (
#     obs: array([ 7.10598257e-01,  9.88425608e-01,  9.49493118e-02, -7.18271244e-02,
#                 9.40229192e-02, -3.01284837e-01,  1.17465416e+00, -2.43319889e-01,
#                -5.44194264e-01,  1.53789400e-01, -8.12021096e-01, -4.93182951e-01,
#                 7.48650571e-01,  2.73499166e-01,  2.22215000e-01, -5.56766147e-01,
#                -1.42775828e+00, -6.42210030e-01,  1.55925900e+00, -4.70906437e+00,
#                 1.51892111e+01, -1.44087514e+00, -6.93402975e-03,  8.42657896e-01,
#                -9.61032399e+00, -9.79113931e+00,  9.06396576e+00]),
#     act: array([-0.99874455,  0.31726632, -0.05791925,  0.80720705, -0.9792974 ,
#                 0.9741304 , -0.6863861 , -0.56782943], dtype=float32),
#     obs_next: array([ 6.67224748e-01,  9.90999320e-01, -1.71166094e-02,  2.17635356e-04,
#                      1.32767924e-01, -5.41120039e-01,  1.28240968e+00, -4.93111721e-01,
#                     -5.03546069e-01,  7.09985279e-02, -1.31040733e+00, -5.64139603e-01,
#                      1.24966429e+00, -1.86329525e-01, -2.91118313e-01, -1.16982737e+00,
#                     -3.72116435e+00,  4.14153730e+00,  2.27445757e+00, -4.00221843e+00,
#                     -1.66704068e+00, -8.47137520e+00, -3.22177889e-01, -4.14435683e+00,
#                     -2.65540315e+00,  1.07355433e+00,  8.94065980e+00]),
#     rew: -1.3266250790742147,
#     terminated: False,self.
#     truncated: False,
#     info: {'distance_from_origin': 0.08161655166037535,
#           'forward_reward': -0.09932705775768631,
#           'reward_ctrl': -2.2272980213165283,
#           'reward_forward': -0.09932705775768631,
#           'reward_survive': 1.0,
#           'x_position': 0.0017602160924797099,
#           'x_velocity': -0.09932705775768631,
#           'y_position': 0.08159756824954095,
#           'y_velocity': -0.20295740870003937},
# )
