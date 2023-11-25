from typing import Optional, Union
from src.data.buffer import ReplayBuffer
from src.data.data_stru import Flow
from copy import deepcopy


class Collector:
    """从env采集s,a,s_,r数据,对数据进行处理后或直接将数据交给buffer储存,同时交给policy进行策略优化"""

    def __init__(
            self,
            buffer: Optional[
                ReplayBuffer] = None,  # ReplayBuffer为None时，不储存数据，在线训练
    ) -> None:
        self._buffer = buffer
        self._data = None
        self.save_obs_next = buffer._save_obs_next

    def collect(self, env_output: Union[Flow, dict]) -> None:
        if not self.save_obs_next:
            env_output.pop('obs_next')
        self._data = Flow(env_output)
        if self._buffer is not None:
            self._buffer.add(self._data)

    def output(self, data_type: str) -> Flow:
        """
        tf.Torch: touch
        np.ndarray: ndarray
        """
        if data_type == 'tensor':
            return deepcopy(self._data).to_torch()
        elif data_type == "ndarray":
            return deepcopy(self._data).to_numpy()
        else:
            raise AttributeError(
                'Please input right data_type: tensor/ndarray')
