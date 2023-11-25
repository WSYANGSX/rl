import warnings
from src.data.convert import to_hdf5, from_hdf5
from copy import deepcopy
import tensorflow as tf
from src.data.data_stru import Flow
import numpy as np
import h5py
from typing import Any, Dict, List, Tuple, Union, Sequence
import operator


class ReplayBuffer:
    """
    用于存储环境的状态、奖励等信息，可以与收集器和策略网路进行直接交互。

    ReplayBuffer 可以被视为对Flow的组织管理. 

    :param int size: the maximum size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater than or
        equal to 1. Default to 1 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next. Default to False.
    :param bool save_only_last_obs: only save the last obs/obs_next when it has a shape
        of (timestep, ...) because of temporal stacking. Default to False.
    :param bool sample_avail: the parameter indicating sampling only available index
        when using frame-stack sampling method. Default to False.
    """

    _reserved_keys = ('obs', 'act', 'rew', 'obs_next', 'terminated',
                      'truncated', 'done', 'info', 'policy')
    _input_keys = ('obs', 'act', 'rew', 'obs_next', 'terminated', 'truncated',
                   'info', 'policy')

    def __init__(self,
                 capacity: int,
                 stack_num: int = 1,
                 save_obs_next: bool = True,
                 save_only_last_obs: bool = False,
                 sample_avail: bool = False,
                 **kwargs: Any) -> None:
        self._capacity = int(capacity)
        self._size = 0
        assert stack_num > 0, "stack_num should be greater than 0"
        self._stack_num = stack_num
        self._save_obs_next = save_obs_next
        self._save_only_last_obs = save_only_last_obs
        self._sample_avail = sample_avail
        self._meta = None
        self.initialize()
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                self.__dict__[key] = value

    def initialize(self) -> None:
        self._meta = Flow()

    def reset(self) -> None:
        self._meta.reset()
        self._size = 0

    def __len__(self):
        return self._size

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(self._meta)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._meta[key]
        except KeyError as exception:
            raise AttributeError from exception

    def __setattr__(self, key: str, value: Any) -> None:
        assert (key not in self._reserved_keys
                ), "key '{}' is reserved and cannot be assigned".format(key)
        super().__setattr__(key, value)

    def __getstate__(self) -> Dict[str, Any]:
        state = {}
        for key, value in self.__dict__.items():
            state[key] = value
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def add(self, obj: Union[Flow, dict, Sequence[Union[dict, Flow]]]) -> None:
        if self._size < self._capacity:
            if isinstance(obj, (Flow, dict)):
                if self._meta.empty():
                    self._meta = Flow(obj)
                    self._size += 1
                else:
                    assert sorted(self._meta.keys()) == sorted(
                        obj.keys()), "Added flow has different attributes!"
                    self._meta.istack(obj)
                    self._size += 1
            else:
                for item in obj:
                    self.add(item)
        else:
            print(
                'The replaybuffer element is full, del element from the top!')

    def __getitem__(
            self, index: Union[int, List[int], Tuple[int], np.ndarray,
                               slice]) -> Flow:
        new_flow = Flow()
        avaiable_indices = list(range(len(self)))
        if operator.eq(avaiable_indices, [0]) and index == 0:
            new_flow = self._meta
        elif isinstance(index, (int, np.number)):
            assert index in avaiable_indices, f'index is out of avaiable_indices!'
            for key in self._meta.keys():
                new_flow[key] = self._meta[key][index]
        elif isinstance(index, (list, tuple, np.ndarray)):
            indices = np.array(index).flatten()
            assert set(indices).issubset(
                avaiable_indices), f'index is out of avaiable_indices!'
            for i in indices:
                new_flow.istack(self[i])
        elif isinstance(index, slice):
            if index == slice(None):
                return deepcopy(self._meta)
            else:
                indices = range(index.stop)[index]
                assert set(indices).issubset(
                    avaiable_indices), f'index is out of avaiable_indices!'
                for i in indices:
                    new_flow.istack(self[i])
        return new_flow

    def items(self):
        return self.__dict__.items()

    def _sample_indices(self, flow_size: int) -> np.ndarray:
        assert flow_size >= 0 and flow_size <= self._size, f'0 =< batch_size <= self._size!'
        if flow_size == 0:
            return np.arange(self._size)
        else:
            return np.random.choice(self._size, flow_size, replace=False)

    def random_sample(self, flow_size: int) -> tuple[Flow, np.ndarray]:
        indices = self._sample_indices(flow_size)
        return self.__getitem__(indices), indices

    def sample(
            self, indices: Union[int, List[int], Tuple[int], np.ndarray,
                                 slice]) -> Flow:
        return self.__getitem__(indices)

    # 以列表形式返回多幕序列,将幕进行提取，便于计算
    def eposide_sample(self, eposide_num: int = 1) -> list[Flow]:
        flow_list = []
        terminated = np.where(self._meta['terminated'] == 1)
        truncated = np.where(self._meta['truncated'] == 1)
        done = np.sort(np.unique(np.append(terminated, truncated)))
        if len(done) < eposide_num == 0:
            warnings.warn(
                'The number of eposides in buffer is less than the number that needs to be extracted, all eposides in buffer are returned!'
            )
            extract_end_pos = done
        elif eposide_num == 0:
            extract_end_pos = done
        else:
            extract_end_pos = np.random.choice(done,
                                               eposide_num,
                                               replace=False)
        for end_pos in list(extract_end_pos):
            index = np.where(done == end_pos)[0]
            start_pos = done[index - 1] + 1 if index > 0 else 0
            flow = self.sample(np.arange(start_pos, end_pos + 1))
            flow_list.append(flow)
        return flow_list

    def pop(
        self,
        index: Union[int, List[int], Tuple[int], np.ndarray, slice] | None = 0
    ) -> Flow:
        new_flow = self[index]
        if isinstance(index, int):
            for key, value in self._meta.items():
                if isinstance(value, np.ndarray):
                    self._meta[key] = np.delete(value, index, axis=0)
                elif isinstance(value, tf.Tensor):
                    self._meta[key] = np.delete(value, index, axis=0)
                    self._meta[key] = tf.convert_to_tensor(self._meta[key])
                else:
                    self._meta[key].pop(index)
            self._size -= 1
        elif isinstance(index, (list, tuple, np.ndarray)):
            for key, value in self._meta.items():
                if isinstance(value, np.ndarray):
                    self._meta[key] = np.delete(value, index, axis=0)
                elif isinstance(value, tf.Tensor):
                    self._meta[key] = np.delete(value, index, axis=0)
                    self._meta[key] = tf.convert_to_tensor(self._meta[key])
                else:
                    self._meta[key] = [
                        i for num, i in enumerate(self._meta[key])
                        if num not in index
                    ]
            self._size -= len(index)
        else:
            indices = range(len(self))[index]
            self.pop(indices)
        return new_flow

    def update(self, another: "ReplayBuffer") -> None:
        if self._capacity == 0:
            pass
        elif another._size <= self._capacity:
            self._meta = deepcopy(another._meta)
            self._size = another._size
        elif another._size > self._capacity:
            tp_replaybuf = deepcopy(another)
            del_indices = list(range(another._size - self._capacity))
            print(del_indices)
            tp_replaybuf.pop(del_indices)
            self._meta = tp_replaybuf._meta
            self._size = self._capacity

    def __deepcopy__(self, memo=None, _nil=[]) -> "Flow":
        if memo is None:
            memo = {}
        d = id(self)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        new_replaybuffer = ReplayBuffer(self._capacity)
        memo[d] = id(new_replaybuffer)
        for key in self.__dict__.keys():
            new_replaybuffer.__setattr__(deepcopy(key, memo),
                                         deepcopy(self.__dict__[key], memo))
        return new_replaybuffer

    def save_as_hdf5(self, path: str, compression: str | None = None) -> None:
        with h5py.File(path, 'w') as f:
            to_hdf5(self, f, compression=compression)

    @classmethod
    def load_hdf5(cls, path: str) -> "ReplayBuffer":
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f))
        return buf

    @classmethod
    def from_h5py_dataset(cls, **kwargs) -> "ReplayBuffer":
        "dataset 中必须为ndarray或tensor"
        assert len(kwargs) > 0, 'There has no dataset input!'
        input_dict = kwargs
        lens = [len(value) for value in input_dict.values()]
        print(lens)
        if len(lens) > 1:
            for i in range(1, len(lens)):
                assert lens[0] == lens[i]
        size = lens[0]
        buf = cls(size)
        if size == 0:
            return buf
        for key, val in input_dict.items():
            if val.attrs['__data_type__'] == 'ndarray':
                input_dict[key] = np.array(val)
            elif val.attrs['__data_type__'] == 'Tensor':
                input_dict[key] = tf.convert_to_tensor(val)
        flow = Flow(kwargs)
        buf._meta = flow
        buf._size = size
        return buf


if __name__ == "__main__":
    buf = ReplayBuffer(20)
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 1,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 1,
        'truncated': 0,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 0,
        'truncated': 1,
    })
    buf.add({
        'act': np.array([0, 1, 2, 3, 5]),
        'obs': np.array([4, 5, 6, 7, 8, 9]),
        'terminated': 1,
        'truncated': 1,
    })

    a = buf.eposide_sample(0)
    print(a)
