import pandas
import pprint
from numbers import Number
import tensorflow as tf
from copy import deepcopy
import numpy as np
from typing import Any, Tuple, List, Union, Dict, Iterable, Generator, Sequence

IndexType = Union[slice, int, np.ndarray, List[int]]


def assert_type_keys(keys: Iterable[str]) -> None:
    assert all(isinstance(key, str) for key in keys), \
        f"keys should all be string, but got {keys}"


class Flow:
    """
    内部自定义数据结构,用于环境、收集器、buffer、网络之间的数据交互和流通;
    """

    def __init__(
        self,
        input_data: Union[Dict, "Flow"] | None = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> None:
        if copy:
            # deepcopy(None)不起作用，还是None
            input_data = deepcopy(input_data)
        if input_data is not None:
            if isinstance(input_data, (Dict, Flow)):
                assert_type_keys(input_data.keys())
                for key, obj in input_data.items():
                    self.__dict__[key] = obj
        if len(kwargs) > 0:
            self.__init__(kwargs, copy=copy)

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value  # 避免使用self.key=value,会发生递归

    def __getattr__(self, key: str) -> Any:
        assert isinstance(key, str) and key in self.__dict__.keys(
        ), "There has no attributes of {}".format(key)
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def __format__(self, __format_spec: str) -> str:
        return str(self)

    def __repr__(self) -> str:
        self_str = "Flow"
        self_str += str(self)
        return self_str

    def keys(self) -> List:
        return self.__dict__.keys()

    def values(self) -> List:
        return self.__dict__.values()

    def __str__(self) -> str:  # 递归调用，输出多级缩进
        self_str = "(\n"
        flag = False
        for key, obj in self.__dict__.items():
            rpl = "\n" + " " * (5 + len(key))
            obj_name = pprint.pformat(obj).replace("\n", rpl)
            self_str += f"    {key}: {obj_name},\n"
            flag = True
        if flag:
            self_str += ")"
        else:
            self_str = "()"
        return self_str

    def __len__(self) -> int:
        return len(self.__dict__)

    def items(self) -> Generator:
        return ((key, value) for key, value in self.__dict__.items())

    def __getitem__(self, index: Union[str, List[str], Tuple[str]]) -> Any:
        if isinstance(index, str):
            assert index in self.__dict__.keys(
            ), 'There is no {} item.'.format(index)
            return self.__dict__[index]
        elif isinstance(index, (list, tuple)):
            assert_type_keys(index)
            new_flow = Flow()
            for key in index:
                new_flow[key] = self.__dict__[key]
            return new_flow
        else:
            raise TypeError("Wrong index type!")

    def __setitem__(self, index: str, value: Any) -> None:
        assert_type_keys(index)
        self.__dict__[index] = value

    def __getstate__(self) -> Dict[str, Any]:
        state = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Flow):
                value = value.__getstate__()
            state[key] = value
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__init__(**state)

    def __deepcopy__(self, memo=None, _nil=[]) -> "Flow":
        if memo is None:
            memo = {}
        d = id(self)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        new_flow = Flow()
        memo[d] = id(new_flow)
        for key in self.__dict__.keys():
            new_flow.__setattr__(deepcopy(key, memo),
                                 deepcopy(self.__dict__[key], memo))
        return new_flow

    def add(self, obj: Union[dict[str, Any], 'Flow']) -> None:  # 遵循底层append
        assert isinstance(obj, (dict, Flow)), 'Wrong obj type!'
        for key, value in obj.items():
            if key in self.keys():
                assert type(value) == type(
                    self.__dict__[key]), "Different type can not add!"
                self.__dict__[key] += value
            else:
                self.__dict__[key] = value

    def del_item(self, index: str) -> None:
        del self.__dict__[index]

    def pop(self, index: str) -> Any:
        re = self[index]
        self.del_item(index)
        return re

    def reset(self) -> None:
        for key in list(self.__dict__.keys()):
            self.del_item(key)

    # +=这个符号并不是一个原地操作,a+=1相当于a=a.__iadd__(1),是将一个返回值赋给a
    def __iadd__(self, other: Union[dict[str, Any], 'Flow']) -> 'Flow':
        assert isinstance(other,
                          Flow), 'Only Flow type can be added to Flow type!'
        for key, value in other.__dict__.items():  # 兼容底层数据的+=运算规则
            if key in self.__dict__.keys():
                self.__dict__[key] += value
            else:
                self.__dict__[key] = value
        return self

    def __add__(self, other: 'Flow') -> 'Flow':
        assert isinstance(other,
                          Flow), 'Only Flow type can be added to Flow type!'
        return deepcopy(self).__iadd__(other)
        # 其他方法
        # new_flow = Flow()
        # for key, value in other.__dict__.items():
        #     if key in new_flow.__dict__.keys():
        #         new_flow.__dict__[key] += value
        #     else:
        #         new_flow.__dict__[key] = value
        # for key, value in self.__dict__.items():
        #     if key in new_flow.__dict__.keys():
        #         new_flow.__dict__[key] += value
        #     else:
        #         new_flow.__dict__[key] = value
        # return new_flow

    def __imul__(self, value: Union[Number, np.number]) -> "Flow":
        assert isinstance(
            value,
            (Number, np.number)), 'Only number instance can be multipled!'
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], (dict, set)):
                pass
            else:
                self.__dict__[key] *= value
        return self

    def __mul__(self, value: Union[Number, np.number]) -> "Flow":
        assert isinstance(value, (Number, np.number))
        return deepcopy(self).__imul__(value)

    def __itruediv__(self, value: Union[Number, np.number]) -> "Flow":
        assert isinstance(value, (Number, np.number))
        for key in self.__dict__.keys():
            self.__dict__[key] /= value
        return self

    def __truediv__(self, value: Union[Number, np.number]) -> "Flow":
        assert isinstance(value, (Number, np.number))
        return deepcopy(self).__itruediv__(value)

    def to_numpy(self) -> None:
        for key, obj in self.__dict__.items():
            if isinstance(obj, tf.Tensor):  # most often case
                self.__dict__[key] = obj.numpy()
            elif isinstance(obj, (np.number, Number, np.bool_)):
                self.__dict__[key] = np.asanyarray(obj, dtype=np.float64)
            elif obj is None:
                self.__dict__[key] = np.array(None, dtype=object)
            elif isinstance(obj, (list, tuple)):
                assert all(
                    isinstance(item, (Number, np.number)) for item in obj)
                self.__dict__[key] = np.array(obj, dtype=np.float64)
            elif isinstance(obj, Flow):
                obj.to_numpy()
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = np.array(value)
            else:
                pass
        return self

    def to_torch(self) -> None:
        for key, obj in self.__dict__.items():
            if isinstance(
                    obj,
                (np.number, Number, np.bool_, np.ndarray)):  # most often case
                self.__dict__[key] = tf.convert_to_tensor(obj,
                                                          dtype=tf.float64)
            elif obj is None:
                self.__dict__[key] = tf.convert_to_tensor(None)
            elif isinstance(obj, (list, tuple)):
                assert all(
                    isinstance(item, (Number, np.number)) for item in obj)
                self.__dict__[key] = tf.convert_to_tensor(obj,
                                                          dtype=tf.float64)
            elif isinstance(obj, Flow):
                obj.to_torch()
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = tf.convert_to_tensor(value)
            else:
                pass
        return self

    def istack(
        self, input_data: Union['Flow', dict,
                                Sequence[Union[dict, "Flow"]]]) -> "Flow":
        assert isinstance(input_data, (dict, Flow, Sequence))
        if self.empty():
            if isinstance(input_data, (dict, Flow)):
                self.__init__(input_data)
            else:
                for i in range(len(input_data)):
                    if i == 0:
                        self.__init__(input_data[i])
                    else:
                        self.istack(input_data[i])

        else:
            if isinstance(input_data, dict):
                if len(input_data) == 0:
                    pass
                else:
                    new_flow = Flow(input_data)
                    self.istack(new_flow)

            elif isinstance(input_data, Flow):
                if len(input_data) == 0:
                    pass
                else:
                    assert sorted(self.keys()) == sorted(input_data.keys())
                    for key, value in input_data.items():  # for in 语句中的变量是副本

                        if isinstance(value, np.ndarray):
                            assert isinstance(self[key], np.ndarray)
                            if self[key].ndim == value.ndim:
                                assert self[key].shape == value.shape
                                shape = value.shape
                                self[key] = np.append(self[key],
                                                      input_data[key])
                                self[key] = self[key].reshape(-1, *shape)
                            else:
                                assert self[key].ndim == value.ndim + \
                                    1 and self[key][0].shape == value.shape
                                shape = value.shape
                                self[key] = np.append(self[key],
                                                      input_data[key])
                                self[key] = self[key].reshape(-1, *shape)

                        elif isinstance(value, tf.Tensor):
                            assert isinstance(self[key], tf.Tensor)
                            if self[key].ndim == value.ndim:
                                assert self[key].shape == value.shape
                                shape = value.shape
                                self[key] = self[key].numpy()
                                value = value.numpy()
                                self[key] = np.append(self[key], value)
                                self[key] = self[key].reshape(-1, *shape)
                                self[key] = tf.convert_to_tensor(self[key])
                            else:
                                assert self[key].ndim == value.ndim + \
                                    1 and self[key][0].shape == value.shape
                                shape = value.shape
                                self[key] = self[key].numpy()
                                value = value.numpy()
                                self[key] = np.append(self[key], value)
                                self[key] = self[key].reshape(-1, *shape)
                                self[key] = tf.convert_to_tensor(self[key])

                        elif isinstance(value, (Number, np.number)):
                            if isinstance(self[key], (Number, np.number)):
                                self[key] = np.array(self[key],
                                                     dtype=np.float64)
                                value = np.array(value, dtype=np.float64)
                                self[key] = np.append(self[key], value)
                            else:
                                assert isinstance(
                                    self[key], np.ndarray) and all(
                                        isinstance(item, (Number, np.number))
                                        for item in self[key])
                                value = np.array(value, dtype=np.float64)
                                self[key] = np.append(self[key], value)

                        # elif isinstance(value, (Flow, dict)):
                        #     if isinstance(self[key], Flow):
                        #         self[key]=self[key].istack(value)
                        #     else:
                        #         if type(self[key]) is not list:
                        #             temp = self[key]
                        #             self[key] = list()
                        #             self[key].append(temp)
                        #         self[key].append(value)

                        else:
                            if type(self[key]) is not list:
                                temp = self[key]
                                self[key] = list()
                                self[key].append(temp)
                            self[key].append(value)

            else:
                for item in input_data:
                    self.istack(item)
        return self

    def stack(
        self, input_data: Union['Flow', dict,
                                Sequence[Union[dict, "Flow"]]]) -> "Flow":
        return deepcopy(self).istack(input_data)

    def empty(self) -> bool:
        if self.__len__() > 0:
            return False
        else:
            return True


# test
if __name__ == "__main__":
    a = Flow({
        'action':
        tf.constant([5, 2, 3, 4, 5, 6, 7]),
        'state':
        np.array([[5, 2, 3, 4, 5, 6, 7], [5, 2, 3, 4, 5, 6, 7]]),
        'next':
        Flow({
            'state': np.array([[1, 1, 1, 1, 1, 1, 1], [5, 2, 3, 4, 5, 6, 7]]),
            'info': tf.constant([5, 2, 3, 4, 5, 6, 7])
        })
    })
    print(a['action'])
