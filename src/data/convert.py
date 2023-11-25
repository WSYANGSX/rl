from src.data.data_stru import Flow
import tensorflow as tf
import numpy as np
import h5py
from typing import Any, Dict, Optional, Union, no_type_check
from numbers import Number
from copy import deepcopy
import pickle

Hdf5ConvertibleValues = Union[int, float, Flow, np.ndarray, tf.Tensor, object,
                              "Hdf5ConvertibleType"]
Hdf5ConvertibleType = Union[Dict[str, Hdf5ConvertibleValues], Flow]


@no_type_check
def to_numpy(x: Any) -> Union[Flow, np.ndarray]:
    if isinstance(x, tf.Tensor):  # most often case
        return x.numpy()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    elif isinstance(x, (dict, Flow)):
        x = Flow(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, (list, tuple)):
        assert all(isinstance(item, (Number, np.number)) for item in x)
        return np.array(x)
    else:
        return np.asanyarray(x)


# @no_type_check
# def to_torch(
#     x: Any,
#     dtype: Optional[torch.dtype] = None,
#     device: Union[str, int, torch.device] = "cpu",
# ) -> Union[Batch, torch.Tensor]:
#     """Return an object without np.ndarray."""
#     if isinstance(x, np.ndarray) and issubclass(
#         x.dtype.type, (np.bool_, np.number)
#     ):  # most often case
#         x = torch.from_numpy(x).to(device)
#         if dtype is not None:
#             x = x.type(dtype)
#         return x
#     elif isinstance(x, torch.Tensor):  # second often case
#         if dtype is not None:
#             x = x.type(dtype)
#         return x.to(device)
#     elif isinstance(x, (np.number, np.bool_, Number)):
#         return to_torch(np.asanyarray(x), dtype, device)
#     elif isinstance(x, (dict, Batch)):
#         x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
#         x.to_torch(dtype, device)
#         return x
#     elif isinstance(x, (list, tuple)):
#         return to_torch(_parse_value(x), dtype, device)
#     else:  # fallback
#         raise TypeError(f"object {x} cannot be converted to torch.")


def to_hdf5(x: Hdf5ConvertibleType,
            y: h5py.Group,
            compression: Optional[str] = None) -> None:

    def to_hdf5_via_pickle(x: object,
                           y: h5py.Group,
                           key: str,
                           compression: Optional[str] = None) -> None:
        data = np.frombuffer(pickle.dumps(x), dtype=np.byte)
        y.create_dataset(key, data=data, compression=compression)

    for k, v in x.items():
        if isinstance(v, (dict, Flow)):
            subgrp = y.create_group(k)
            if isinstance(v, Flow):
                subgrp.attrs["__data_type__"] = "Flow"
            else:
                subgrp.attrs["__data_type__"] = "Dict"
            to_hdf5(v, subgrp, compression=compression)
        elif isinstance(v, tf.Tensor):
            y.create_dataset(k, data=v, compression=compression)
            y[k].attrs["__data_type__"] = "Tensor"
        elif isinstance(v, np.ndarray):
            try:
                y.create_dataset(k, data=v, compression=compression)
                y[k].attrs["__data_type__"] = "ndarray"
            except TypeError:
                # If data type is not supported by HDF5 fall back to pickle.
                # This happens if dtype=object (e.g. due to entries being None)
                # and possibly in other cases like structured arrays.
                try:
                    to_hdf5_via_pickle(v, y, k, compression=compression)
                except Exception as exception:
                    raise RuntimeError(
                        f"Attempted to pickle {v.__class__.__name__} due to "
                        "data type not supported by HDF5 and failed."
                    ) from exception
                y[k].attrs["__data_type__"] = "pickled_ndarray"
        elif isinstance(v, (int, float)):
            y.attrs[k] = v
        else:
            try:
                to_hdf5_via_pickle(v, y, k, compression=compression)
            except Exception as exception:
                raise NotImplementedError(
                    f"No conversion to HDF5 for object of type '{type(v)}' "
                    "implemented and fallback to pickle failed."
                ) from exception
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(x: h5py.Group) -> Hdf5ConvertibleValues:
    if isinstance(x, h5py.Dataset):
        if x.attrs["__data_type__"] == "ndarray":
            return np.array(x)
        elif x.attrs["__data_type__"] == "Tensor":
            return tf.convert_to_tensor(x)
        else:
            return pickle.loads(x[...])
    else:
        y = dict(x.attrs.items())
        data_type = y.pop("__data_type__", None)
        for k, v in x.items():
            y[k] = from_hdf5(v)
        return Flow(y) if data_type == "Flow" else y
