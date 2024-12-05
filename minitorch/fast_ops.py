from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT a function."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out
        # slow version:
        # return TensorOps.matrix_multiply(a, b)


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # Check if out and in are stride-aligned to avoid indexing
        stride_aligned: bool = False
        if len(out_strides) == len(in_strides) and np.all((out_strides == in_strides)):
            # Check shape is same too
            if np.all(out_shape == in_shape):
                stride_aligned = True
        if stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # Convert for loop into parallel
            for ordinal in prange(len(out)):
                # Initialize the out_index and in_index arrays for each thread
                out_index: Index = np.zeros(MAX_DIMS, np.int32)
                in_index: Index = np.zeros(MAX_DIMS, np.int32)
                # Convert ordinal (linear position) to the multidimensional `out_index`
                to_index(ordinal, out_shape, out_index)
                # Broadcast `out_index` to `in_index`
                broadcast_index(out_index, out_shape, in_shape, in_index)
                # Convert `out_index` and `in_index` to positions in the respective storage arrays
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                # Apply the function and store the result in the `out` array
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # Check if out, a, b are stride-aligned to avoid indexing
        if (len(out_strides) == len(a_strides) 
            and len(out_strides) == len(b_strides)
            and np.all((out_strides == a_strides))
            and np.all((out_strides == b_strides))
            and np.all((out_shape == a_shape))
            and np.all((out_shape == b_shape))
        ):
            # Directly apply function to the whole array
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])

        else:
            # Iterate over all positions in the `out` tensor in parallel
            for i in prange(len(out)):
                # Initialize the out_index and in_index arrays locally
                out_index: Index = np.zeros(MAX_DIMS, np.int32)
                a_index: Index = np.zeros(MAX_DIMS, np.int32)
                b_index: Index = np.zeros(MAX_DIMS, np.int32)
                # Convert ordinal (linear position) to the multidimensional `out_index`
                to_index(i, out_shape, out_index)

                # Broadcast `out_index` to `a/b_index`
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                # Convert indices to positions in the respective storage arrays
                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)

                # Apply the function and store the result in the `out` array
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        reduce_size = a_shape[reduce_dim]

        # Iterate over all positions in the `out` tensor in parallel
        for ordinal in prange(len(out)):
            # Initialize the out_index locally in thread
            out_index: Index = np.zeros(MAX_DIMS, np.int32)
            # Convert ordinal (linear position) to the multidimensional `out_index`
            to_index(ordinal, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            # Initialize the output value as a local variable for reduction
            temp_output = out[out_pos]
            # Calculate initial index for when out_index[reduce_dim] = 0
            out_index[reduce_dim] = 0
            j = index_to_position(out_index, a_strides)
            reduce_dim_stride = a_strides[reduce_dim]
            # Iterate over the `reduce_dim` and apply the reduction function `fn`
            for _ in range(reduce_size):
                # replace function call with direct calculation
                temp_output = fn(
                    temp_output, a_storage[j]
                )  # Start with j first since we calculated it already
                j += reduce_dim_stride  # Increment j by stride of reduce_dim

            # Store the reduced value in the `out` array
            out[out_pos] = temp_output

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    # Initialize local variables to avoid global reads
    # a = (batch, i, k), b = (batch, k, j), Out = (batch, i, j),
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0
    out_i_stride = out_strides[-2]
    out_j_stride = out_strides[-1]
    a_i_stride = a_strides[-2]
    a_k_stride = a_strides[-1]
    b_j_stride = b_strides[-1]
    b_k_stride = b_strides[-2]
    # Perform the matrix multiplication
    for batch in prange(out_shape[0]):
        for i in prange(out_shape[-2]):
            for j in prange(out_shape[-1]):
                # Calculate linear index for out
                out_pos = batch * out_batch_stride + i * out_i_stride + j * out_j_stride
                # Calculate linear index for a (start with k = 0)
                a_pos = batch * a_batch_stride + i * a_i_stride
                # Calculate linear index for b
                b_pos = batch * b_batch_stride + j * b_j_stride
                # Calculate the value for out
                temp = 0.0
                for _ in range(a_shape[-1]):
                    temp += a_storage[a_pos] * b_storage[b_pos]  # avoid global write
                    a_pos += a_k_stride  # add for each k
                    b_pos += b_k_stride  # add for each k
                out[out_pos] = temp


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
