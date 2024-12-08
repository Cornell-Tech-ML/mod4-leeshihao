from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)
    tiled = permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """2D average pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    tiled, new_height, new_width = tile(input, kernel)
    avg_pooled = tiled.mean(dim=4).view(
        input.shape[0], input.shape[1], new_height, new_width
    )
    return avg_pooled


# TODO: Implement for Task 4.4.
# Define max reduce function (done outside to reduce overhead)
max_reduce = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    max_val = max_reduce(input, dim)
    return input == max_val


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Max Forward"""
        ctx.save_for_backward(input, dim)
        max_vals = max_reduce(input, int(dim.item()))
        return max_vals

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backwards is the same as argmax"""
        input, dim = ctx.saved_values
        dim = int(dim.item())
        max_indices = argmax(input, dim)
        return grad_output * max_indices, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum values along a specified dimension.

    Args:
    ----
        input: Tensor of any shape.
        dim: The dimension along which to perform the max reduction.

    Returns:
    -------
        - A tensor of the maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    exponential = input.exp()
    t = exponential.sum(dim)
    return exponential / t


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    max_input = max(input, dim)
    logsumexp = (input - max_input).exp().sum(dim).log() + max_input
    return input - logsumexp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    tiled, new_height, new_width = tile(input, kernel)
    max_pooled = max(tiled, dim=4).view(
        input.shape[0], input.shape[1], new_height, new_width
    )
    return max_pooled


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor by zeroing out random elements.

    Args:
    ----
        input: Tensor of any shape.
        p: Probability of zeroing out each element (dropout rate). Must be between 0 and 1.
        ignore: If True, dropout is ignored and the input is returned unchanged.

    Returns:
    -------
        A tensor of the same shape as `input`, with random elements zeroed out based on `p` during training.

    """
    if ignore:
        return input
    mask = rand(input.shape) > p
    return input * mask
