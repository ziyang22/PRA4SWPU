"""Inference backend registry. Parameters remain ordinary nn.Conv2d parameters."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Spatial = tuple[int, int]
ConvBackend = Callable[[Tensor, Tensor, Tensor | None, Spatial, Spatial, Spatial, int], Tensor]
BACKENDS: dict[str, ConvBackend] = {}


def register_backend(name: str, implementation: ConvBackend) -> None:
    """Register before constructing a model; replacing an existing name is forbidden."""
    if not name or name in BACKENDS or not callable(implementation):
        raise ValueError(f"Invalid or duplicate backend: {name!r}")
    BACKENDS[name] = implementation


def reference_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Spatial,
    padding: Spatial,
    dilation: Spatial,
    groups: int,
) -> Tensor:
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


def example_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Spatial,
    padding: Spatial,
    dilation: Spatial,
    groups: int,
) -> Tensor:
    """Educational output-channel decomposition; not intended as an acceleration."""
    if groups != 1 or dilation != (1, 1) or stride != (1, 1):
        raise NotImplementedError("example supports groups=1, dilation=1, stride=1 only")
    if weight.shape[-2:] not in ((1, 1), (3, 3)):
        raise NotImplementedError("example supports 1x1 and 3x3 kernels only")
    if input.dtype != torch.float32:
        raise NotImplementedError("example supports float32 only")
    outputs = []
    for start in range(0, weight.shape[0], 32):
        end = start + 32
        block_bias = None if bias is None else bias[start:end]
        outputs.append(F.conv2d(input, weight[start:end], block_bias, stride, padding))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


@lru_cache(maxsize=1)
def load_hip_extension():
    """Compile with the active DCU PyTorch toolchain; no CPU/CUDA fallback."""
    if not torch.version.hip:
        raise RuntimeError("hip backend requires a HIP/DCU build of PyTorch")
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="pra4swpu_hip_conv",
        sources=[str(source / "bindings.cpp"), str(source / "conv2d.hip")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        with_cuda=True,
        verbose=True,
    )


def hip_conv2d(input, weight, bias, stride, padding, dilation, groups):
    """FP32 contiguous inference example implemented in HIP, bound with pybind11."""
    if not torch.version.hip or input.device.type != "cuda":
        raise RuntimeError("hip backend requires HIP/DCU tensors on the cuda device")
    if torch.is_grad_enabled():
        raise RuntimeError("hip backend is inference only: use inference_mode")
    extension = load_hip_extension()
    stream = torch.cuda.current_stream(input.device).cuda_stream
    return extension.conv2d(input, weight, bias, stride, padding, dilation, groups, stream)


register_backend("torch", reference_conv2d)
register_backend("example", example_conv2d)
register_backend("hip", hip_conv2d)


class ReplaceableConv2d(nn.Conv2d):
    """NCHW input, OIHW weights. Custom backends require eval and disabled gradients."""

    def __init__(self, *args, backend="torch", **kwargs):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; available: {list(BACKENDS)}")
        super().__init__(*args, **kwargs)
        if self.padding_mode != "zeros" or isinstance(self.padding, str):
            raise ValueError("Only explicit zero padding is supported")
        self.backend = backend

    def forward(self, input: Tensor) -> Tensor:
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend {self.backend!r}")
        if self.backend != "torch" and (self.training or torch.is_grad_enabled()):
            raise RuntimeError("Custom backends are inference only: use eval + inference_mode")
        output = BACKENDS[self.backend](
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.backend != "torch":
            expected = (
                input.shape[0],
                self.out_channels,
                *[
                    (
                        input.shape[i + 2]
                        + 2 * self.padding[i]
                        - self.dilation[i] * (self.kernel_size[i] - 1)
                        - 1
                    )
                    // self.stride[i]
                    + 1
                    for i in range(2)
                ],
            )
            if (
                output.shape != expected
                or output.dtype != input.dtype
                or output.device != input.device
            ):
                raise RuntimeError("Backend violated output shape/dtype/device contract")
        return output
