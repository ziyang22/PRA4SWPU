// pybind11 entry point. No host copies, implicit casts or synchronization.
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <array>
#include <cstdint>
#include <optional>

int launch_conv2d_hip(const float*, const float*, const float*, float*,
                     int64_t, int64_t, int64_t, int64_t,
                     int64_t, int64_t, int64_t, int64_t, int64_t, uintptr_t);

at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight,
                  const std::optional<at::Tensor>& bias,
                  std::array<int64_t, 2> stride, std::array<int64_t, 2> padding,
                  std::array<int64_t, 2> dilation, int64_t groups, uintptr_t stream) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "Inference only: disable gradients");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Expected HIP/DCU tensors");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "Expected NCHW/OIHW tensors");
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "Only float32 is supported");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
                "Only contiguous NCHW/OIHW is supported; implicit copies are forbidden");
    TORCH_CHECK(input.device() == weight.device(), "Device mismatch");
    TORCH_CHECK(stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
                dilation[1] == 1 && groups == 1, "Only stride=1, dilation=1, groups=1");
    TORCH_CHECK(padding[0] >= 0 && padding[0] <= 1 && padding[1] >= 0 && padding[1] <= 1,
                "Only padding 0 or 1 is supported");
    const auto n = input.size(0), ci = input.size(1), h = input.size(2), w = input.size(3);
    const auto co = weight.size(0), kh = weight.size(2), kw = weight.size(3);
    TORCH_CHECK(n > 0 && ci > 0 && h > 0 && w > 0 && co > 0, "Empty tensors unsupported");
    TORCH_CHECK(ci == weight.size(1), "Input channel mismatch");
    TORCH_CHECK((kh == 1 && kw == 1) || (kh == 3 && kw == 3), "Only 1x1/3x3 kernels");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == input.device() && bias->scalar_type() == at::kFloat &&
                    bias->dim() == 1 && bias->size(0) == co && bias->is_contiguous(),
                    "Bias must be contiguous float32 [out_channels] on the input device");
    }
    const int64_t oh = h + 2 * padding[0] - kh + 1;
    const int64_t ow = w + 2 * padding[1] - kw + 1;
    TORCH_CHECK(oh > 0 && ow > 0, "Nonpositive output size");
    const c10::DeviceGuard guard(input.device());
    auto output = at::empty({n, co, oh, ow}, input.options());
    const int status = launch_conv2d_hip(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr, output.data_ptr<float>(),
        ci, h, w, co, kh, kw, padding[0], padding[1], output.numel(), stream);
    TORCH_CHECK(status == 0, "HIP kernel launch failed, error code ", status);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "FP32 HIP convolution (internal current-stream argument)");
}
