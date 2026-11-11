// Readable reference kernel, not an optimized convolution implementation.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias,
                              float* output, int64_t ci, int64_t h, int64_t w,
                              int64_t co, int64_t kh, int64_t kw, int64_t ph,
                              int64_t pw, int64_t total) {
    const int64_t oh = h + 2 * ph - kh + 1, ow = w + 2 * pw - kw + 1;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t ox = index % ow, oy = (index / ow) % oh;
        const int64_t oc = (index / (ow * oh)) % co, batch = index / (ow * oh * co);
        // Wider accumulation limits rounding drift across the full U-Net.
        double value = 0.0;
        for (int64_t ic = 0; ic < ci; ++ic) {
            for (int64_t ky = 0; ky < kh; ++ky) {
                const int64_t iy = oy + ky - ph;
                for (int64_t kx = 0; kx < kw; ++kx) {
                    const int64_t ix = ox + kx - pw;
                    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                        value += static_cast<double>(input[((batch * ci + ic) * h + iy) * w + ix]) *
                                 static_cast<double>(weight[((oc * ci + ic) * kh + ky) * kw + kx]);
                    }
                }
            }
        }
        output[index] = static_cast<float>(value + (bias ? bias[oc] : 0.0f));
    }
    return;
}

int launch_conv2d_hip(const float* input, const float* weight, const float* bias,
                     float* output, int64_t ci, int64_t h, int64_t w,
                     int64_t co, int64_t kh, int64_t kw, int64_t ph, int64_t pw,
                     int64_t total, uintptr_t stream) {
    const unsigned int blocks = static_cast<unsigned int>((total + 255) / 256 > 65535
                                    ? 65535 : (total + 255) / 256);
    hipLaunchKernelGGL(conv2d_kernel, dim3(blocks), dim3(256), 0,
                       reinterpret_cast<hipStream_t>(stream), input, weight, bias,
                       output, ci, h, w, co, kh, kw, ph, pw, total);
    return static_cast<int>(hipGetLastError());
}
