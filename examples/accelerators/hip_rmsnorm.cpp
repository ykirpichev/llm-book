// Source-only HIP RMSNorm teaching program.  It has not been compiled or run
// in this repository's validation environment.
// Fixed fixture: FP32 forward-only, 3 rows, widths 3/129/4097, positive element
// strides, finite bounded values, epsilon 1e-6, and exactly 256 threads/block.
// No general user-input wrapper, backward pass, or performance claim is supplied.
// std::vector staging is pageable; hipMemcpyAsync need not overlap host work.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr float kEpsilon = 1.0e-6f;

void check_hip(hipError_t status, const char* expression, const char* file, int line) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 " " + expression + ": " + hipGetErrorString(status));
    }
}

#define HIP_CHECK(expression) check_hip((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            // Destructors cannot safely throw during stack unwinding. All
            // allocations/copies/launches/synchronization are checked explicitly.
            // Traditional hipMalloc/hipFree is used here, not the async allocator;
            // hipFree has an implicit device synchronization for these allocations.
            hipFree(pointer_);
        }
    }

    T* get() const { return pointer_; }

private:
    T* pointer_ = nullptr;
};

class Stream {
public:
    Stream() {
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    ~Stream() {
        if (stream_ != nullptr) {
            hipStreamDestroy(stream_);
        }
    }

    hipStream_t get() const { return stream_; }

private:
    hipStream_t stream_ = nullptr;
};

// One block owns one complete row. Every one of the 256 threads must reach
// each barrier, including threads that own no element of a short row.
__global__ void rmsnorm_f32(const float* x, const float* weight, float* y,
                             int64_t n, int64_t sx0, int64_t sx1, int64_t sw,
                             float epsilon) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t tid = static_cast<int64_t>(threadIdx.x);
    __shared__ float sums[kThreads];

    float local_sum = 0.0f;
    for (int64_t column = tid; column < n; column += kThreads) {
        // The multiplication is intentionally int64 so a large row stride
        // cannot silently wrap a 32-bit intermediate offset.
        const float value = x[row * sx0 + column * sx1];
        local_sum += value * value;
    }
    sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int step = kThreads / 2; step != 0; step >>= 1) {
        if (threadIdx.x < step) {
            sums[threadIdx.x] += sums[threadIdx.x + step];
        }
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(sums[0] / static_cast<float>(n) + epsilon);
    for (int64_t column = tid; column < n; column += kThreads) {
        const float value = x[row * sx0 + column * sx1];
        const float scale = weight[column * sw];
        // Output intentionally has contiguous rows, regardless of x strides.
        y[row * n + column] = value * inverse_rms * scale;
    }
}

float input_value(int row, int64_t column) {
    if (row == 0) {
        return 0.0f;  // Zero row tests the epsilon path and signed zero behavior.
    }
    const int signed_pattern = static_cast<int>(column % 17) - 8;
    return (row == 1 ? 0.125f : -0.0625f) * static_cast<float>(signed_pattern);
}

float weight_value(int64_t column) {
    const float magnitude = 0.25f + 0.03125f * static_cast<float>(column % 5);
    return (column & 1) == 0 ? magnitude : -magnitude;
}

void compare_with_cpu(const std::vector<float>& input, const std::vector<float>& weights,
                      const std::vector<float>& actual, int rows, int64_t n,
                      int64_t sx0, int64_t sx1, int64_t sw) {
    constexpr double kAbsoluteTolerance = 2.0e-4;
    constexpr double kRelativeTolerance = 3.0e-4;
    for (int row = 0; row < rows; ++row) {
        double square_sum = 0.0;
        for (int64_t column = 0; column < n; ++column) {
            const double value = input[static_cast<int64_t>(row) * sx0 + column * sx1];
            square_sum += value * value;
        }
        const double inverse_rms = 1.0 / std::sqrt(square_sum / n + kEpsilon);
        for (int64_t column = 0; column < n; ++column) {
            const double expected =
                input[static_cast<int64_t>(row) * sx0 + column * sx1] * inverse_rms *
                weights[column * sw];
            const double observed = actual[static_cast<int64_t>(row) * n + column];
            const double allowed = kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
            if (!std::isfinite(observed) || std::abs(observed - expected) > allowed) {
                throw std::runtime_error("CPU comparison failed at row " + std::to_string(row) +
                                         ", column " + std::to_string(column));
            }
        }
    }
}

void run_case(int64_t n) {
    constexpr int kRows = 3;
    const int64_t sx0 = 2 * n + 3;
    constexpr int64_t sx1 = 2;
    constexpr int64_t sw = 2;
    std::vector<float> input(static_cast<std::size_t>(kRows * sx0), -99.0f);
    std::vector<float> weights(static_cast<std::size_t>(n * sw), -99.0f);
    std::vector<float> output(static_cast<std::size_t>(kRows * n));
    for (int row = 0; row < kRows; ++row) {
        for (int64_t column = 0; column < n; ++column) {
            input[static_cast<int64_t>(row) * sx0 + column * sx1] = input_value(row, column);
        }
    }
    for (int64_t column = 0; column < n; ++column) {
        weights[column * sw] = weight_value(column);
    }

    // This non-default stream is an explicit lifetime example: buffers stay
    // alive until its async copies, kernel, output copy, and synchronization
    // have completed. A different stream would need an explicit dependency.
    Stream stream;
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_weights(weights.size());
    DeviceBuffer<float> device_output(output.size());
    HIP_CHECK(hipMemcpyAsync(device_input.get(), input.data(), input.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream.get()));
    HIP_CHECK(hipMemcpyAsync(device_weights.get(), weights.data(), weights.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream.get()));
    hipLaunchKernelGGL(rmsnorm_f32, dim3(kRows), dim3(kThreads), 0, stream.get(),
                       device_input.get(), device_weights.get(), device_output.get(), n,
                       sx0, sx1, sw, kEpsilon);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(output.data(), device_output.get(), output.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    compare_with_cpu(input, weights, output, kRows, n, sx0, sx1, sw);
    std::printf("PASS N=%lld (zero and mixed-sign rows; strided input)\n",
                static_cast<long long>(n));
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no HIP device is available");
        }
        HIP_CHECK(hipSetDevice(0));
        for (const int64_t n : {int64_t{3}, int64_t{129}, int64_t{4097}}) {
            run_case(n);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
