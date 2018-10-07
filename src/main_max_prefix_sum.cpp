#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, const char* message, const char* filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    unsigned benchmarkingIters = 10;
    unsigned max_n = (1 << 24);

    for (unsigned n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023u, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCLgpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");

            bool printLog = false;
            kernel.compile(printLog);

            const unsigned workGroupSize = 256;

            gpu::gpu_mem_32i storage;
            gpu::gpu_mem_32i storage1;
            storage.resizeN(n);
            storage1.resizeN(2 * (n + workGroupSize - 1) / workGroupSize);

            {
                timer t;

                for (int iter = 0; iter < benchmarkingIters; iter++) {
                    storage.writeN(as.data(), n);

                    unsigned i = 0;
                    for (unsigned now = n; now > 1; now = (now + workGroupSize - 1) / workGroupSize, i++) {
                        gpu::WorkSize size(workGroupSize, (now + workGroupSize - 1) / workGroupSize * workGroupSize);
                        if (i % 2 == 0) {
                            kernel.exec(size, storage, now, i, storage1);
                        } else {
                            kernel.exec(size, storage1, now, i, storage);
                        }
                    }

                    int max_sum = 0;
                    if (i % 2 == 0) {
                        storage.readN(&max_sum, 1);
                    } else {
                        storage1.readN(&max_sum, 1);
                    }
                    EXPECT_THE_SAME(reference_max_sum, std::max(0, max_sum), "GPU result should be consistent!");
                    t.nextLap();
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
