#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    cs_gpu.resizeN(n);

    gpu::gpu_mem_32u* ass = &as_gpu;
    gpu::gpu_mem_32u* css = &cs_gpu;

    const unsigned LAST = 16;
    const unsigned BITS_COUNT = 4;

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            ass->writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            const unsigned int workGroupSize = 128;
            for (int shift = 0; shift < 32; shift += BITS_COUNT) {
                std::vector<unsigned> positions_in_buffer{0, 0};
                std::vector<unsigned> wg_sizes;
                for (unsigned i = n; i != 1; i = (i + workGroupSize - 1) / workGroupSize) {
                    wg_sizes.push_back(i);
                    unsigned int global_work_size = (i + workGroupSize - 1) / workGroupSize * workGroupSize;
                    unsigned stage = 1u + 2u * (i == n);
                    radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               *ass, bs_gpu, *css, 0, i, stage, shift, positions_in_buffer.back(), positions_in_buffer[positions_in_buffer.size() - 2]);
                    positions_in_buffer.push_back(positions_in_buffer.back() + LAST * ((i + workGroupSize - 1) / workGroupSize));
                }
                positions_in_buffer.pop_back();
                unsigned total_sum_ind = positions_in_buffer.back();
                for (int i = (int)wg_sizes.size() - 1; i >= 0; i--) {
                    unsigned int global_work_size = (wg_sizes[i] + workGroupSize - 1) / workGroupSize * workGroupSize;
                    unsigned stage = 6u * (wg_sizes[i] == n) + 8u * ((int)wg_sizes.size() - 1 == i);
                    radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               *ass, bs_gpu, *css, total_sum_ind, wg_sizes[i], stage, shift, positions_in_buffer.back(), positions_in_buffer[positions_in_buffer.size() - 2]);
                    positions_in_buffer.pop_back();
                }
                std::swap(ass, css);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        ass->readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
