#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void bitonic(__global float* as, __global float* result, unsigned n, unsigned i, unsigned size)
{
    const unsigned global_id = get_global_id(0);
    const unsigned local_id  = get_local_id(0);
    const bool fparity = global_id / 2 / i % 2 == 1;

    __local float memory[WORK_GROUP_SIZE];

    // Assume, that size of 'as' is a power of 2, otherwise there are some problems on borders.

    if (2 * size > WORK_GROUP_SIZE) {
        const bool fparity = global_id / 2 / i % 2 == 1;
        const bool parity = global_id / size % 2 == 1;
        const unsigned next_item = global_id + (!parity ? size : -size);

        float a = as[global_id];
        float b = as[next_item];
        float min_el = min(a, b);
        float max_el = max(a, b);

        result[global_id] = (!parity ^ fparity) ? min_el : max_el;
    } else {
        memory[local_id] = as[global_id];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned cur = size; cur != 0; cur /= 2) {
            const bool parity = global_id / cur % 2 == 1;
            const unsigned next_item = local_id + (!parity ? cur : -cur);

            float a = memory[local_id];
            float b = memory[next_item];
            float min_el = min(a, b);
            float max_el = max(a, b);

            barrier(CLK_LOCAL_MEM_FENCE);

            memory[local_id] = (!parity ^ fparity) ? min_el : max_el;

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[global_id] = memory[local_id];
    }
}
