#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WARP_SIZE 1
#define WORK_GROUP 1024

__kernel void max_prefix_sum(__global const int* data, unsigned size, __global int* result)
{
    const unsigned local_id = get_local_id(0);
    const unsigned workGroup = get_local_size(0);

    //Expected that workGroup and WORK_GROUP are the same
    __local int max_[2 * WORK_GROUP];
    __local int sum_[2 * WORK_GROUP];

    int global_sum = 0;
    __local int result_;
    if (local_id == 0) {
        result_ = 0;
    }

    const unsigned iterations = (size + WORK_GROUP - 1) / WORK_GROUP;
    for (unsigned iter = 0; iter < iterations; iter++) {
        const unsigned index = WORK_GROUP * iter + local_id;
        max_[local_id] = (index < size) ? data[index] : 0;
        if (local_id == 0) {
            max_[local_id] += global_sum;
        }
        sum_[local_id] = max_[local_id];
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned start = 0;
        for (unsigned curSize = WORK_GROUP; curSize > 1; curSize /= 2) {
            const unsigned nextIndex = start + curSize + local_id;
            const unsigned a = start + 2 * local_id;
            const unsigned b = start + 2 * local_id + 1;

            if (b < start + curSize) {
                sum_[nextIndex] = sum_[a] + sum_[b];
                max_[nextIndex] = max_[a];
                const int value = max_[b] + sum_[a];
                if (max_[nextIndex] < value) {
                    max_[nextIndex] = value;
                }
            }
            start += curSize;

            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (local_id == 0) {
            if (result_ < max_[start]) {
                result_ = max_[start];
            }
        }
        global_sum = sum_[start];
    }
    result[0] = result_;
}
