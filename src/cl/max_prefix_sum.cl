#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WARP_SIZE 1
#define WORK_GROUP 256

__kernel void max_prefix_sum(__global int* data, unsigned size, unsigned iteration, __global int* result)
{
    const unsigned id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned workGroup = get_local_size(0);
    const unsigned workGroupId = get_group_id(0);

    //Expected that workGroup and WORK_GROUP are the same
    __local int max_[2 * WORK_GROUP];
    __local int sum_[2 * WORK_GROUP];

    if (iteration == 0) {
        max_[local_id] = (id < size) ? data[id] : 0;
        sum_[local_id] = max_[local_id];
    } else {
        max_[local_id] = (id < size) ? data[id]        : 0;
        sum_[local_id] = (id < size) ? data[id + size] : 0;
    }

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

    const unsigned padding = (size + workGroup - 1) / workGroup;
    result[workGroupId]           = max_[start];
    result[workGroupId + padding] = sum_[start];
}
