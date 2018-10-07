#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WARP_SIZE 1
#define WORK_GROUP 256

__kernel void sum(__global const unsigned* data, unsigned size, __global unsigned* result)
{
    const unsigned id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned workGroup = get_local_size(0);

    //Expected that workGroup and WORK_GROUP are the same
    __local unsigned local_[WORK_GROUP];

    local_[local_id] = (id >= size) ? 0 : data[id];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned curSize = workGroup / 2; curSize > 0; curSize /= 2) {
        if (local_id < curSize) {
            local_[local_id] += local_[local_id + curSize];
        }

        if (curSize >= WARP_SIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (local_id == 0) {
        atomic_add(result, local_[0]);
    }
}