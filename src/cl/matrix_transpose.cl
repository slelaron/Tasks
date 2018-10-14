#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE 32

__kernel void matrix_transpose(__global int* data, __global int* result, unsigned height, unsigned width)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    const unsigned local_x = get_local_id(0);
    const unsigned local_y = get_local_id(1);

    __local int buffer[TILE * TILE];
    if (x < width && y < height) {
        buffer[local_y * TILE + local_x] = data[y * width + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned start_x = get_group_id(0) * TILE;
    const unsigned start_y = get_group_id(1) * TILE;
    if (x < width && y < height) {
        result[(start_x + local_y) * height + (start_y + local_x)] = buffer[local_x * TILE + local_y];
    }
}