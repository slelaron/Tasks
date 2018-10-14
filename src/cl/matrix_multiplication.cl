#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE 32

__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c, unsigned m, unsigned k, unsigned n)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    const unsigned local_x = get_local_id(0);
    const unsigned local_y = get_local_id(1);

    __local float buffer_a[TILE * TILE];
    __local float buffer_b[TILE * TILE];

    float result = 0;
    for (unsigned i = 0; i < k / TILE; i++) {
        const unsigned cur = i * TILE;
        const unsigned pos = local_y * TILE + local_x;
        buffer_a[pos] = (x < n && y < m) ? a[y * k + (cur + local_x)] : 0;
        buffer_b[pos] = (x < n && y < m) ? b[(cur + local_y) * n + x] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned j = 0; j < TILE; j++) {
            result += buffer_a[local_y * TILE + j] * buffer_b[j * TILE + local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * n + x] = result;
}