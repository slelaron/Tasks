#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
#define LAST            16
#define AND             15

__kernel void radix(
        __global unsigned int* as,
        __global unsigned int* buffer,
        __global unsigned int* result,
        unsigned total_sum_ind,
        unsigned n,
        unsigned stage, // bitset: 0 - upOrDown, 1 - readFromAsOrFromBuffer, 2 - writeToResultOrToBuffer, 3 - needToMakeNodeZero
        unsigned shift,
        unsigned write_point,
        unsigned read_point)
{
    const unsigned global_id = get_global_id(0);
    const unsigned  local_id = get_local_id (0);
    const unsigned  group_id = get_group_id(0);

    __local int memory[2 * WORK_GROUP_SIZE][LAST];

    if (stage & 2) {
        for (unsigned i = 0; i < LAST; i++) {
            memory[local_id][i] = 0;
        }
        if (global_id < n) {
            memory[local_id][(as[global_id] >> shift) & AND] = 1;
        }
    } else {
        for (unsigned i = 0; i < LAST; i++) {
            memory[local_id][i] = (global_id < n) ? buffer[read_point + LAST * global_id + i] : 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int next_pack = WORK_GROUP_SIZE;
    int  cur_pack = 0;

    for (unsigned i = WORK_GROUP_SIZE / 2; i > 0; i >>= 1) {
        if (local_id < i) {
            for (unsigned j = 0; j < LAST; j++) {
                unsigned sum =
                        memory[cur_pack + 2 * local_id][j] + memory[cur_pack + 2 * local_id + 1][j];

                memory[next_pack + local_id][j] = sum;
            }
        }
        cur_pack = next_pack;
        next_pack += i;

        barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (stage & 1) {
        if (local_id == 0) {
            for (unsigned i = 0; i < LAST; i++) {
                buffer[write_point + LAST * group_id + i] = memory[cur_pack][i];
            }
        }
    } else {
        if (local_id == 0) {
            if (stage & 8) {
                for (unsigned i = 0; i < LAST; i++) {
                    memory[cur_pack][i] = 0;
                }
            } else {
                for (unsigned i = 0; i < LAST; i++) {
                    memory[cur_pack][i] = buffer[write_point + LAST * group_id + i];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned i = 1; i < WORK_GROUP_SIZE; i <<= 1) {
            next_pack = cur_pack;
            cur_pack -= 2 * i;

            if (local_id < i) {
                for (unsigned j = 0; j < LAST; j++) {
                    memory[cur_pack + 2 * local_id + 1][j] = memory[cur_pack + 2 * local_id][j] + memory[next_pack + local_id][j];
                    memory[cur_pack + 2 * local_id    ][j] = memory[next_pack + local_id][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (!(stage & 4)) {
            for (unsigned i = 0; i < LAST; i++) {
                if (global_id < n) {
                    buffer[read_point + LAST * global_id + i] = memory[cur_pack + local_id][i];
                }
            }
        } else {
            if (local_id == 0) {
                for (unsigned i = 0; i < LAST; i++) {
                    memory[2 * WORK_GROUP_SIZE - 1][i] = 0;
                }
                for (unsigned i = 1; i < LAST; i++) {
                    memory[2 * WORK_GROUP_SIZE - 1][i] = memory[2 * WORK_GROUP_SIZE - 1][i - 1] + buffer[total_sum_ind + i - 1];
                }
            }
            int element = as[global_id];

            unsigned index  = memory[local_id               ][(element >> shift) & AND];
            unsigned margin = memory[2 * WORK_GROUP_SIZE - 1][(element >> shift) & AND];
            result[index + margin] = element;
        }
    }
}
