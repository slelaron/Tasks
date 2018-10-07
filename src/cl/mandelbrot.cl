#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define THRESHOLD 256.0f
#define THRESHOLD2 (THRESHOLD * THRESHOLD)
#define ANTIALIASING 1

__kernel void mandelbrot(__global float* results,
        unsigned width, unsigned height,
        float fromX, float fromY,
        float sizeX, float sizeY,
        unsigned iterations,
        int smoothing)
{
    // TODO если хочется избавиться от зернистости и дрожжания при интерактивном погружении - добавьте anti-aliasing:
    // грубо говоря при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    float curResult = 0.0;
    for (int i = 0; i < ANTIALIASING; i++) {
        for (int j = 0; j < ANTIALIASING; j++) {
            float x0 = fromX + (x + 1.0f / (ANTIALIASING + 1) * (i + 1)) * sizeX / width;
            float y0 = fromY + (y + 1.0f / (ANTIALIASING + 1) * (j + 1)) * sizeY / height;

            float x = x0;
            float y = y0;

            int iter = 0;
            for (; iter < iterations; iter++) {
                float xPrev = x;
                x = x * x - y * y + x0;
                y = 2.0f * xPrev * y + y0;
                if ((x * x + y * y) > THRESHOLD2) {
                    break;
                }
            }
            float result = iter;

            if (smoothing && iter != iterations) {
                result = result - log(log(sqrt(x * x + y * y))) / log(THRESHOLD) / log(2.0f);
            }

            result = result / iterations;
            curResult += result;
        }
    }
    results[y * width + x] = curResult / ANTIALIASING / ANTIALIASING;
}
