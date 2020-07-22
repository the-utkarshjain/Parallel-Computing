#include "is.h"
#include <cmath>
#include <cstdlib>
#include <x86intrin.h>
#include <vector>

struct Best
{
    double utility;
    int x0, y0, x1, y1;
};

Result segment(int ny, int nx, const float *data)
{

    std::vector<float> utk_sum((ny + 1) * (nx + 1));

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y <= ny; y++)
    {
        for (int x = 0; x <= nx; x++)
        {
            if (x == 0 || y == 0)
            {
                utk_sum[y * (nx + 1) + x] = 0;
                continue;
            }
            float sum_rectangle = utk_sum[y * (nx + 1) + x - 1];
            for (int j = 0; j < y; j++)
            {
                sum_rectangle += data[j * nx * 3 + (x - 1) * 3 + 0];
            }

            utk_sum[y * (nx + 1) + x] = sum_rectangle;
        }
    }

    float total_sum = utk_sum[(ny) * (nx + 1) + (nx)];

    struct Best *coordinates_best = (struct Best *)calloc(nx * ny, sizeof(struct Best));

    #pragma omp parallel for schedule(dynamic, 2)
    for (int size = 0; size < nx * ny; size++)
    {
        int size_x = size / ny + 1;
        int size_y = (size + 1) % ny + 1;
        double max_util = -std::numeric_limits<double>::infinity();
        int x0_optimal = 0;
        int y0_optimal = 0;
        int x1_optimal = 0;
        int y1_optimal = 0;

        int input_size = size_x * size_y;
        int output_size = nx * ny - input_size;
        if (output_size == 0)
            continue;
        float input_size_reci = 1.0 / input_size;
        float output_size_reci = 1.0 / output_size;
        float input_output_size_reci = output_size_reci + input_size_reci;
        float total_sum_output = 2 * output_size_reci * total_sum;
        float hsum_total_sq_sum = output_size_reci * (total_sum * total_sum);

        for (int y0 = 0; y0 <= ny - size_y; y0++)
        {
            for (int x0 = 0; x0 <= nx - size_x; x0++)
            {

                int x1 = x0 + size_x;
                int y1 = y0 + size_y;
                int new_nx = nx + 1;

                float input_sum = utk_sum[y1 * new_nx + x1] - utk_sum[y1 * new_nx + (x0)] - utk_sum[(y0)*new_nx + x1] + utk_sum[(y0)*new_nx + (x0)];

                float utility = (input_output_size_reci * input_sum * input_sum - total_sum_output * input_sum);

                if (utility > max_util)
                {
                    max_util = utility;
                    x0_optimal = x0;
                    y0_optimal = y0;
                    x1_optimal = x1 - 1;
                    y1_optimal = y1 - 1;
                }
            }
        }
        coordinates_best[size].utility = max_util + hsum_total_sq_sum;
        coordinates_best[size].x0 = x0_optimal;
        coordinates_best[size].x1 = x1_optimal;
        coordinates_best[size].y0 = y0_optimal;
        coordinates_best[size].y1 = y1_optimal;
    }

    double max_util = 0;
    int x0_optimal = 0, x1_optimal = 0, y0_optimal = 0, y1_optimal = 0;
    for (int size = 0; size < nx * ny; size++)
    {
        if (coordinates_best[size].utility > max_util)
        {
            max_util = coordinates_best[size].utility;
            x0_optimal = coordinates_best[size].x0;
            x1_optimal = coordinates_best[size].x1;
            y0_optimal = coordinates_best[size].y0;
            y1_optimal = coordinates_best[size].y1;
        }
    }

    int x0 = x0_optimal, x1 = x1_optimal, y0 = y0_optimal, y1 = y1_optimal;
    float input_sum = utk_sum[(y1 + 1) * (nx + 1) + (x1 + 1)] - utk_sum[(y1 + 1) * (nx + 1) + (x0)] - utk_sum[(y0) * (nx + 1) + (x1 + 1)] + utk_sum[(y0) * (nx + 1) + (x0)];
    float output_sum = total_sum - input_sum;
    int input_size = (x1_optimal - x0_optimal + 1) * (y1_optimal - y0_optimal + 1);
    int output_size = nx * ny - input_size;

    Result result = {
        y0_optimal, x0_optimal,y1_optimal + 1, x1_optimal + 1,
        {(float)output_sum / output_size, (float)output_sum / output_size, (float)output_sum / output_size},
        {(float)input_sum / input_size, (float)input_sum / input_size, (float)input_sum / input_size}};

    free(coordinates_best);
    return result;
}
