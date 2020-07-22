#include "cp.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <new>
using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

static double4_t *double4_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n))
    {
        throw std::bad_alloc();
    }
    return (double4_t *)tmp;
}
static inline double vsum(double4_t vv)
{
    double fsum = 0;
    for (int i = 0; i < 4; i++)
    {
        fsum += vv[i];
    }
    return fsum;
}

void correlate(int ny, int nx, const float *data, float *result)
{
    int nb = 4;
    int na = (nx + nb - 1) / nb;

    double4_t *d = double4_alloc(ny * na);
    double fsum;

    for (int i = 0; i < ny; i++)
    {
        double4_t sum = {0};

        for (int j = 0; j < na; j++)
        {
            for (int k = 0; k < nb; k++)
            {
                d[i * na + j][k] =  (nb * j + k < nx) ? (double)data[i * nx + nb * j + k] : 0;
            }
            sum += d[i * na + j];
        }

        fsum = vsum(sum) / nx;
        double4_t norm = {0};

        for (int j = 0; j < na; j++)
        {
            for (int k = 0; k < nb; k++)
            {
                d[i * na + j][k] -=  (nb * j + k < nx) ? fsum : 0;
            }
            norm += d[i * na + j] * d[i * na + j];
        }

        double fnorm = vsum(norm);
        fnorm = sqrt(fnorm);
        double4_t ffnorm = {fnorm, fnorm, fnorm, fnorm};
        for (int j = 0; j < na; j++)
        {
                d[i * na + j] /= ffnorm;
        }
    }

    for (int y = 0; y < ny; y++)
    {
        for (int x = y; x < ny; x++)
        {
            double4_t inner = {0};
            double finner = 0;
            for (int i = 0; i < na; i++)
            {
                inner += d[y * na + i] * d[x * na + i];
            }

            finner = vsum(inner);
            result[x + y * ny] = (float)finner;
        }
    }
    free(d);
}
