#include "cp.h"
#include <cmath>
#include <stdlib.h>
#include <new>
#include <tuple>
#include <vector>
#include <algorithm>
#include <immintrin.h>

using namespace std;

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr double4_t double4_0 = {0,0,0,0};
const int dim_Block = 4;

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

double vsum(double4_t& x)
{
    double sum = 0;
    for (int i = 0; i < 4; ++i)
    {
        sum += x[i];
    }
    return sum;
}

void correlate(int ny, int nx, const float* data, float* result) {

    int nb = 4;
    int na = (nx-1)/4 + 1;
    double4_t* d = double4_alloc(ny*na);

    #pragma omp parallel for
    for (int i = 0; i < ny; i++)
    {   
        double fsum;
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
    
    int ctr = 0;
    vector< tuple<int, int, int> > rows(((ny - dim_Block +1)/dim_Block + 1) * (((ny - dim_Block +1)/(dim_Block))/2 + 2));

    for (int ia = 0; ia < ny - dim_Block +1; ia += dim_Block) {
        for (int ja = ia; ja < ny - dim_Block +1; ja += dim_Block) {
            int ija = _pdep_u32(ia, 0x55555555) | _pdep_u32(ja, 0xAAAAAAAA);
            rows[ctr] = make_tuple(ija, ia, ja);
            ++ctr;
        }
    }
    sort(rows.begin(), rows.begin()+ctr);


    #pragma omp parallel for schedule(dynamic,2)
    for(int itr = 0; itr < ctr; ++itr)
    {
        int i = get<1>(rows[itr]), j = get<2>(rows[itr]);

            double4_t val[2*dim_Block];
            double4_t fi_value[dim_Block * dim_Block];

            for(int k = 0; k < dim_Block * dim_Block; ++k)
            {
                fi_value[k] = double4_0;
            }

            for(int k = 0; k<na/2; ++k)
            {
                for(int l = 0; l<dim_Block; ++l)
                {
                    val[2*l] = d[(i+l)*na + k];
                    val[2*l+1] = d[(j+l)*na + k];
                }

                for(int l = 0; l < dim_Block; ++l)
                {
                    for(int m = 0; m < dim_Block; ++m)
                    {
                        fi_value[dim_Block*l+m] += val[2*l] * val[2*m+1];
                    }
                }

            }

            for(int l = 0; l < dim_Block; ++l)
            {
                for(int m = 0; m < dim_Block; ++m)
                {
                    result[j+m+(i+l)*ny] = vsum(fi_value[dim_Block*l+m]);
                }
            }
    }

    #pragma omp parallel for schedule(dynamic,2)
    for(int itr = 0; itr < ctr; ++itr)
    {
        int i = get<1>(rows[itr]), j = get<2>(rows[itr]);

            double4_t val[2*dim_Block];
            double4_t fi_value[dim_Block * dim_Block];

            for(int k = 0; k < dim_Block * dim_Block; ++k)
            {
                fi_value[k] = double4_0;
            }

            for(int k = na/2; k<na ; ++k)
            {
                for(int l = 0; l<dim_Block; ++l)
                {
                    val[2*l] = d[(i+l)*na + k];
                    val[2*l+1] = d[(j+l)*na + k];
                }

                for(int l = 0; l < dim_Block; ++l)
                {
                    for(int m = 0; m < dim_Block; ++m)
                    {
                        fi_value[dim_Block*l+m] += val[2*l] * val[2*m+1];
                    }
                }

            }

            for(int l = 0; l < dim_Block; ++l)
            {
                for(int m = 0; m < dim_Block; ++m)
                {
                    result[j+m+(i+l)*ny] += vsum(fi_value[dim_Block*l+m]);
                }
            }
    }

    if(ny<=2)
    {
        #pragma omp parallel for schedule(dynamic,2)
        for(int j = 0; j<ny; ++j)
        {
            double4_t last_row = double4_0, sec_last_row = double4_0;
            for(int k = 0; k < na; k++)
            {
                last_row += d[(j)*na + k] * d[(ny-1)*na + k];
                sec_last_row += d[(j)*na + k] * d[(ny-2)*na + k];
            }
            result[ny-1+j*ny] = vsum(last_row);
            result[ny-2+j*ny] = vsum(sec_last_row);
        }
    }
    else
    {
        #pragma omp parallel for schedule(dynamic,2)
        for(int j = 0; j<ny; ++j)
        {
            double4_t last_row = double4_0, sec_last_row = double4_0, third_last_row = double4_0;
            for(int k = 0; k < na; k++)
            {
                last_row += d[(j)*na + k] * d[(ny-1)*na + k];
                sec_last_row += d[(j)*na + k] * d[(ny-2)*na + k];
                third_last_row += d[(j)*na + k] * d[(ny-3)*na + k];
            }
            result[ny-1+j*ny] = vsum(last_row);
            result[ny-2+j*ny] = vsum(sec_last_row);
            result[ny-3+j*ny] = vsum(third_last_row);

        }
    }

    free(d);
}