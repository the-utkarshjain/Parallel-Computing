#include "cp.h"
#include <x86intrin.h>
#include <cmath>
#include "vector.h"
using namespace std;

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

void correlate(int ny, int nx, const float* data, float* result) {
    
    int na = (ny + 8 - 1) / 8;
    float8_t* vd = float8_alloc(na*nx);  

    #pragma omp parallel for
    for (int ja = 0; ja < na; ++ja) {
        for (int i = 0; i < nx; ++i) {
            for (int jb = 0; jb < 8; ++jb) {
                int j = ja * 8 + jb;
                vd[nx*ja + i][jb] = j < ny ? data[nx*j + i] : 0;
            }
        }
    }

    float8_t div = {(float)nx,(float)nx,(float)nx,(float)nx,(float)nx,(float)nx,(float)nx,(float)nx};

    #pragma omp parallel for
    for(int i=0; i<na; i++){

        float8_t temp = {0};
        for(int j=0; j<nx ;j++){
            temp += vd[nx*i + j];
        }

        temp = temp/div;
        float8_t mag = {0};

        for(int j=0; j<nx ;j++){
            vd[nx*i + j] -= temp;
            mag += vd[nx*i+j]*vd[nx*i+j];
        }

        for(int z=0;z<8;z++)
         mag[z] = sqrt(mag[z]);

        for(int j=0; j<nx ;j++){
            vd[nx*i + j] /= mag;
        }
    }

    #pragma omp parallel for schedule(dynamic,2)
    for(int ia=0; ia<na; ia++){
        for(int ja=ia; ja<na; ja++){
            float8_t vv000 = {0};
            float8_t vv001 = {0};
            float8_t vv010 = {0};
            float8_t vv011 = {0};
            float8_t vv100 = {0};
            float8_t vv101 = {0};
            float8_t vv110 = {0};
            float8_t vv111 = {0};

            for(int k=0; k<nx; k++){
                float8_t a000 = vd[nx*ia + k];
                float8_t b000 = vd[nx*ja + k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);
                vv000 += a000*b000;
                vv001 += a000*b001;
                vv010 += a010*b000;
                vv011 += a010*b001;
                vv100 += a100*b000;
                vv101 += a100*b001;
                vv110 += a110*b000;
                vv111 += a110*b001;
            }
            float8_t vv[8] = { vv000, vv001, vv010, vv011, vv100, vv101, vv110, vv111 };
            for (int kb = 1; kb < 8; kb += 2) {
                vv[kb] = swap1(vv[kb]);
            }

            for (int ib = 0; ib < 8; ++ib) {
                for (int jb = 0; jb < 8; ++jb) {
                    int i = ib + ia*8;
                    int j = jb + ja*8;
                    if (j < ny && i < ny) {
                        result[ny*i + j] =  vv[ib^jb][jb];
                    }
                }
            }
        }
    }

    free(vd);
}