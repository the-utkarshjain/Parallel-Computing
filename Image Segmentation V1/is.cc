#include "is.h"
#include<cmath>
#include<cstdlib>
#include <x86intrin.h>
#include <new>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

double vsum(double4_t x){
    double utk_sum = 0;
    for(int i=0; i<4; i++){
        utk_sum += x[i];
    }
    return utk_sum;
}

struct optimal{
    double utility;
    int x0, y0, x1, y1;
};

Result segment(int ny, int nx, const float* data) {

    double4_t* utk_sum = double4_alloc((ny+1)*(nx+1));
    double4_t vector_zero = {0};

    #pragma omp parallel for schedule(dynamic, 1)
    for(int y=0; y<=ny; y++){
        for(int x=0; x<=nx; x++){
            if(x==0 || y==0){
                utk_sum[y*(nx+1)+x] = vector_zero;
                continue;
            }
            double sum_rectangle[3] = {utk_sum[y*(nx+1)+x-1][0], utk_sum[y*(nx+1)+x-1][1], utk_sum[y*(nx+1)+x-1][2]};
            for(int j=0; j<y; j++){
                for(int c=0; c<3; c++){
                    sum_rectangle[c] += (double)data[j*nx*3 + (x-1)*3 + c];
                }
            }
            utk_sum[y*(nx+1)+x][0] = sum_rectangle[0];
            utk_sum[y*(nx+1)+x][1] = sum_rectangle[1];
            utk_sum[y*(nx+1)+x][2] = sum_rectangle[2];
            utk_sum[y*(nx+1)+x][3] = 0;
        }
    }

    double4_t total_sum = utk_sum[(ny)*(nx+1)+(nx)];
    struct optimal* coordinates_best = (struct optimal*)calloc(nx*ny, sizeof(struct optimal));
    
    #pragma omp parallel for schedule(dynamic, 1)
    for(int size=0; size<nx*ny; size++){
        int size_x = size/ny+1;
        int size_y = (size+1)%ny+1;
        double max_util = 0;
        int  x0_optimal=0;
        int y0_optimal=0;
        int x1_optimal=0;
        int y1_optimal=0;
        
        int input_size = size_x*size_y;
        int output_size = nx*ny - input_size;
        if(output_size==0)continue; 
        double input_size_reci = 1.0/input_size;
        double output_size_reci = 1.0/output_size;
        int new_nx = nx+1;
        for(int y0=0; y0<=ny-size_y; y0++){
            for(int x0=0; x0<=nx-size_x; x0++){
                int x1 = x0+size_x-1;
                int y1 = y0+size_y-1;
                
                double4_t input_sum = utk_sum[(y1+1)*new_nx+(x1+1)] - utk_sum[(y1+1)*new_nx+(x0)] - utk_sum[(y0)*new_nx+(x1+1)] + utk_sum[(y0)*new_nx+(x0)];
                double4_t output_sum = total_sum - input_sum;
                double utility = vsum(input_sum*input_sum*input_size_reci + output_sum*output_sum*output_size_reci);   
            
                if(utility>max_util){
                    max_util = utility;
                    x0_optimal = x0;
                    y0_optimal = y0;
                    x1_optimal = x1;
                    y1_optimal = y1;
                }
            }
        }
        coordinates_best[size].utility = max_util;
        coordinates_best[size].x0 = x0_optimal;
        coordinates_best[size].x1 = x1_optimal;
        coordinates_best[size].y0 = y0_optimal;
        coordinates_best[size].y1 = y1_optimal;

    }
    
    double max_util = 0;
    int x0_optimal=0, x1_optimal=0, y0_optimal=0, y1_optimal=0;
    for(int size=0; size<nx*ny; size++){
        if(coordinates_best[size].utility>max_util){
            max_util = coordinates_best[size].utility;
            x0_optimal = coordinates_best[size].x0;
            x1_optimal = coordinates_best[size].x1;
            y0_optimal = coordinates_best[size].y0;
            y1_optimal = coordinates_best[size].y1;
        }
    }


    int x0=x0_optimal;
    int x1=x1_optimal;
    int y0=y0_optimal;
    int y1=y1_optimal;
    double4_t input_sum = utk_sum[(y1+1)*(nx+1)+(x1+1)] - utk_sum[(y1+1)*(nx+1)+(x0)] - utk_sum[(y0)*(nx+1)+(x1+1)] + utk_sum[(y0)*(nx+1)+(x0)];
    double4_t output_sum = total_sum - input_sum;
    int input_size = (x1_optimal-x0_optimal+1)*(y1_optimal-y0_optimal+1);
    int output_size = nx*ny - input_size;

    Result result = {
        y0_optimal, x0_optimal, y1_optimal+1, x1_optimal+1,
        {(float)output_sum[0]/output_size, (float)output_sum[1]/output_size, (float)output_sum[2]/output_size},
        {(float)input_sum[0]/input_size, (float)input_sum[1]/input_size, (float)input_sum[2]/input_size}
    };
    free(utk_sum);
    free(coordinates_best);
    return result;
}