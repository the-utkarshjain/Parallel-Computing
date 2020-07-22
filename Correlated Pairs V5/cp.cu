#include "cp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#define CHECK(x) check(x, #x)

using namespace std;

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline int static divup(int a, int b) {
    return (a + b - 1)/b;
}

inline int static roundup(int a, int b) {
    return divup(a, b) * b;
}

__global__ void mykernel(int ny, int nx, int round_ny, int round_nx, const float* data, float* result) {
	
    int i_threadId = threadIdx.x;
    int j_threadId = threadIdx.y;
    int i_BlockId = blockIdx.x;
    int j_BlockId = blockIdx.y;

    if (i_BlockId > j_BlockId)
        return;

    float temp_result[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            temp_result[ib][jb] = 0;
        }
    }

    // for (int ib = 0; ib < 8; ++ib) {
    //     cout<< endl;
    //     for (int jb = 0; jb < 8; ++jb) {
    //         cout<<temp_result[ib][jb];
    //     }
    // }

    float x[8];
    float y[8];
    for(int k = 0; k<nx; ++k)
    {
        for (int ib = 0; ib < 8; ++ib) {
            int i = i_BlockId * 64 + ib * 8 + i_threadId;
            x[ib] = data[round_ny*k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = j_BlockId * 64 + jb * 8 + j_threadId;
            y[jb] = data[round_ny*k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                temp_result[ib][jb] += x[ib] * y[jb];
            }
        }
    }

    // for (int ib = 0; ib < 8; ++ib) {
    //     cout<< endl;
    //     for (int jb = 0; jb < 8; ++jb) {
    //        cout<< temp_result[ib][jb];
    //     }
    // }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = i_BlockId * 64 + ib * 8 + i_threadId;
            int j = j_BlockId * 64 + jb * 8 + j_threadId;
            if (i < ny && j < ny) {
                result[ny*i + j] = temp_result[ib][jb];
            }
        }
    }
}

void correlate(int ny, int nx, const float* data, float* result) {

    int round_ny = roundup(ny, 64);
    int round_nx = roundup(nx, 64);

    // cout<<"Success0";

    float* X = (float*) calloc(round_ny*round_nx ,sizeof(float));
    float* Y = (float*) calloc(round_ny*round_nx ,sizeof(float));
    int elem_size =32;
    float mean[elem_size], norm[elem_size];

    // cout<<"Success1";

    for (int i = 0; i < ny; ++i)
    {
    	for(int j=0; j<elem_size;++j)
        {
            mean[j]=norm[j]=0;
        }

    	for (int j = 0; j+elem_size <= nx; j+=elem_size)
    	{
            for(int k=0; k<elem_size; ++k)
            {
                mean[k] += data[j+k + nx*i];
            }
        }
        
        // cout<<"Success2";
        // for (int j = 0; j+elem_size <= nx; j+=elem_size)
    	// {
        //     cout<< endl;
        //     for(int k=0; k<elem_size; ++k)
        //     {
        //         cout<<mean[k];
        //     }
        // }

        for(int j = nx%elem_size; j>0; --j)
        {
            mean[0] += data[nx-j + nx*i];
        }

        for(int j = 1; j<elem_size; ++j)
        {
            mean[0] += mean[j];
        }
        
    	mean[0] /= nx;

    	for(int j=0; j+elem_size<=nx; j +=elem_size)
    	{
            for(int k=0; k<elem_size; ++k)
            {
                X[j+k + round_nx*i] = data[j+k + nx*i] - mean[0];
                norm[k] += pow(data[j+k + nx*i]-mean[0], 2);
            }
    	}
        // cout<<"Success3";
        for(int j = nx%elem_size; j>0; --j)
        {
            X[nx-j + round_nx*i] = data[nx-j + nx*i] - mean[0];
            norm[0] += pow(data[nx-j + nx*i]-mean[0], 2);
        }

        for(int j =1; j<elem_size; ++j)
            norm[0] += norm[j];

        norm[0] = sqrt(norm[0]);
        
        // cout<<"Success4";
        for(int j=0; j<nx; ++j)
        {
            X[j + round_nx*i] /= norm[0];
        }
    }

    for(int i = 0; i < ny; ++i)
    {
        // cout<<endl;
        for(int j=0; j<nx; ++j)
        {
            Y[round_ny*j+i] = X[round_nx*i+j];
            // cout<<"Y[round_ny*j+i]";
        }
    }

    // Transfering data between CPU and GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, round_ny * round_nx * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, Y, round_ny * round_nx * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8);
    dim3 dimGrid(round_ny/64, round_ny/64);

    mykernel<<<dimGrid, dimBlock>>>(ny, nx, round_ny, round_nx, dGPU, rGPU);

    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

    free(X);
    free(Y);


}
