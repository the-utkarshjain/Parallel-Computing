#include "cp.h"
#include <math.h>
#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#define BLOCK_SIZE 8

//#define GPU_NORMALISATION 1

#define CHECK_CUDA_ERROR(call) \
        do { \
          cudaError_t result_ = (call); \
          if (result_ != cudaSuccess) \
          { \
            fprintf(stderr, #call " failed: %s\n", \
                    cudaGetErrorString(result_)); \
            exit(1); \
          } \
        } while(0)
        
__global__ void var(float *input,float *output, int N, float mean)
{
  int idx=threadIdx.x+(blockDim.x*blockIdx.x);
  if (idx < N) output[idx] = (input[idx]-mean)*(input[idx]-mean);
}

__global__ void norm(float *input, int N,float mean,float stddev)
{
  int idx=threadIdx.x+(blockDim.x*blockIdx.x);
  if (idx < N) input[idx] =  (input[idx]-mean)/stddev;
}
 
__global__ void matrixMul( float* C, float* A, int ny,int nx)
{
   int tx = threadIdx.x + (blockDim.x * blockIdx.x);
   int ty = threadIdx.y + (blockDim.y * blockIdx.y);
  
   if(tx>= ny || ty>=ny)
    return;
   float value = 0;
   if(tx<ty)
    return;
   for (int i = 0; i < nx; ++i)
   {
      float row1 = A[ty * nx + i];
      float row2 = A[tx * nx + i];
      value += row1 * row2;
   }
    C[ty * ny + tx] = value;
}
 
void correlate(int ny, int nx, const float* data, float* result) 
{
  float* norm_data = (float*)malloc(ny*nx*sizeof(float));

  #ifdef GPU_NORMALISATION
  float *cuda_input;
  float *cuda_output; 
  float *temp_row= (float*)malloc(nx*sizeof(float));
  size_t row_size = nx * sizeof(float);
  CHECK_CUDA_ERROR(cudaMalloc((void **) &cuda_input, row_size)); 
  CHECK_CUDA_ERROR(cudaMalloc((void **) &cuda_output, row_size)); 
  //normalise the matrix
  for (int y = 0; y < ny; ++y) 
  {
    float mean = 0.0;
    float stddev = 0.0;

    //Finding the mean
    for (int x = 0; x < nx; ++x) 
    {
      norm_data[x + y*nx] = data[x + y*nx];
      mean += norm_data[x + y*nx];
    }
    mean= mean/nx;
    CHECK_CUDA_ERROR(cudaMemcpy(cuda_input, &norm_data[y*nx], row_size, 
          cudaMemcpyHostToDevice));
    int block_size = 10;
    int n_blocks = nx/block_size + (nx%block_size == 0 ? 0:1);
    var<<< n_blocks, block_size >>> (cuda_input,cuda_output,nx,mean);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(temp_row, cuda_output, row_size,
          cudaMemcpyDeviceToHost));
    for (int x= 0; x< nx; ++x) 
      stddev += temp_row[x];

    stddev= sqrt(stddev);
    CHECK_CUDA_ERROR(cudaMemcpy(cuda_output, &norm_data[y*nx], row_size, 
          cudaMemcpyHostToDevice));
    norm<<< n_blocks, block_size >>> (cuda_output,nx,mean,stddev);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(&norm_data[y*nx], cuda_output, row_size, 
          cudaMemcpyDeviceToHost));
  }
  free(temp_row);
  cudaFree(cuda_input);
  cudaFree(cuda_output);
  #endif

  #ifndef GPU_NORMALISATION
  for(int i=0; i<ny; i++)
  {
    int s= i*nx;
    float mean=0.0f;
    for(int i=0; i<nx; i++)
    {
      mean+= data[s+i];
    }
    mean/=nx;
    float var=0.0f;
    float tmp= 0.0f;
    for(int i=0; i<nx; i++)
    {   
      tmp= data[s+i]-mean;
      norm_data[s+i]= tmp;
      var+= tmp*tmp;
    }   
    var=std::sqrt(var);
    for(int i=0; i<nx; i++)
      norm_data[s+i]/=var;
  }
  #endif

  //matrix multiplication
  float *send_data;
  float *result_data;
  int size = nx*ny*sizeof(float);
  CHECK_CUDA_ERROR(cudaMalloc((void**) &send_data, size));
  CHECK_CUDA_ERROR(cudaMalloc((void**) &result_data, ny*ny*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(send_data, norm_data, size,cudaMemcpyHostToDevice));
  dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
  int n_blocks = ny/BLOCK_SIZE + (ny%BLOCK_SIZE == 0 ? 0:1);
  dim3 grid(n_blocks,n_blocks);

  matrixMul<<< grid, threads >>>(result_data, send_data, ny,nx);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaMemcpy(result, result_data, ny*ny*sizeof(float), 
        cudaMemcpyDeviceToHost));

  free(norm_data);
  cudaFree(send_data);
  cudaFree(result_data);
}