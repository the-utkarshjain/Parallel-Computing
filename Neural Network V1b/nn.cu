#include <cstdio>
#include <cmath>
#include "nn.h"
#include <cuda_runtime.h>
#include <iostream>

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

#define MAX(a, b) ((a) > (b) ? (a) : (b))


// ------------------------------------------------------------------------

float* g_weights = NULL;    // store all network weights in one big array.

// ------------------------------------------------------------------------

ConvLayer g_convLayers[16] = {
    { 224,  64,   3,        0,     1728 },
    { 224,  64,  64,     1792,    38656 },    // 2x2 maxpool (224 x 224 -> 112 x 112)
    { 112, 128,  64,    38720,   112448 },
    { 112, 128, 128,   112576,   260032 },    // 2x2 maxpool (112 x 112 -> 56 x 56)
    {  56, 256, 128,   260160,   555072 },
    {  56, 256, 256,   555328,  1145152 },
    {  56, 256, 256,  1145408,  1735232 },
    {  56, 256, 256,  1735488,  2325312 },    // 2x2 maxpool (56 x 56 -> 28 x 28)
    {  28, 512, 256,  2325568,  3505216 },
    {  28, 512, 512,  3505728,  5865024 },
    {  28, 512, 512,  5865536,  8224832 },
    {  28, 512, 512,  8225344, 10584640 },    // 2x2 maxpool (28 x 28 -> 14 x 14)
    {  14, 512, 512, 10585152, 12944448 },
    {  14, 512, 512, 12944960, 15304256 },
    {  14, 512, 512, 15304768, 17664064 },
    {  14, 512, 512, 17664576, 20023872 },    // 2x2 maxpool (14 x 14 -> 7 x 7) -> interpret as flat array
};

DenseLayer g_denseLayers[3] = {
    { 4096, 25088,  20024384, 122784832, false },
    { 4096,  4096, 122788928, 139566144, false },
    { 1000,  4096, 139570240, 143666240, true  },
};

// ------------------------------------------------------------------------

__global__ void convKernel(int idx, const float* bufIn, float* bufOut, int n_threads, float* g_weights, ConvLayer* g_convLayers) {
    const ConvLayer& layer = g_convLayers[idx];

    const float* W = g_weights + layer.ofsW;
    const float* B = g_weights + layer.ofsB;


    int sz = layer.sz;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int x = threadIdx.x;


        float sum = B[blockIdx_x];
        for (int j = 0; j < layer.nIn; j++)
        for (int dy = 0; dy < 3; dy++)
        for (int dx = 0; dx < 3; dx++)
        {
            int yy = blockIdx_y + dy - 1;
            int xx = x + dx - 1;
            if (yy >= 0 && yy < sz && xx >= 0 && xx < sz)
                sum += bufIn[sz*sz*j + sz*yy + xx] * W[layer.nIn*3*3*blockIdx_x + 3*3*j + 3*(2-dy) + (2-dx)];
        }
        bufOut[sz*sz*blockIdx_x + sz*blockIdx_y + x] = (sum > 0.f) ? sum : 0.f; // ReLu activation.
}


__global__ void denseKernel1(int idx, const float* bufIn, float* bufOut, int n_threads, float* g_weights, DenseLayer* g_denseLayers, float* total) {

    const DenseLayer& layer = g_denseLayers[idx];
    const float* W = g_weights + layer.ofsW;
    const float* B = g_weights + layer.ofsB;

    int i = blockIdx.x * n_threads + threadIdx.x;

    if(i>=layer.nOut)
        return;

    float sum = B[i];
    for (int j = 0; j < layer.nIn; j++)
        sum += bufIn[j] * W[layer.nIn*i + j];

    if (layer.softmax)
    {
        atomicAdd(total, (bufOut[i] = expf(sum)));
    }
    else
        bufOut[i] = (sum > 0.f) ? sum : 0.f;

}

__global__ void denseKernel2(int idx, float* bufOut, int n_threads, float* g_weights, DenseLayer* g_denseLayers, float *total) {

    int i = blockIdx.x * n_threads + threadIdx.x;
    const DenseLayer& layer = g_denseLayers[idx];

    if(i>=layer.nOut)
        return;

    bufOut[i] *= 1.f / total[0];
}

__global__ void maxPoolkernel(int sz, int n, const float* bufIn, float* bufOut, int n_threads)
{
    int h = sz >> 1;

    int y = blockIdx.x;
    int i = threadIdx.x;

    for (int x = 0; x < h; x++)
    {
        float v0 = bufIn[sz*sz*i + sz*(y*2)   + (x*2)];
        float v1 = bufIn[sz*sz*i + sz*(y*2)   + (x*2+1)];
        float v2 = bufIn[sz*sz*i + sz*(y*2+1) + (x*2)];
        float v3 = bufIn[sz*sz*i + sz*(y*2+1) + (x*2+1)];
        bufOut[i*h*h + x + h*y] = MAX(MAX(MAX(v0, v1), v2), v3);
    }
}

void evalNetwork(float *buf0) {

    int n_threads = 4;
    int n_threads_max = 8;

    float* buf1 = new float[64 * 224 * 224];

    int size_d = 64 * 224 * 224;
    int size_w = 143667240;

    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, size_d * sizeof(float)));

    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, size_d * sizeof(float)));

    CHECK(cudaMemcpy(dGPU, buf0, size_d * sizeof(float), cudaMemcpyHostToDevice));

    float* wGPU = NULL;
    CHECK(cudaMalloc((void**)&wGPU, size_w * sizeof(float)));
    CHECK(cudaMemcpy(wGPU, g_weights, size_w * sizeof(float), cudaMemcpyHostToDevice));

    ConvLayer* gGPU = NULL;
    CHECK(cudaMalloc((void**)&gGPU, 16 * sizeof(ConvLayer)));
    CHECK(cudaMemcpy(gGPU, g_convLayers, 16 * sizeof(ConvLayer), cudaMemcpyHostToDevice));


    dim3 blocks(g_convLayers[0].nOut, g_convLayers[0].sz);
    convKernel<<<blocks, g_convLayers[0].sz>>>(0, dGPU, rGPU, n_threads, wGPU, gGPU);
    CHECK(cudaGetLastError());

    blocks = dim3(g_convLayers[1].nOut, g_convLayers[1].sz);
    convKernel<<<blocks, g_convLayers[1].sz>>>(1, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = 112;
    maxPoolkernel<<<blocks, 64>>>(224, 64, dGPU, rGPU, n_threads_max);

    blocks = dim3(g_convLayers[2].nOut, g_convLayers[2].sz);
    convKernel<<<blocks, g_convLayers[2].sz>>>(2, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[3].nOut, g_convLayers[3].sz);
    convKernel<<<blocks, g_convLayers[3].sz>>>(3, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = 64;
    maxPoolkernel<<<blocks, 128>>>(112, 128, rGPU, dGPU, n_threads_max);

    blocks = dim3(g_convLayers[4].nOut, g_convLayers[4].sz);
    convKernel<<<blocks, g_convLayers[4].sz>>>(4, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[5].nOut, g_convLayers[5].sz);
    convKernel<<<blocks, g_convLayers[5].sz>>>(5, rGPU, dGPU, n_threads, wGPU, gGPU);
    
    blocks = dim3(g_convLayers[6].nOut, g_convLayers[6].sz);
    convKernel<<<blocks, g_convLayers[6].sz>>>(6, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[7].nOut, g_convLayers[7].sz);
    convKernel<<<blocks, g_convLayers[7].sz>>>(7, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = 28;
    maxPoolkernel<<<blocks, 256>>>(56, 256, dGPU, rGPU, n_threads_max);

    blocks = dim3(g_convLayers[8].nOut, g_convLayers[8].sz);
    convKernel<<<blocks, g_convLayers[8].sz>>>(8, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[9].nOut, g_convLayers[9].sz);
    convKernel<<<blocks, g_convLayers[9].sz>>>(9, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[10].nOut, g_convLayers[10].sz);
    convKernel<<<blocks, g_convLayers[10].sz>>>(10, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[11].nOut, g_convLayers[11].sz);
    convKernel<<<blocks, g_convLayers[11].sz>>>(11, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = 14;
    maxPoolkernel<<<blocks, 512>>>(28, 512, rGPU, dGPU, n_threads_max);

    blocks = dim3(g_convLayers[12].nOut, g_convLayers[12].sz);
    convKernel<<<blocks, g_convLayers[12].sz>>>(12, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[13].nOut, g_convLayers[13].sz);
    convKernel<<<blocks, g_convLayers[13].sz>>>(13, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[14].nOut, g_convLayers[14].sz);
    convKernel<<<blocks, g_convLayers[14].sz>>>(14, dGPU, rGPU, n_threads, wGPU, gGPU);

    blocks = dim3(g_convLayers[15].nOut, g_convLayers[15].sz);
    convKernel<<<blocks, g_convLayers[15].sz>>>(15, rGPU, dGPU, n_threads, wGPU, gGPU);

    blocks = 7;
    maxPoolkernel<<<blocks, 512>>>(14, 512, dGPU, rGPU, n_threads_max);

    float* tGPU = NULL;
    float zero[1] = {0};
    CHECK(cudaMalloc((void**)&tGPU, 1 * sizeof(float)));
    CHECK(cudaMemcpy(tGPU, zero, 1 * sizeof(float), cudaMemcpyHostToDevice));

    DenseLayer* gdGPU = NULL;
    CHECK(cudaMalloc((void**)&gdGPU, 3 * sizeof(DenseLayer)));
    CHECK(cudaMemcpy(gdGPU, g_denseLayers, 3 * sizeof(DenseLayer), cudaMemcpyHostToDevice));

    int threads = 64;

    denseKernel1<<<divup(g_denseLayers[0].nOut, threads), threads>>>(0, rGPU, dGPU, threads, wGPU, gdGPU, tGPU);
    if(g_denseLayers[0].softmax)
        denseKernel2<<<divup(g_denseLayers[0].nOut, threads), threads>>>(0, dGPU, threads, wGPU, gdGPU, tGPU);

    denseKernel1<<<divup(g_denseLayers[1].nOut, threads), threads>>>(1, dGPU, rGPU, threads, wGPU, gdGPU, tGPU);
    if(g_denseLayers[1].softmax)
        denseKernel2<<<divup(g_denseLayers[1].nOut, threads), threads>>>(1, rGPU, threads, wGPU, gdGPU, tGPU);

    denseKernel1<<<divup(g_denseLayers[2].nOut, threads), threads>>>(2, rGPU, dGPU, threads, wGPU, gdGPU, tGPU);
    if(g_denseLayers[2].softmax)
        denseKernel2<<<divup(g_denseLayers[2].nOut, threads), threads>>>(2, dGPU, threads, wGPU, gdGPU, tGPU);

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(buf0, dGPU, size_d * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(buf1, rGPU, size_d * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    CHECK(cudaFree(wGPU));
    CHECK(cudaFree(gGPU));
    CHECK(cudaFree(tGPU));
    CHECK(cudaFree(gdGPU));

    

    delete[] buf1;
}


