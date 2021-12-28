%% cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>
#define CHECK(call) \
{ \
 const cudaError_t error = call; \
 if (error != cudaSuccess) \
 { \
 printf("Error: %s:%d, ", __FILE__, __LINE__); \
 printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
 exit(1); \
 } \
}


void printMatrix(int* C, const int nx, const int ny)
{
    int* ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int iy = 0;iy < ny;iy++)
    {
        for (int ix = 0;ix < nx;ix++)
        {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}



void saveMatrix(int* C, char* name, const int nx, const int ny)
{
    int* ic = C;
    std::ofstream outdata; // outdata is like cin
    outdata.open(name); // opens the file
    if (!outdata) // file couldn't be opened
    {
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }
    for (int iy = 0;iy < ny;iy++)
    {
        for (int ix = 0;ix < nx;ix++)
        {
            outdata << (int)ic[ix] << " ";
        }
        ic += nx;
        outdata << std::endl;
    }
    outdata << std::endl;
    outdata.close();
}


__device__ int check_condition(float aa, float bb, int max_itr)
{

    float re = 0;
    float im = 0;
    for (int ii = 0;ii < max_itr;ii++)
    {
        float re_o = re;
        float im_o = im;
        re = re_o * re_o - im_o * im_o + aa;
        im = 2 * re_o * im_o + bb;

        if (sqrt(re_o * re_o + im_o * im_o) >= 2)
        {
            return 0;
        }
    }
    return 1;
}
__global__ void MandelbrotOnGPU(int* Matrix, int nx, int ny, float* range_re, float* range_im)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        float aa = range_re[0] + ix * (range_re[1] - range_re[0]) / (nx - 1);
        float bb = range_im[0] + iy * (range_im[1] - range_im[0]) / (ny - 1);
        //float aa=range_re[0]+ix*(2.7)/(nx-1);
        //float bb=range_im[0]+iy*(2.4)/(ny-1);
        Matrix[idx] = check_condition(aa, bb, 200);
        //printf("%d",Matrix[idx]);
    }

}

void generateMandelbrotSet(int nx, int ny)
{
    //std::ofstream outdata; // outdata is like cin
    //outdata.open("timings2.txt",std::ios_base::app); // opens the file

    int ranges = 3 * sizeof(float);
    float* h_range_re, * h_range_im;
    h_range_re = (float*)malloc(ranges);
    h_range_im = (float*)malloc(ranges);

    // initialize data at host side
    h_range_re[0] = -2.1f;
    h_range_re[1] = 0.6f;
    h_range_im[0] = -1.2f;
    h_range_im[1] = 1.2f;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    // malloc host memory
    int* gpuRef;
    gpuRef = (int*)malloc(nBytes);

    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float* d_range_re, * d_range_im;
    int* d_Matrix;
    cudaMalloc((void**)&d_range_re, ranges);
    cudaMalloc((void**)&d_range_im, ranges);
    cudaMalloc((void**)&d_Matrix, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_range_re, h_range_re, ranges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_range_im, h_range_im, ranges, cudaMemcpyHostToDevice);

    // set up execution configuration
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    clock_t begin = clock();
    MandelbrotOnGPU << <grid, block >> > (d_Matrix, nx, ny, d_range_re, d_range_im);
    CHECK(cudaDeviceSynchronize());
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    //std::cout<<"GPU"<<std::endl;
    //std::cout<<time_spent<<std::endl;
    //synchronization is implicite for cudaMemcpy
    CHECK(cudaMemcpy(gpuRef, d_Matrix, nBytes, cudaMemcpyDeviceToHost));
    saveMatrix(gpuRef, "Mandelbrot.txt", nx, ny);
    // free device global memory
    cudaFree(d_range_re);
    cudaFree(d_range_im);
    cudaFree(d_Matrix);

    // free host memory
    free(gpuRef);
    free(h_range_im);
    free(h_range_re);

    // reset device
    cudaDeviceReset();
}

int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx = 16384;
    int ny = 16384;

    generateMandelbrotSet(nx, ny);


    return (0);
}