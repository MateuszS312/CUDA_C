
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>

namespace cg = cooperative_groups;

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
void initializeData(int *ip, int size)
{
    //generate different seeds for random N
    time_t tt;
    srand((unsigned int) time(&tt));
    for(int ii=0;ii<size;ii++)
    {
        ip[ii]=rand()%2;
    }
}

void printMatrix(int *C, const int nx, const int ny)
{
  int *ic=C;
  printf("\nMatrix: (%d.%d)\n",nx,ny);
  for(int iy=0;iy<ny;iy++)
  {
    for(int ix=0;ix<nx;ix++)
    {   
        if(ic[ix]==1)
          printf("%d",0);
        else
          printf(" ");
    }
    ic+=nx;
    printf("\n");
  }
  printf("\n");
}
    
 
    
void saveMatrix(int *C, char* name, const int nx, const int ny)
{
  int *ic=C;
  std::ofstream outdata; // outdata is like cin
  outdata.open(name); // opens the file
  if( !outdata ) // file couldn't be opened
  { 
    std::cerr << "Error: file could not be opened" << std::endl;
      exit(1);
  }
  for(int iy=0;iy<ny;iy++)
  {
    for(int ix=0;ix<nx;ix++)
    {
        outdata << (int) ic[ix] << " ";
    }
    ic+=nx;
    outdata <<std::endl;
  }
  outdata <<std::endl;
  outdata.close();
}


__device__ int return_element(int *Matrix, int ix, int iy, int nx, int ny)
{
    //Periodic boundry conditions
    iy=(ny+iy)%ny;
    ix=(nx+ix)%nx;
    unsigned int idx = iy*nx+ix;
    return Matrix[idx];
}
__global__ void computeGOL(int *Matrix, int* neighMatrix, int nx, int ny)
{
  unsigned int ix= threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int iy = threadIdx.y+blockIdx.y*blockDim.y;
  if(ix<nx && iy<ny)
  {
      int sum= return_element(neighMatrix,ix,iy,nx,ny);
   
      if(return_element(Matrix, ix, iy, nx ,ny))
      {
          if(sum<2 || sum>3)
          {
              Matrix[iy*nx+ix]=0;
          }
      }
      else
      {
          if(sum==3)
          {
              Matrix[iy*nx+ix]=1;
          }
      }

  }
    
}
__global__ void computeNeighbours(int *Matrix,int* neighMatrix, int nx, int ny)
{
  unsigned int ix= threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int iy = threadIdx.y+blockIdx.y*blockDim.y;
  if(ix<nx && iy<ny)
  {
      int sum=0;
      for(int ii=-1;ii<2;ii++)
      {
          for(int jj=-1;jj<2;jj++)
          {
              if(ii!=0 || jj!=0)
              {
                  sum+=return_element(Matrix, ix+ii, iy+jj, nx ,ny);
              }
                
          }
      }
      neighMatrix[iy*nx+ix]=sum;
  }
    
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
  int nx = 200;
  int ny = 200;  

  int nxy = nx*ny;
  int nBytes = nxy * sizeof(float);
  // malloc host memory
  int *gpuRef;
  gpuRef = (int *)malloc(nBytes);
  initializeData(gpuRef,nxy);

 // set up execution configuration
  int dimx = 32;
  int dimy = 32;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  int NN=100000;
  // malloc device global memory
 
  int *board_Matrix;
  int *neigh_Matrix;
  clock_t begin = clock();
  cudaMalloc((void **)&board_Matrix, nBytes);
  cudaMalloc((void **)&neigh_Matrix, nBytes);
  // transfer data from host to device
  cudaMemcpy(board_Matrix, gpuRef, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(neigh_Matrix, gpuRef, nBytes, cudaMemcpyHostToDevice);
 
  //printMatrix(gpuRef,nx,ny);
 
  
  for(int ii=0; ii<NN;ii++)
  {
    computeNeighbours<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
   
    computeGOL<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
    //synchronization always occurs between kernel lunches
    //CHECK(cudaDeviceSynchronize());
  }

  //synchronization is implicite for cudaMemcpy
  CHECK(cudaMemcpy(gpuRef, board_Matrix, nBytes, cudaMemcpyDeviceToHost));
  saveMatrix(gpuRef, "Mandelbrot.txt", nx, ny);
  //printMatrix(gpuRef,nx,ny);

  // free device global memory
  cudaFree(board_Matrix);
  cudaFree(neigh_Matrix);
  // free host memory
  free(gpuRef);
 
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout<<"GPU1"<<std::endl;
  std::cout<<time_spent<<std::endl;
  
  //Pinned memory
  begin = clock();
  cudaMallocHost(&gpuRef, nBytes);
  
  initializeData(gpuRef,nxy);

  // malloc device global memory
 
  cudaMalloc((void **)&board_Matrix, nBytes);
  cudaMalloc((void **)&neigh_Matrix, nBytes);
  // transfer data from host to device
  cudaMemcpy(board_Matrix, gpuRef, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(neigh_Matrix, gpuRef, nBytes, cudaMemcpyHostToDevice);

  //printMatrix(gpuRef,nx,ny);
 
  for(int ii=0; ii<NN;ii++)
  {
    computeNeighbours<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
    computeGOL<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
  }
  //synchronization is implicite for cudaMemcpy
  CHECK(cudaMemcpy(gpuRef, board_Matrix, nBytes, cudaMemcpyDeviceToHost));
  saveMatrix(gpuRef, "Mandelbrot2.txt", nx, ny);
  //printMatrix(gpuRef,nx,ny);
  // free device global memory
  cudaFree(board_Matrix);
  cudaFree(neigh_Matrix);
  // free host memory
  cudaFreeHost(gpuRef);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout<<"GPU2"<<std::endl;
  std::cout<<time_spent<<std::endl;
 
   //Mapped memory
  begin = clock();
  cudaHostAlloc(&board_Matrix, nBytes, cudaHostAllocMapped);
  cudaHostAlloc(&neigh_Matrix, nBytes, cudaHostAllocMapped);
  initializeData(board_Matrix,nxy);


  //printMatrix(gpuRef,nx,ny);
 
  for(int ii=0; ii<NN;ii++)
  {
    computeNeighbours<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
    computeGOL<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
  }

  CHECK(cudaDeviceSynchronize());
  saveMatrix(board_Matrix, "Mandelbrot3.txt", nx, ny);
  //printMatrix(gpuRef,nx,ny);

  // free host memory
  cudaFreeHost(board_Matrix);
  cudaFreeHost(neigh_Matrix);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout<<"GPU3"<<std::endl;
  std::cout<<time_spent<<std::endl;
 
    //Managed memory
  begin = clock();
  cudaMallocManaged(&board_Matrix, nBytes);
  cudaMallocManaged(&neigh_Matrix, nBytes);
  initializeData(board_Matrix,nxy);
  

  for(int ii=0; ii<NN;ii++)
  {
    computeNeighbours<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
    computeGOL<<<grid,block>>>(board_Matrix,neigh_Matrix, nx, ny);
  }

  CHECK(cudaDeviceSynchronize());
  saveMatrix(board_Matrix, "Mandelbrot4.txt", nx, ny);
  // free host memory
  cudaFree(board_Matrix);
  cudaFree(neigh_Matrix);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout<<"GPU4"<<std::endl;
  std::cout<<time_spent<<std::endl;


  // reset device
  cudaDeviceReset();


  return (0);
}