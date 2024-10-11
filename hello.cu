#include <stdio.h>

__global__ void hello() // __global__ is a CUDA keyword that indicates a function that runs on the GPU
{
    printf("HELLLLLLLLLLLLLLLLLLLLP!\n");
}

int main(void)
{
    // <<< blocks, threads per block >>>
    hello<<<1, 1>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}