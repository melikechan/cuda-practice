#include <stdio.h>
#include <stdlib.h>

/**
 * threadIdx -> 3-dim vector that holds index of thread in block
 * blockIdx -> 3-dim vector that holds index of block in grid
 * blockDim -> 3-dim vector that holds number of threads in block
 */
__global__ void add(int *out, int *a, int *b, int n)
{
    int current_thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (current_thread < n)
    {
        out[current_thread] = a[current_thread] + b[current_thread];
    }
}

int main(void)
{
    int *a, *b, *out;

    // Allocate memory on the CPU
    a = (int *)calloc(1000, sizeof(int));
    b = (int *)calloc(1000, sizeof(int));
    out = (int *)calloc(1000, sizeof(int));

    // Allocate memory on the GPU
    int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, 1000 * sizeof(int));
    cudaMalloc(&d_b, 1000 * sizeof(int));
    cudaMalloc(&d_out, 1000 * sizeof(int));

    // Initialize array
    for (int i = 0; i < 1000; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }

    // Transfer to GPU
    // cudaMemcpy(destination, source, size, direction) (host to device or device to host)
    cudaMemcpy(d_a, a, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 1000 * sizeof(int), cudaMemcpyHostToDevice);

    // dim3 that represents the block and grid dimensions
    dim3 block(256);
    dim3 grid(1000 / block.x + 1);

    // <<< blocks, threads per block >>>
    add<<<grid, block>>>(d_out, d_a, d_b, 1000);

    // Transfer back to CPU
    cudaMemcpy(out, d_out, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1000; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], out[i]);
    }

    // Free memory
    free(a);
    free(b);
    free(out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}