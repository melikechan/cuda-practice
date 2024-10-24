#include <stdio.h>
#include <stdlib.h>

const long long N = 1e8;

/**
 * threadIdx -> 3-dim vector that holds index of thread in block
 * blockIdx -> 3-dim vector that holds index of block in grid
 * blockDim -> 3-dim vector that holds number of threads in block
 */
__global__ void add(long long *out, long long *a, long long *b, long long n)
{
    long long current_thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (current_thread < n)
    {
        out[current_thread] = a[current_thread] + b[current_thread];
    }
}

__global__ void sub(long long *out, long long *a, long long *b, long long n)
{
    long long current_thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (current_thread < n)
    {
        out[current_thread] = a[current_thread] - b[current_thread];
    }
}

__global__ void dot_prod(long long *out, long long *a, long long *b, long long n)
{
    long long current_thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (current_thread < n)
    {
        out[current_thread] = a[current_thread] * b[current_thread];
    }
}

int main(void)
{
    long long *a, *b, *out;

    size_t size = N * sizeof(long long);
    // Allocate memory on the CPU
    a = (long long *)malloc(size);
    b = (long long *)malloc(size);
    out = (long long *)malloc(size);

    // Allocate memory on the GPU
    long long *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    // Initialize array
    for (long long i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }

    // Transfer to GPU
    // cudaMemcpy(destination, source, size, direction) (host to device or device to host)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // dim3 that represents the block and grid dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // <<< blocks, threads per block >>>
    add<<<grid, block>>>(d_out, d_a, d_b, N); // Example function call

    // Transfer back to CPU
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < N; i++)
    {
        if (fabs(out[i] - (a[i] + b[i])) > 1e-5)
        {
            printf("Error at index %lld: %lld != %lld\n", i, out[i], a[i] + b[i]);
            break;
        }
    }

    // Free memory
    free(a);
    free(b);
    free(out);

    // Free memory on GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}