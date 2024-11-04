#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const long long N = 1e4, M = 1e4;

/*
    In CUDA, N-dimensional arrays are accessed in row-major order.
    For example, a 2D array of size N x M is accessed as a[i * M + j].
*/
__global__ void add(long long *out, long long *a, long long *b, long long n, long long m)
{
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
    long long j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < n && j < m)
    {
        out[i * m + j] = a[i * m + j] + b[i * m + j];
    }
}

__global__ void sub(long long *out, long long *a, long long *b, long long n, long long m)
{
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
    long long j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < n && j < m)
    {
        out[i * m + j] = a[i * m + j] - b[i * m + j];
    }
}

__global__ void dot_prod(long long *out, long long *a, long long *b, long long n, long long m)
{
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
    long long j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < n && j < m)
    {
        out[i * m + j] = a[i * m + j] * b[i * m + j];
    }
}

__global__ void transpose(long long *out, long long *a, long long n, long long m)
{
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
    long long j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < n && j < m)
    {
        out[j * n + i] = a[i * m + j];
    }
}

__global__ void matrix_mult(long long *out, long long *a, long long *b, long long n, long long m, long long p)
{
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
    long long j = threadIdx.y + blockIdx.y * blockDim.y;

    // transpose b
    long long *b_t = (long long *)malloc(m * p * sizeof(long long));

    for (long long i = 0; i < m; i++)
    {
        for (long long j = 0; j < p; j++)
        {
            b_t[j * m + i] = b[i * p + j];
        }
    }

    if (i < n && j < p)
    {
        long long sum = 0;
        for (long long k = 0; k < m; k++)
        {
            sum += a[i * m + k] * b_t[j * m + k];
        }
        out[i * p + j] = sum;
    }

    free(b_t);
}

int main(void)
{
    long long *a, *b, *out;

    size_t size = N * M * sizeof(long long);

    // Allocate memory on the CPU
    a = (long long *)malloc(size);
    b = (long long *)malloc(size);
    out = (long long *)malloc(size);

    for (long long i = 0; i < N; i++)
    {
        for (long long j = 0; j < M; j++)
        {
            a[i * M + j] = i;
            b[i * M + j] = j;
        }
    }

    long long *d_a, *d_b, *d_out;

    // Allocate memory on the GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    // Transfer to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // dim3 that represents the block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // <<< blocks, threads per block >>>
    add<<<grid, block>>>(d_out, d_a, d_b, N, M);

    // Transfer back to CPU
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < N; i++)
    {
        for (long long j = 0; j < M; j++)
        {
            if (fabs(out[i * M + j] - (a[i * M + j] + b[i * M + j])) > 1e-5)
            {
                printf("Error at index (%lld, %lld): %lld != %lld\n", i, j, out[i * M + j], a[i * M + j] + b[i * M + j]);
                break;
            }
        }
    }

    // Free memory on GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Free memory on CPU
    free(a);
    free(b);
    free(out);

    return 0;
}