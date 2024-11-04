#include <stdio.h>

__global__ void hello(int *a, int n, int m)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Distance in y direction -> row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Distance in x direction -> col

    if (row < n && col < m)
    {
        int final_index = row * m + col;
        printf("block: (%d, %d), thread: (%d, %d), index: (%d, %d), value: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col, a[final_index]);
    }
}

int main(void)
{
    /**
     *  Grid of blocks
     *  ------------------- (x, col)
     *  | (0, 0) | (1, 0) |
     *  -------------------
     *  | (0, 1) | (1, 1) |
     *  -------------------
     *  | (0, 2) | (1, 2) |
     *  -------------------
     *  (y, row)
     *
     *  If we look inside each block, the threads are arranged as follows:
     *  ------------------- (x, col)
     *  | (0, 0) | (1, 0) |
     *  -------------------
     *  (y, row)
     *
     *  Hence, if we want to represent the grid of threads, we can do so as follows:
     *  -------------------------------------- (x, col)
     *  | (0, 0) | (1, 0) || (0, 0) | (1, 0) |
     *  --------------------------------------
     *  | (0, 1) | (1, 1) || (0, 1) | (1, 1) |
     *  --------------------------------------
     *  | (0, 2) | (1, 2) || (0, 2) | (1, 2) |
     *  --------------------------------------
     *  (y, row)
     *
     *  Well,
     *  index_x = blockIdx.x * blockDim.x + threadIdx.x;
     *  index_y = blockIdx.y * blockDim.y + threadIdx.y;
     *
     *  What does the index_x and index_y represent?
     *  - They represent the index of the thread in the grid of threads (as a point in the grid)
     *  It is a bit tricky because grid[index_x][index_y] != grid[row][col]
     *  As (index_x, index_y) represents a distance in the x and y direction respectively, we need to convert it to indices in the matrix
     *  - row = index_y
     *  - col = index_x
     */

    // Be aware of the order of the dimensions
    dim3 grid(2, 3, 1);
    dim3 block(2, 1, 1);
    // We now have a ((2 * 3), (2 * 1)) grid of threads

    int n = 3, m = 4;

    int *a = (int *)malloc(n * m * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            a[i * m + j] = i * m + j;
        }
    }

    int *d_a;
    cudaMalloc(&d_a, n * m * sizeof(int));
    cudaMemcpy(d_a, a, n * m * sizeof(int), cudaMemcpyHostToDevice);

    hello<<<grid, block>>>(d_a, n, m);
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    free(a);
    cudaFree(d_a);
    return 0;
}