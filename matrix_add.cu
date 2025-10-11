#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixAdd(float *A, float *B, float *C, int m, int n) {
    int blockNum = (blockIdx.x + blockIdx.y * gridDim.x);
    int firstBlockIdx = blockNum * (blockDim.x * blockDim.y);
    int threadIdxWithinBlock = threadIdx.x + threadIdx.y * blockDim.x;

    int globalIndex = firstBlockIdx + threadIdxWithinBlock;

    if (globalIndex < m * n) {
        C[globalIndex] = A[globalIndex] + B[globalIndex];
    }
}

int main() {
    const int m = 1 << 10;
    const int n = 1 << 5;

    float *hMA = new float[m * n];
    float *hMB = new float[m * n];
    float *hMC = new float[m * n];

    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            hMA[y * n + x] = 1.0; 
        }
    }

    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            hMB[y * n + x] = 2.0; 
        }
    }

    float *dA, *dB, *dC;

    size_t totalMatrixBytes = m * n * sizeof(float);

    cudaMalloc(&dA, totalMatrixBytes);
    cudaMalloc(&dB, totalMatrixBytes);
    cudaMalloc(&dC, totalMatrixBytes);

    cudaMemcpy(dA, hMA, totalMatrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hMB, totalMatrixBytes, cudaMemcpyHostToDevice);

    int blockSizeCols = 16; // 16 threads for 16 columns
    int blockSizeRows = 16; // 16 threads for 16 rows

    int numBlocksInCol = (n + blockSizeCols - 1) / blockSizeCols;
    int numBlocksInRow = (m + blockSizeRows - 1) / blockSizeRows;

    dim3 blockSize(blockSizeCols, blockSizeRows);
    dim3 numBlocks(numBlocksInCol, numBlocksInRow);

    matrixAdd<<<numBlocks, blockSize>>>(dA, dB, dC, m, n);

    cudaMemcpy(hMC, dC, totalMatrixBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        std::cout << hMC[i] << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hMA;
    delete[] hMB;
    delete[] hMC;

    system("PAUSE");
    return 0;
}