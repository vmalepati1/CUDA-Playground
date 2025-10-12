#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixAdd(float *A, float *B, float *C, int m, int n) {
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

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

    int blockSize = 256; // 256 threads
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixAdd<<<numBlocks, blockSize>>>(dA, dB, dC, m, n);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;

    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

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