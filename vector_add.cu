#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 1 << 20; // ~1M elements
    size_t bytes = n * sizeof(float);

    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = -2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    std::cout << "Num blocks: " << numBlocks << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;

    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        std::cout << h_C[i] << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    system("pause");
    return 0;
}