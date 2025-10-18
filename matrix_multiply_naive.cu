#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <math.h>
#include <vector>
#include <stdio.h>

__global__ void matrixMultiply(float *A, float *B, float *C, int m, int n, int p) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx < m && colIdx < p) {
        // Where we are writing to within C (flattened index)
        int writeMatrixIdx = rowIdx * p + colIdx;

        // Accumulated dot product
        float dotProduct = 0.0;
        
        // Go through n cols in A and n rows in B to compute dot product
        for (int i = 0; i < n; i++) {
            dotProduct += A[rowIdx * n + i] * B[i * p + colIdx];
        }

        C[writeMatrixIdx] = dotProduct;
    }
}

float *compareKernelAndCUBLAS(int m, int n, int p) {
    // std::cout << "Matrix result size: " << m << "x" << p << std::endl;

    float *hA = new float[m * n];
    float *hB = new float[n * p];
    float *hC = new float[m * p];
    float *hCRef = new float[m * p];

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dist(1.0, 1000.0);

    for (int i = 0; i < m * n; i++) {
        hA[i] = dist(gen);
    }

    
    for (int i = 0; i < n * p; i++) {
        hB[i] = dist(gen);
    }

    float *dA;
    float *dB;
    float *dC;

    cudaMalloc(&dA, m * n * sizeof(float));
    cudaMalloc(&dB, n * p * sizeof(float));
    cudaMalloc(&dC, m * p * sizeof(float));
    
    cudaMemcpy(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * p * sizeof(float), cudaMemcpyHostToDevice);

    int blockSizeRows = 2;
    int blockSizeCols = 2;

    int numBlocksRows = (m + blockSizeRows - 1) / blockSizeRows;
    int numBlocksCols = (p + blockSizeCols - 1) / blockSizeCols;

    dim3 blockSize(blockSizeCols, blockSizeRows);
    dim3 numBlocks(numBlocksCols, numBlocksRows);

    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    matrixMultiply<<<numBlocks, blockSize>>>(dA, dB, dC, m, n, p);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float kernelMilliseconds = 0;
    cudaEventElapsedTime(&kernelMilliseconds, start, end);
    // std::cout << "Kernel time: " << kernelMilliseconds << " ms" << std::endl;  

    cudaMemcpy(hC, dC, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta =  0.0f;

    cudaEventRecord(start);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                p, m, n, &alpha,
                dB, p,
                dA, n,
                &beta,
                dC, p);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float cublasMilliseconds;
    cudaEventElapsedTime(&cublasMilliseconds, start, end);
    // std::cout << "cuBLAS time: " << cublasMilliseconds << " ms" << std::endl;  

    cudaMemcpy(hCRef, dC, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    const float relTol = 1e-5f;
    const float absTol = 1e-3f;

    for (int i = 0; i < m * p; ++i) {
        double diff = fabs((double)hCRef[i] - (double)hC[i]);
        double maxVal = fmax(fabs((double)hCRef[i]), fabs((double)hC[i]));

        if (maxVal <= 0) {
            maxVal = 1.0;
        }

        if (diff > absTol && diff / maxVal > relTol) {
            if (++errors <= 10) {
                std::cout << "Mismatch at " << i << ": "
                        << hC[i] << " (yours) vs. " << hCRef[i] << " (cuBLAS)\n";
            }
        }
    }

    if (errors < 10) {
        std::cout << "Results match!" << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hCRef;

    return new float[2]{kernelMilliseconds, cublasMilliseconds};
}

int main() {
    /* Run the kernel for each problem size (i.e. square matrix size) and see how much time
       the matrix multiplication takes. Then compute GFLOPS at each problem size.
    */

    FILE *fptr;

    fptr = fopen("data/NaiveMatrixMultVsCuBLAS_GFLOPS.csv", "w");

    fprintf(fptr, "Matrix Size, Kernel GFLOPS, cuBLAS GFLOPS\n");

    for (int problemSize = 32; problemSize <= 2048; problemSize *= 2) {
        float *runtimes = compareKernelAndCUBLAS(problemSize, problemSize, problemSize);

        double naiveMilliseconds = runtimes[0];
        double cublasMilliseconds = runtimes[1];

        double kernelGFLOPS = (2.0 * problemSize * problemSize * problemSize) / (naiveMilliseconds / 1000.0) / 1e9;
        double cublasGFLOPS = (2.0 * problemSize * problemSize * problemSize) / (cublasMilliseconds / 1000.0) / 1e9;

        fprintf(fptr, "%d, %.3f, %.3f\n", problemSize, kernelGFLOPS, cublasGFLOPS);

        delete[] runtimes;
    }

    fclose(fptr);

    system("pause");
    return 0;
}