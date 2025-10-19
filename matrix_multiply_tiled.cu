#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>

#define TILE_SIZE 32

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int m, int n, int p) {
    __shared__ float tileA[TILE_SIZE * TILE_SIZE];
    __shared__ float tileB[TILE_SIZE * TILE_SIZE];

    float dotProduct = 0.0;

    for (int phase = 0; phase < (n + TILE_SIZE - 1) / TILE_SIZE; phase++) {
        int rowInA = blockIdx.y * TILE_SIZE + threadIdx.y;
        int colInA = phase * TILE_SIZE + threadIdx.x;

        int rowInB = phase * TILE_SIZE + threadIdx.y;
        int colInB = blockIdx.x * TILE_SIZE + threadIdx.x;

        int rowTile = threadIdx.y;
        int colTile = threadIdx.x;

        if (rowInA < m && colInA < n) {
            tileA[rowTile * TILE_SIZE + colTile] = A[rowInA * n + colInA];
        } else {
            tileA[rowTile * TILE_SIZE + colTile] = 0.0f;
        }

        if (rowInB < n && colInB < p) {
            tileB[rowTile * TILE_SIZE + colTile] = B[rowInB * p + colInB];
        } else {
            tileB[rowTile * TILE_SIZE + colTile] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            int tileARow = threadIdx.y;
            int tileACol = i;

            int tileBRow = i;
            int tileBCol = threadIdx.x;

            dotProduct += tileA[tileARow * TILE_SIZE + tileACol] * tileB[tileBRow * TILE_SIZE + tileBCol];
        }

        __syncthreads();
    }

    int cRow = blockIdx.y * blockDim.y + threadIdx.y;
    int cCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (cRow < m && cCol < p) {
        C[cRow * p + cCol] = dotProduct;
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

    int numBlocksRows = (m + TILE_SIZE - 1) / TILE_SIZE;
    int numBlocksCols = (p + TILE_SIZE - 1) / TILE_SIZE;

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks(numBlocksCols, numBlocksRows);

    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    matrixMultiplyTiled<<<numBlocks, blockSize>>>(dA, dB, dC, m, n, p);

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

    fptr = fopen("data/TiledMatrixMultVsCuBLAS_GFLOPS.csv", "w");

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