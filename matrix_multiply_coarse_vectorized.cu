#include <assert.h>

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

// Coalescing Factor
#define COARSE_FACTOR_X 8
#define COARSE_FACTOR_Y 8

// Tiles of A
#define TILES_A_ROWS 128
#define TILES_A_COLS 16

// Tiles of B
#define TILES_B_COLS 128

__global__ void matrixMultiplyCoarse2dVectorized(float *A, float *B, float *C, int m, int n, int p)
{
    int rowsC = m;
    int colsA = n;
    int colsC = p;

    // Number of threads per block
    const int n_threads_per_block = TILES_A_ROWS * TILES_B_COLS / (COARSE_FACTOR_X*COARSE_FACTOR_Y);
    static_assert(n_threads_per_block % TILES_A_COLS == 0, "");
    static_assert(n_threads_per_block % TILES_B_COLS == 0, "");
    static_assert(TILES_A_COLS % 4 == 0, "TILES_A_COLS % 4 == 0");
    static_assert(TILES_B_COLS % 4 == 0, "TILES_B_COLS % 4 == 0");
    assert(rowsC % 4 == 0);
    assert(colsC % 4 == 0);
    assert(colsA % 4 == 0);

    // Details regarding this thread
    const int by = blockIdx.y;
    const int bx = blockIdx.x; 

    const int tx = threadIdx.x;

    // 1D -> 2D while loading A
    const int A_view_ty = tx / (TILES_A_COLS / 4);
    const int A_view_tx = tx % (TILES_A_COLS / 4);
    const int stride_A = n_threads_per_block/(TILES_A_COLS/4);

    // 1D -> 2D while loading B
    const int B_view_ty = tx / (TILES_B_COLS / 4);
    const int B_view_tx = tx % (TILES_B_COLS / 4);
    const int stride_B = n_threads_per_block/(TILES_B_COLS / 4);

    // Working on C[row, col]
    const int row = COARSE_FACTOR_Y * (tx / (TILES_B_COLS/COARSE_FACTOR_X));
    const int col = COARSE_FACTOR_X * (tx % (TILES_B_COLS/COARSE_FACTOR_X));
    
    // Allocating shared memory
    __shared__ float sh_A[TILES_A_COLS][TILES_A_ROWS];
    __shared__ float sh_B[TILES_A_COLS][TILES_B_COLS];

    // Parallel mat mul
    float value[COARSE_FACTOR_Y][COARSE_FACTOR_X] = {0.0f};
    float register_A[COARSE_FACTOR_X] = {0.0f};
    float register_B[COARSE_FACTOR_Y] = {0.0f};

    // Phases
    const int phases = ceil((float)colsA/TILES_A_COLS);

    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        for (int load_offset = 0; load_offset < TILES_A_ROWS; load_offset+=stride_A)
        {
            if ((by*TILES_A_ROWS + load_offset+A_view_ty < rowsC) && (((phase*TILES_A_COLS+A_view_tx*4)) < colsA))
            {
                float4 A_tmp = reinterpret_cast<float4 *>(&A[(by*TILES_A_ROWS + load_offset+A_view_ty)*colsA + ((phase*TILES_A_COLS+A_view_tx*4))])[0];
                sh_A[A_view_tx*4+0][load_offset+A_view_ty] = A_tmp.x;
                sh_A[A_view_tx*4+1][load_offset+A_view_ty] = A_tmp.y;
                sh_A[A_view_tx*4+2][load_offset+A_view_ty] = A_tmp.z;
                sh_A[A_view_tx*4+3][load_offset+A_view_ty] = A_tmp.w;
            }
            else
            {
                sh_A[A_view_tx*4+0][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+1][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+2][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+3][load_offset+A_view_ty] = 0.0f;
            }
            
        }
        
        for (int load_offset = 0; load_offset < TILES_A_COLS; load_offset+=stride_B)
        {
            if (((phase*TILES_A_COLS + B_view_ty+load_offset) < colsA) && (((bx*TILES_B_COLS + B_view_tx*4)) < colsC))
            {
                float4 B_tmp = reinterpret_cast<float4 *>(&B[(phase*TILES_A_COLS + B_view_ty+load_offset)*colsC + ((bx*TILES_B_COLS + B_view_tx*4))])[0];
                sh_B[B_view_ty+load_offset][B_view_tx*4+0] = B_tmp.x;
                sh_B[B_view_ty+load_offset][B_view_tx*4+1] = B_tmp.y;
                sh_B[B_view_ty+load_offset][B_view_tx*4+2] = B_tmp.z;
                sh_B[B_view_ty+load_offset][B_view_tx*4+3] = B_tmp.w;
            }
            else
            {
                sh_B[B_view_ty+load_offset][B_view_tx*4+0] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+1] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+2] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+3] = 0.0f;
            }
            
        }
        __syncthreads();

        // calculate per-thread results
        for (int k = 0; k < TILES_A_COLS; ++k) 
        {
            // block into registers
            for (int i = 0; i < COARSE_FACTOR_Y; ++i)
                register_A[i] = sh_A[k][row+i];
            
            for (int i = 0; i < COARSE_FACTOR_X; ++i)
                register_B[i] = sh_B[k][col+i];
            
            for (int cy = 0; cy < COARSE_FACTOR_Y; ++cy) 
            {
                for (int cx = 0; cx < COARSE_FACTOR_X; ++cx) 
                    value[cy][cx] += register_A[cy] * register_B[cx];
            }
        }
        __syncthreads();
    }

    // Assigning calculated value
    for (int cy = 0; cy < COARSE_FACTOR_Y; ++cy)
    {
        for (int cx = 0; cx < COARSE_FACTOR_X; cx++)
        {
            if ((by*TILES_A_ROWS+row+cy < rowsC) && (bx*TILES_B_COLS+col+cx < colsC))
                C[(by*TILES_A_ROWS+row+cy)*colsC + (bx*TILES_B_COLS+col+cx)] = 1*value[cy][cx] + 0*C[(by*TILES_A_ROWS+row+cy)*colsC + (bx*TILES_B_COLS+col+cx)];
        }
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

    int numBlocksRows = (m + TILES_A_ROWS - 1) / TILES_A_ROWS;
    int numBlocksCols = (p + TILES_B_COLS - 1) / TILES_B_COLS;

    dim3 blockSize(TILES_A_ROWS*TILES_B_COLS/(COARSE_FACTOR_X*COARSE_FACTOR_Y));
    dim3 numBlocks(numBlocksCols, numBlocksRows);

    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    matrixMultiplyCoarse2dVectorized<<<numBlocks, blockSize>>>(dA, dB, dC, m, n, p);

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

    fptr = fopen("data/VectorizedMatrixMultVsCuBLAS_GFLOPS.csv", "w");

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