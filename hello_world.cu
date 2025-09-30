#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello, world from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // 2 blocks, 4 threads each
    helloFromGPU<<<2, 4>>>();

    cudaDeviceSynchronize();

    system("pause");
    return 0;
}