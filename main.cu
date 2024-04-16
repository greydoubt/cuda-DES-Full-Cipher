#include <iostream>
#include <cuda_runtime.h>

// Kernel function to be run on the GPU
__global__ void kernelFunction() {
    // Get the global thread ID
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print the global thread ID
    printf("Hello from thread %d\n", globalThreadId);
}

int main() {
    // Define the size of the grid and block
    int numBlocks = 4;
}
