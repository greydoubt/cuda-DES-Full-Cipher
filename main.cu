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
    int threadsPerBlock = 256;
    
    // Launch the kernel on the GPU
    kernelFunction<<<numBlocks, threadsPerBlock>>>();
    
    // Check for kernel launch errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cudaError) << std::endl;
        return 1;
    }
    
    // Wait for all threads to complete
    cudaDeviceSynchronize();
    
    // Print a completion message
    std::cout << "CUDA kernel executed successfully!" << std::endl;
    
    return 0;
}
