#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

const int bytesPerGB = 1024 * 1024 * 1024; // 1 GB


__global__ void init_random(float *data, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        data[idx] = curand_uniform(&state) * 100.0f;
    }
}

int main() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Find " << device_count << " CUDA devices:" << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Total VRAM: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

        size_t totalAllocated = 0;
        std::vector<float*> pointers;

        while (totalAllocated + bytesPerGB <= prop.totalGlobalMem) {
            float* d_data;
            size_t num_elements = bytesPerGB / sizeof(float);

            error = cudaMalloc(&d_data, bytesPerGB);
            if (error != cudaSuccess) {
                std::cerr << "Memory allocation error: " << cudaGetErrorString(error) << std::endl;
                break; // 
            }

            pointers.push_back(d_data);
            totalAllocated += bytesPerGB;

            std::cout << "Allocated " << totalAllocated / (1024 * 1024) << " MB" << std::endl;

            // 
            int threadsPerBlock = 256;
            int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
            init_random<<<blocksPerGrid, threadsPerBlock>>>(d_data, num_elements);
            cudaDeviceSynchronize();
        }

        // 
        for (auto ptr : pointers) {
            cudaFree(ptr);
        }
    }

    return 0;
}
