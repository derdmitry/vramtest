#include <iostream>
#include <vector>
#include <thread>
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

void run_on_device(int device_id, int device_count) {
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::cout << "\nDevice " << device_id << ": " << prop.name << std::endl;
    std::cout << "  Total VRAM: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    size_t totalAllocated = 0;
    std::vector<float*> pointers;

    while (totalAllocated + bytesPerGB <= prop.totalGlobalMem) {
        float* d_data;
        size_t num_elements = bytesPerGB / sizeof(float);

        cudaError_t error = cudaMalloc(&d_data, bytesPerGB);
        if (error != cudaSuccess) {
            std::cerr << "Memory allocation error: " << cudaGetErrorString(error) << std::endl;
            break;
        }

        pointers.push_back(d_data);
        totalAllocated += bytesPerGB;

        std::cout << "Allocated " << totalAllocated / (1024 * 1024) << " MB on device " << device_id << std::endl;

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
        init_random<<<blocksPerGrid, threadsPerBlock>>>(d_data, num_elements);
        cudaDeviceSynchronize();
    }

    for (auto ptr : pointers) {
        cudaFree(ptr);
    }
}

int main() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Found " << device_count << " CUDA devices:" << std::endl;

    std::vector<std::thread> threads;
    for (int i = 0; i < device_count; ++i) {
        threads.push_back(std::thread(run_on_device, i, device_count));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
