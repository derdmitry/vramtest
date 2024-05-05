apt install nvidia-cuda-toolkit -y
nvcc -o cuda_vram_test cuda_vram_test.cu
./cuda_vram_test
