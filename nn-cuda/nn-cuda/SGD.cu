#include "common.hpp"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#define TEST_PYTORTH true

void SGD_cpu(float* params_memory, const float* grads_memory, long num_parameters, float learning_rate=1e-3, float weight_decay=0.0) {
    for (int i = 0; i < num_parameters; i++) {
        params_memory[i] -= learning_rate * (grads_memory[i] + weight_decay * params_memory[i]);
    }
}

__global__ void SGD_kernel(float* params_memory, const float* grads_memory, long num_parameters, float learning_rate, float weight_decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_parameters) {
        params_memory[i] -= learning_rate * (grads_memory[i] + weight_decay * params_memory[i]);
    }
}

void SGD_run_kernel(float* params_memory, const float* grads_memory, long num_parameters, float learning_rate=1e-3, float weight_decay=0.0, int block_size=16) {
	int num_blocks = ceil_div<float>(num_parameters, block_size);
    SGD_kernel<<<num_blocks, block_size>>>(params_memory, grads_memory, num_parameters, learning_rate, weight_decay);
}


int main()
{
    srand(0);
	float* params_memory;
	float* grads_memory;
	const unsigned long num_parameters = 1000;
	
	int deviceIdx = 0;
	cudaCheck(cudaSetDevice(deviceIdx));

	params_memory = make_random_float(num_parameters);
	grads_memory = make_random_float(num_parameters);

#if TEST_PYTORTH
write_npy_float("SGD-optimizer\\params_memory.npy", params_memory, 1, new size_t[1]{num_parameters});
write_npy_float("SGD-optimizer\\grads_memory.npy", grads_memory, 1, new size_t[1]{num_parameters});
#endif

	// move to GPU
	float* d_params_memory;
	float* d_grads_memory;
	cudaCheck(cudaMalloc(&d_params_memory, num_parameters * sizeof(float)));
	cudaCheck(cudaMalloc(&d_grads_memory, num_parameters * sizeof(float)));
	cudaCheck(cudaMemcpy(d_grads_memory, grads_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_params_memory, params_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));

	// run cpu
    SGD_cpu(params_memory, grads_memory, num_parameters);
#if TEST_PYTORTH
write_npy_float("SGD-optimizer\\updated_params_memory.npy", params_memory, 1, new size_t[1]{num_parameters});
#endif

	// run the kernel
	int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
	// first check the correctness of the kernel
	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
		int block_size = block_sizes[j];
		printf("Checking block size %d.\n", block_size);
        SGD_run_kernel(d_params_memory, d_grads_memory, num_parameters, 1e-3, 0.0);
		validate_result(d_params_memory, params_memory, "out", num_parameters, 1e-4f);
	}

	printf("All results match. Starting benchmarks.\n\n");
	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
		int block_size = block_sizes[j];

		int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, SGD_run_kernel, d_params_memory, d_grads_memory, num_parameters, 1e-3, 0.0, block_size);
		printf("block_size %4d | time %.4f ms | per token %.2f �s\n", block_size, elapsed_time, elapsed_time * 1'000 / (num_parameters));
	}

	//free memory
	free(params_memory);
	free(grads_memory);
	cudaCheck(cudaFree(d_params_memory));
	cudaCheck(cudaFree(d_grads_memory));
    return 0;
}