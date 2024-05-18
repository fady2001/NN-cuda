 //#include "common.hpp"
 //#include "device_launch_parameters.h"
 //#include <cmath>
 //#include <iostream>
 //#define TEST_PYTORTH true
 ///**
 //* @brief
 //*  this is a template function to perform NLL loss
 //*  its input is the probabilities from the softmax and the targets
 //*
 //* @param losses: output tensor of shape (N)
 //* @param input: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
 //* @param targets: target tensor of shape (N) contains number from 0 to C-1
 //* @param N: number of rows
 //* @param C: number of columns
 //*/
 //template<class T>
 //void nll_loss(T* losses,T* input, uint* targets,int N, int C) {
 //	 output: losses is (N) of the individual losses for each batch
 //	 input: input are (N,C) of the probabilities from softmax
 //	 input: targets is (N) of integers giving the correct index in logits
 //	for (int i = 0; i < N; i++) {
	//	losses[i] = -input[i * C + targets[i]];
 //	}
 //}

 // kernel for cross_entropy
 //template<class T>
 //__global__ void nll_loss_kernel(T* losses, T* input, uint* targets, int N, int C) {
 //	int i = blockIdx.x * blockDim.x + threadIdx.x;
 //	if (i < N) {
 //		losses[i] = -input[i * C + targets[i]];
 //	}
 //}

 //template <class T>
 //void run_nll_loss_kernel(T *losses, T *probs, uint *targets, int N, int C, const int block_size)
 //{
 //  const int grid_size = ceil_div(N, block_size);
 //  nll_loss_kernel<<<grid_size, block_size>>>(losses, probs, targets, N, C);
 //  cudaCheck(cudaGetLastError());
 //}

 //int main()
 //{
	//srand(0);
	//float* h_losses;
	//float* h_predictions;
 //	uint* h_targets;
 //	const unsigned long C = 3;
 //	const unsigned long N = 3;

 //	int deviceIdx = 0;
 //	cudaCheck(cudaSetDevice(deviceIdx));

 //	h_losses = (float*)malloc(N * sizeof(float));

 //	h_targets = make_random_int(N, C);
 //	h_predictions = make_random_float_01(N * C);

 //#if TEST_PYTORTH
 //  write_npy<float>("nll-loss-layer\\h_predictions.npy", h_predictions, 2, new unsigned long[2]{N, C});
 //  write_npy<uint>("nll-loss-layer\\h_targets.npy", h_targets, 1, new unsigned long[1]{N});
 //#endif

 //	 move to GPU
 //	float* d_losses;
 //	float* d_predictions;
 //	uint* d_targets;
 //	cudaCheck(cudaMalloc(&d_losses, N * sizeof(float)));
 //	cudaCheck(cudaMalloc(&d_predictions, N * C * sizeof(float)));
 //	cudaCheck(cudaMalloc(&d_targets, N * sizeof(uint)));
 //	cudaCheck(cudaMemcpy(d_predictions, h_predictions, N * C * sizeof(float), cudaMemcpyHostToDevice));
 //	cudaCheck(cudaMemcpy(d_targets, h_targets, N * sizeof(uint), cudaMemcpyHostToDevice));

 //	nll_loss<float>(h_losses, h_predictions, h_targets, N, C);
 //#if TEST_PYTORTH
 // write_npy<float>("nll-loss-layer\\h_losses.npy", h_losses, 1, new unsigned long[1]{N});
 //#endif

 //	 run the kernel
 //	int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
 //	 first check the correctness of the kernel
 //	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
 //		int block_size = block_sizes[j];
 //		printf("Checking block size %d.\n", block_size);
 //		run_nll_loss_kernel(d_losses, d_predictions, d_targets, N, C, block_sizes[j]);
 //		validate_result(d_losses, h_losses, "out", N, 1e-4f);
 //	}

 //	printf("All results match. Starting benchmarks.\n\n");
 //	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
 //		int block_size = block_sizes[j];

 //		int repeat_times = 100;
 //		float elapsed_time = benchmark_kernel(repeat_times, run_nll_loss_kernel<float>, d_losses, d_predictions, d_targets, N, C, block_sizes[j]);

 //		printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (N * C));
 //	}

 //	free memory
 //	free(h_losses);
 //	free(h_predictions);
 //	free(h_targets);
 //	cudaCheck(cudaFree(d_losses));
 //	cudaCheck(cudaFree(d_predictions));
 //	cudaCheck(cudaFree(d_targets));
 //	return 0;
 //}