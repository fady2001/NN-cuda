//#include "common.hpp"
//#include <stdio.h>
//#define TEST_PYTORTH true
//
//// ------------------------------- cpu version -------------------------------
///**
//* @brief
//*  this a template function to be suitable for float and double numbers to apply softmax
//*
//* @param in: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
//* @param out: output tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
//* @param N: number of rows
//* @param C: number of columns
//*/
//template <class T>
//void log_softmax_cpu(T *in, T *out, int N, int C)
//{
//	// loop over each row. each row will get softmaxed
//	for (int i = 0; i < N; i++)
//	{
//		// assume that the first element in the row is the maximum
//		T max_val = in[i * C];
//		// loop to get the maximum value of each row
//		for (int j = 1; j < C; j++)
//		{
//			if (in[i * C + j] > max_val)
//			{
//				max_val = in[i * C + j];
//			}
//		}
//
//		T sum = 0;
//		// loop over the row to calculate the sum and apply normalization
//		for (int j = 0; j < C; j++)
//		{
//			// apply normalization step to ensure that the maximum value will be 0 to avoid overflow
//			in[i * C + j] = in[i * C + j] - max_val;
//			sum += exp(in[i * C + j]);
//		}
//		// output softmaxed values
//		for (int j = 0; j < C; j++)
//		{
//			out[i * C + j] = in[i * C + j] - log(sum);
//		}
//	}
//}
//
//
//// ------------------------------- gpu version -------------------------------
//
///* each thread will process only one row */
//template <class T>
//__global__ void log_softmax_kernel(T *in_h, T *out_d, int N, int C)
//{
//	// input dimension (N,C)
//	// output dimension (N,C)
//	// get actual index in in_h and out_d
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < N)
//	{
//		T max_val = in_h[i * C];
//		for (int j = 1; j < C; j++)
//		{
//			if (in_h[i * C + j] > max_val)
//			{
//				max_val = in_h[i * C + j];
//			}
//		}
//
//		T sum = 0;
//		for (int j = 0; j < C; j++)
//		{
//			// apply normalization step to ensure that the maximum value will be 0 to avoid overflow
//			in_h[i * C + j] = in_h[i * C + j] - max_val;
//			sum += exp(in_h[i * C + j]);
//		}
//		// output softmaxed values
//		for (int j = 0; j < C; j++)
//		{
//			out_d[i * C + j] = in_h[i * C + j] - log(sum);
//		}
//	}
//}
//
//template <class T>
//void run_kernel1(T* input, T* output, int N, int C, int block_size)
//{
//	int num_blocks = ceil_div(N, block_size);
//	log_softmax_kernel << <num_blocks, block_size >> > (input, output, N, C);
//}
//
//int main()
//{
//	srand(0);
//	const unsigned long N = 10000, C = 10000;
//
//	int deviceIdx = 0;
//	cudaCheck(cudaSetDevice(deviceIdx));
//
//	// create host memory of random numbers
//	float* h_out = (float*)malloc(N * C * sizeof(float));
//	float* h_inp = make_random_float(N * C);
//
//#if TEST_PYTORTH
//  write_npy("softmax-layer\\h_inp.npy", h_inp, 2, new unsigned long[2]{N, C});
//#endif
//
//
//	// move to GPU
//	float* d_out;
//	float* d_inp;
//	cudaCheck(cudaMalloc(&d_out, N * C * sizeof(float)));
//	cudaCheck(cudaMalloc(&d_inp, N * C * sizeof(float)));
//	cudaCheck(cudaMemcpy(d_inp, h_inp, N * C * sizeof(float), cudaMemcpyHostToDevice));
//
//	measureExecutionTime(log_softmax_cpu<float>,h_inp, h_out, N, C);
//  
//#if TEST_PYTORTH
// write_npy("softmax-layer\\h_out.npy", h_out, 2, new unsigned long[2]{N, C});
//#endif
//
//	int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
//	// first check the correctness of the kernel
//	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
//		int block_size = block_sizes[j];
//		printf("Checking block size %d.\n", block_size);
//		run_kernel1(d_inp, d_out, N, C, block_sizes[j]);
//		validate_result(d_out, h_out, "out", N * C, 1e-4f);
//	}
//
//	printf("All results match. Starting benchmarks.\n\n");
//	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
//		int block_size = block_sizes[j];
//
//		int repeat_times = 100;
//		float elapsed_time = benchmark_kernel(repeat_times, run_kernel1<float>, d_inp, d_out, N, C, block_sizes[j]);
//
//		printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (N * C));
//	}
//
//	// free memory
//	free(h_out);
//	free(h_inp);
//	cudaCheck(cudaFree(d_out));
//	cudaCheck(cudaFree(d_inp));
//	return 0;
//}