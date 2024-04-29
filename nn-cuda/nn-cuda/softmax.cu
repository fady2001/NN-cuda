#include <stdio.h>
#include "common.cuh"
#include <iostream>
#include <fstream>

#define BLOCK_SIZE 256


template <class T>
// ------------------------------- cpu version -------------------------------
void softmax_cpu(const T* in, T* out, int N, int C)
{
	// loop over each row. each row will get softmaxed
	for (int i = 0; i < N; i++)
	{
		// assume that the first element in the row is the maximum
		T max_val = in[i * C];
		// loop to get the maximum value of each row
		for (int j = 1; j < C; j++)
		{
			if (in[i * C + j] > max_val)
			{
				max_val = in[i * C + j];
			}
		}

		T sum = 0;
		// loop over the row to calculate the sum and apply normalization
		for (int j = 0; j < C; j++)
		{
			// apply normalization step to ensure that the maximum value will be 0 to avoid overflow 
			out[i * C + j] = exp(in[i * C + j] - max_val);
			sum += out[i * C + j];
		}
		// output softmaxed values
		for (int j = 0; j < C; j++)
		{
			out[i * C + j] /= sum;
		}
	}
}


// ------------------------------- gpu version -------------------------------
/* each thread will process only one row */
template <class T>
__global__ void softmax_kernel(const T* in_h, T* out_d, int N, int C)
{
	// input dimension (N,C)
	// output dimension (N,C)
	// get actual index in in_h and out_d
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		T max_val = in_h[i * C];
		for (int j = 1; j < C; j++) {
			max_val = in_h[i * C + j];
		}

		T sum = 0;
		for (int j = 0; j < C; j++)
		{
			// apply normalization step to ensure that the maximum value will be 0 to avoid overflow 
			out_d[i * C + j] = exp(in_h[i * C + j] - max_val);
			sum += out_d[i * C + j];
		}
		// output softmaxed values
		for (int j = 0; j < C; j++)
		{
			out_d[i * C + j] /= sum;
		}
	}
}

template <class T>
void run_kernel1(const T* input, T* output, int N, int C, int Depth,int block_size)
{
	int num_blocks = ceil_div(N, block_size);
	softmax_kernel << <num_blocks, block_size >> > (input, output, N * C, Depth);
}

int main()
{
	srand(0);
	int B = 100, T = 100, V = 10;

	int deviceIdx = 0;
	cudaCheck(cudaSetDevice(deviceIdx));

	// create host memory of random numbers
	float* h_out = (float*)malloc(B * T * V * sizeof(float));
	float* h_inp = make_ones_float(B * T * V);
	// write h_inp to file
	std::ofstream outfile("output.txt");
	if (outfile.is_open()) {
		for (int i = 0; i < B * T * V; i++) {
			outfile << h_inp[i] << std::endl;
		}
		outfile.close();
	} else {
		std::cout << "Unable to open file";
	}
	

	// make the input less uniformly random: Otherwise, all probabilities will be basically zero,
	// and the tests are not actually meaningful.
	const int* outliers = make_random_int(B * T * 3, V);
	for (int k = 0; k < 3; ++k) {
		for (int j = 0; j < B * T; ++j) {
			h_inp[j * V + outliers[j * 3 + k]] *= 20;
		}
	}

	// move to GPU
	float* d_out;
	float* d_inp;
	cudaCheck(cudaMalloc(&d_out, B * T * V * sizeof(float)));
	cudaCheck(cudaMalloc(&d_inp, B * T * V * sizeof(float)));
	cudaCheck(cudaMemcpy(d_inp, h_inp, B * T * V * sizeof(float), cudaMemcpyHostToDevice));
	
	softmax_cpu(h_inp, h_out, B * T, V);

	int block_sizes[] = { 32 };
	// first check the correctness of the kernel
	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
		int block_size = block_sizes[j];
		printf("Checking block size %d.\n", block_size);
		run_kernel1(d_inp, d_out, B, T, V, block_sizes[j]);
		validate_result(d_out, h_out, "out", B * T * V, 1e-4f);
	}

	printf("All results match. Starting benchmarks.\n\n");
	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
		int block_size = block_sizes[j];

		int repeat_times = 100;
		float elapsed_time = benchmark_kernel(repeat_times, run_kernel1<float>, d_inp, d_out, B, T, V, block_sizes[j]);

		printf("block_size %4d | time %.4f ms | per token %.2f �s\n", block_size, elapsed_time, elapsed_time * 1'000 / (B * T));
	}

	// free memory
	free(h_out);
	free(h_inp);
	cudaCheck(cudaFree(d_out));
	cudaCheck(cudaFree(d_inp));
	return 0;
}