#include <stdio.h>
#include "common.cuh"
#include <iostream>
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

void softmax(float* input, float* output, int N, int C) {
	float* d_input, * d_output;

	cudaMalloc((void**)& d_input, N * C * sizeof(float));
	cudaMalloc((void**)& d_output, N * C * sizeof(float));

	cudaMemcpy(d_input, input, N * C * sizeof(float), cudaMemcpyHostToDevice);

	int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	softmax_kernel <<<num_blocks, BLOCK_SIZE>>> (d_input, d_output, N, C);

	cudaMemcpy(output, d_output, N * C * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

int main() {
	const int N = 5; // Number of samples
	const int C = 3; // Number of classes
	float input[N][C] = {
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
		{13.0, 14.0, 15.0}
	};
	float output[N][C];

	softmax(reinterpret_cast<float*>(input), reinterpret_cast<float*>(output), N, C);

	std::cout << "Softmax output:" << std::endl;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < C; ++j) {
			std::cout << output[i][j] << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}

//int main()
//{
//	// define the input and output arrays
//	float in[] = { 1, 2, 3, 4 };
//	float out[4];
//	// call the softmax function
//	softmax_cpu(in, out, 1, 4);
//	// print the output
//	for (int i = 0; i < 4; i++)
//	{
//		printf("%f ", out[i]);
//	}
//	printf("\n");
//	return 0;
//}