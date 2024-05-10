#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class ModelLayersKernels {
public:
	/**
	* @brief
	*  this performs the forward pass of a linear layer
	* y = x W.T  + b
	*
	* @param X: input tensor of shape (B, N) where B is the batch size and N is the number of input neurons
	* @param W: weight tensor of shape (M, N) where M is the number of output neurons
	* @param bias: bias tensor of shape (M)
	* @param y: output tensor of shape (B, M)
	*/
	template<class T>
 __global__ static void linear_layer_forward_naive(T* X, T* W, T* bias, T* y, int B, int N, int M)
 {
 	// this maps o  ne thread to one output element
 	// the grid size is (B,M,1)
 	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	int j = blockIdx.y * blockDim.y + threadIdx.y;

 	// will be used to store the dot product of the i-th row of X and the j-th row of W
 	if (i < B && j < M)
 	{
 		T dot_product = bias[j];
 		for (unsigned long k = 0; k < N; k++)
 		{
 			dot_product += X[i * N + k] * W[j * N + k];
 		}
 		// store the result in y with the bias
 		y[i * M + j] = dot_product;
 	}
 }

  /**
 * @brief
 *  This function performs the forward pass of a ReLU activation function.
 *
 * @param input: Input tensor of shape (B, N) where B is the batch size and N is the number of elements per batch.
 * @param output: Output tensor of the same shape as the input.
 */
 template<class T>
 __global__ static void relu_forward(T* input, T* output, int B, int N)
 {
 	// This maps one thread to one element in the input.
 	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	int j = blockIdx.y * blockDim.y + threadIdx.y;

 	if (i < B && j < N)
 	{
 		int idx = i * N + j;
 		output[idx] = fmaxf(0.0f, input[idx]);
 	}
 }

 /* each thread will process only one row */
 template <class T>
 __global__ static void softmax_kernel(const T* in_h, T* out_d, int N, int C)
 {
 	// input dimension (N,C)
 	// output dimension (N,C)
 	// get actual index in in_h and out_d
 	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	if (i < N)
 	{
 		T max_val = in_h[i * C];
 		for (int j = 1; j < C; j++)
 		{
 			if (in_h[i * C + j] > max_val)
 			{
 				max_val = in_h[i * C + j];
 			}
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

 // kernel for cross_entropy
 template <class T>
 __global__ static void cross_entropy_kernel(T* losses, const T* input, const int* targets, int N, int C)
 {
 	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	if (i < N)
 	{
 		losses[i] = -log(input[i * C + targets[i]]);
 	}
 }

 template <class T>
 __global__ void array_sum_kernel3(T* d_a, T* d_result, int size)
 {
 	extern __shared__ T v[];
 	int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
 	int start_index = threadIdx.x * amount_per_thread;
 	int end_index = min(start_index + amount_per_thread, size);
 	T partialsum = 0.0f;
 	for (int k = start_index; k < end_index; k++)
 	{
 		partialsum += d_a[k];
 		v[threadIdx.x] = partialsum;
 	}
 	__syncthreads();

 	/*
 	The loop starts with `s` equal to half the block size (`blockDim.x`).
 	In each iteration of the loop, each thread with an index less than `s` adds the element at position `threadIdx.x + s` to the element at position `threadIdx.x` in the array `v`.
 	The operation `s>>=1` halves `s` in each iteration, effectively reducing the active size of the array by half in each step.
 	After each step, `__syncthreads()` is called to ensure that all threads have completed their computations before the next iteration begins. This is necessary because in the next iteration, some threads will be working with results computed by other threads in the current iteration.
 	This process continues until `s` becomes 0, at which point all elements of the array have been added together and the total is stored in `v[0]`.
 	*/
 	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
 	{
 		if (threadIdx.x < s)
 		{
 			v[threadIdx.x] += v[threadIdx.x + s];
 		}
 		__syncthreads();
 	}
 	if (threadIdx.x == 0)
 	{
 		d_result[0] = v[0];
 	}
 }
};
