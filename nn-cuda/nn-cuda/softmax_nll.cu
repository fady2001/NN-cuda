#include "common.hpp"
#include <stdio.h>
#define TEST_PYTORTH false

// ------------------------------- cpu version -------------------------------
/**
* @brief
*  this a template function to be suitable for float and double numbers to apply softmax
*
* @param in: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
* @param out: output tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
* @param N: number of rows
* @param C: number of columns
*/
template <class T>
void log_softmax_cpu(T *in, T *out, int N, int C)
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
			in[i * C + j] = in[i * C + j] - max_val;
			sum += exp(in[i * C + j]);
		}
		// output softmaxed values
		for (int j = 0; j < C; j++)
		{
			out[i * C + j] = in[i * C + j] - log(sum);
		}
	}
}

template<class T>
void nll_loss(T* losses, T* input, uint* targets, int N, int C) {
	//output: losses is(N) of the individual losses for each batch
	//input : input are(N, C) of the probabilities from softmax
	//input : targets is(N) of integers giving the correct index in logits
	for (int i = 0; i < N; i++) {
		losses[i] = -input[i * C + targets[i]];
	}
}

// ------------------------------- gpu version -------------------------------

/* each thread will process only one row */
template <class T>
__global__ void log_softmax_kernel(T *in_h, T *out_d, int N, int C)
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
			in_h[i * C + j] = in_h[i * C + j] - max_val;
			sum += exp(in_h[i * C + j]);
		}
		// output softmaxed values
		for (int j = 0; j < C; j++)
		{
			out_d[i * C + j] = in_h[i * C + j] - log(sum);
		}
	}
}

//  kernel for cross_entropy
template<class T>
__global__ void nll_loss_kernel(T* losses, T* input, uint* targets, int N, int C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		losses[i] = -input[i * C + targets[i]];
	}
}

template <class T>
void run_kernel1(T* input, T* output, int N, int C, int block_size)
{
	int num_blocks = ceil_div(N, block_size);
	log_softmax_kernel << <num_blocks, block_size >> > (input, output, N, C);
}

template <class T>
void run_nll_loss_kernel(T* losses, T* probs, uint* targets, int N, int C, const int block_size)
{
	const int grid_size = ceil_div(N, block_size);
	nll_loss_kernel << <grid_size, block_size >> > (losses, probs, targets, N, C);
	cudaCheck(cudaGetLastError());
}

int main()
{
	srand(0);
    float* h_in_softmax;
    float* h_out_softmax;
    float* h_losses;
    uint* h_targets;
    const unsigned long C = 10000;
    const unsigned long N = 10000;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    h_in_softmax = make_random_float(N * C);
    h_out_softmax = (float*)malloc(N * C * sizeof(float));
    h_losses = (float*)malloc(N * sizeof(float));
    h_targets = make_random_int(N, C);

#if TEST_PYTORTH
    write_npy<float>("log-softmax-layer\\h_in_softmax.npy", h_in_softmax, 2, new unsigned long[2]{ N, C });
    write_npy<uint>("log-softmax-layer\\h_targets.npy", h_targets, 1, new unsigned long[1]{ N });
#endif
        // CPU
        log_softmax_cpu<float>(h_in_softmax, h_out_softmax, N, C);
        nll_loss<float>(h_losses, h_out_softmax, h_targets, N, C);
        measureExecutionTime(log_softmax_cpu<float>, h_in_softmax, h_out_softmax, N, C);
        measureExecutionTime(nll_loss<float>, h_losses, h_out_softmax, h_targets, N, C);

        //  move to GPU
        float* d_in_softmax;
        float* d_out_softmax;
        float* d_losses;
        uint* d_targets;
        cudaCheck(cudaMalloc(&d_in_softmax, N * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out_softmax, N * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_losses, N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_targets, N * sizeof(uint)));
        cudaCheck(cudaMemcpy(d_in_softmax, h_in_softmax, N * C * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_targets, h_targets, N * sizeof(uint), cudaMemcpyHostToDevice));

        //  run the kernel

        int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
        // first check the correctness of the kernel
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("Checking block size %d.\n", block_size);
            run_kernel1(d_in_softmax, d_out_softmax, N, C, block_sizes[j]);
            validate_result(d_out_softmax, h_out_softmax, "out", N * C, 1e-4f);
            run_nll_loss_kernel(d_losses, d_out_softmax, d_targets, N, C, block_sizes[j]);
            validate_result(d_losses, h_losses, "out", N, 1e-4f);
        }
    
        printf("All results match. Starting benchmarks.\n\n");
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
    
            int repeat_times = 100;
            float elapsed_time = benchmark_kernel(repeat_times, run_kernel1<float>, d_in_softmax, d_out_softmax, N, C, block_sizes[j]);
            float elapsed_time2 = benchmark_kernel(repeat_times, run_nll_loss_kernel<float>, d_losses, d_out_softmax, d_targets, N, C, block_sizes[j]);

            // total time
            printf("Total time for block size %d is %f ms.\n", block_size, elapsed_time + elapsed_time2);
        }
    
        // free memory
        
        return 0;}