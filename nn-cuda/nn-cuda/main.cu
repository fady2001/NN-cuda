// #include "common.cuh"
// #include <math.h>
// #define TEST_PYTORTH true
// #include "device_launch_parameters.h"
// /**
// * @brief
// *  this performs the forward pass of a linear layer
// * y = x W.T  + b
// *
// * @param X: input tensor of shape (B, N) where B is the batch size and N is the number of input neurons
// * @param W: weight tensor of shape (M, N) where M is the number of output neurons
// * @param bias: bias tensor of shape (M)
// * @param y: output tensor of shape (B, M)
// */
// __global__ void linear_layer_forward_naive(float* X, float* W, float* bias, float* y, int B, int N, int M)
// {
// 	// this maps o  ne thread to one output element
// 	// the grid size is (B,M,1)
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	int j = blockIdx.y * blockDim.y + threadIdx.y;

// 	// will be used to store the dot product of the i-th row of X and the j-th row of W
// 	if (i < B && j < M)
// 	{
// 		float dot_product = bias[j];
// 		for (unsigned long k = 0; k < N; k++)
// 		{
// 			dot_product += X[i * N + k] * W[j * N + k];
// 		}
// 		// store the result in y with the bias
// 		y[i * M + j] = dot_product;
// 	}
// }

// void linear_layer_forward_cpu(float* X, float* W, float* bias, float* y, int B, int N, int M)
// {

// 	for (int i = 0; i < B; i++)
// 	{
// 		for (int j = 0; j < M; j++)
// 		{
// 			y[i * M + j] = bias[j];
// 			for (int k = 0; k < N; k++)
// 			{
// 				y[i * M + j] += X[i * N + k] * W[j * N + k];
// 			}
// 		}
// 	}
// }
// void linear_layer(float* X, float* W, float* bias, float* y, int B, int N, int M, int sqrt_block_size)
// {
// 	dim3 block(sqrt_block_size, sqrt_block_size);
// 	dim3 grid((B + block.x - 1) / block.x, (M + block.y - 1) / block.y);
// 	linear_layer_forward_naive << <grid, block >> > (X, W, bias, y, B, N, M);
// 	cudaCheck(cudaDeviceSynchronize());
// }
// /**
// * @brief
// *  This function performs the forward pass of a ReLU activation function.
// *
// * @param input: Input tensor of shape (B, N) where B is the batch size and N is the number of elements per batch.
// * @param output: Output tensor of the same shape as the input.
// */
// __global__ void relu_forward(float* input, float* output, int B, int N)
// {
// 	// This maps one thread to one element in the input.
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	int j = blockIdx.y * blockDim.y + threadIdx.y;

// 	if (i < B && j < N)
// 	{
// 		int idx = i * N + j;
// 		output[idx] = fmaxf(0.0f, input[idx]);
// 	}
// }
// void relu_forward_cpu(float* input, float* output, int B, int N)
// {
// 	for (int i = 0; i < B; i++)
// 	{
// 		for (int j = 0; j < N; j++)
// 		{
// 			int idx = i * N + j;
// 			output[idx] = fmaxf(0.0f, input[idx]);
// 		}
// 	}
// }
// void run_relu_kernel(float* input, float* output, int B, int N, int sqrt_block_size)
// {
// 	dim3 block(sqrt_block_size, sqrt_block_size);
// 	dim3 grid((B + block.x - 1) / block.x, (N + block.y - 1) / block.y);
// 	relu_forward << <grid, block >> > (input, output, B, N);
// 	cudaCheck(cudaDeviceSynchronize());
// }

// //-----------------------------------------------------------------------------------
// // ------------------------------- cpu version -------------------------------
// /**
// * @brief
// *  this a template function to be suitable for float and double numbers to apply softmax
// *
// * @param in: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
// * @param out: output tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
// * @param N: number of rows
// * @param C: number of columns
// */
// template <class T>
// void softmax_cpu(const T* in, T* out, int N, int C)
// {
// 	// loop over each row. each row will get softmaxed
// 	for (int i = 0; i < N; i++)
// 	{
// 		// assume that the first element in the row is the maximum
// 		T max_val = in[i * C];
// 		// loop to get the maximum value of each row
// 		for (int j = 1; j < C; j++)
// 		{
// 			if (in[i * C + j] > max_val)
// 			{
// 				max_val = in[i * C + j];
// 			}
// 		}

// 		T sum = 0;
// 		// loop over the row to calculate the sum and apply normalization
// 		for (int j = 0; j < C; j++)
// 		{
// 			// apply normalization step to ensure that the maximum value will be 0 to avoid overflow
// 			out[i * C + j] = exp(in[i * C + j] - max_val);
// 			sum += out[i * C + j];
// 		}
// 		// output softmaxed values
// 		for (int j = 0; j < C; j++)
// 		{
// 			out[i * C + j] /= sum;
// 		}
// 	}
// }

// // ------------------------------- gpu version -------------------------------
// /* each thread will process only one row */
// template <class T>
// __global__ void softmax_kernel(const T* in_h, T* out_d, int N, int C)
// {
// 	// input dimension (N,C)
// 	// output dimension (N,C)
// 	// get actual index in in_h and out_d
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (i < N)
// 	{
// 		T max_val = in_h[i * C];
// 		for (int j = 1; j < C; j++)
// 		{
// 			if (in_h[i * C + j] > max_val)
// 			{
// 				max_val = in_h[i * C + j];
// 			}
// 		}

// 		T sum = 0;
// 		for (int j = 0; j < C; j++)
// 		{
// 			// apply normalization step to ensure that the maximum value will be 0 to avoid overflow
// 			out_d[i * C + j] = exp(in_h[i * C + j] - max_val);
// 			sum += out_d[i * C + j];
// 		}
// 		// output softmaxed values
// 		for (int j = 0; j < C; j++)
// 		{
// 			out_d[i * C + j] /= sum;
// 		}
// 	}
// }

// template <class T>
// void run_softmax_kernel(const T* input, T* output, int N, int C, int block_size)
// {
// 	int num_blocks = (N + block_size - 1) / block_size;
// 	softmax_kernel << <num_blocks, block_size >> > (input, output, N, C);
// }
// //-----------------------------------------------------------------------------------
// /**
// * @brief
// *  this is a template function to perform NLL loss
// *  its input is the probabilities from the softmax and the targets
// *
// * @param losses: output tensor of shape (N)
// * @param input: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
// * @param targets: target tensor of shape (N) contains number from 0 to C-1
// * @param N: number of rows
// * @param C: number of columns
// */
// template <class T>
// void cross_entropy_cpu(T* losses, const T* input, const int* targets, int N, int C)
// {
// 	// output: losses is (N) of the individual losses for each batch
// 	// input: input are (N,C) of the probabilities from softmax
// 	// input: targets is (N) of integers giving the correct index in logits
// 	for (int i = 0; i < N; i++)
// 	{
// 		losses[i] = -log(input[i * C + targets[i]]);
// 	}
// }

// // kernel for cross_entropy
// template <class T>
// __global__ void cross_entropy_kernel(T* losses, const T* input, const int* targets, int N, int C)
// {
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (i < N)
// 	{
// 		losses[i] = -log(input[i * C + targets[i]]);
// 	}
// }

// template <class T>
// void run_cross_entropy_kernel(T* losses, const T* probs, const int* targets, int N, int C, const int block_size)
// {
// 	//(dividend + divisor - 1) / divisor
// 	const int grid_size = (N + block_size - 1) / block_size;
// 	cross_entropy_kernel << <grid_size, block_size >> > (losses, probs, targets, N, C);
// 	cudaCheck(cudaGetLastError());
// }
// //-----------------------------------------------------------------------------------
// template <class T>
// __global__ void array_sum_kernel3(T* d_a, T* d_result, int size)
// {
// 	extern __shared__ T v[];
// 	int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
// 	int start_index = threadIdx.x * amount_per_thread;
// 	int end_index = min(start_index + amount_per_thread, size);
// 	T partialsum = 0.0f;
// 	for (int k = start_index; k < end_index; k++)
// 	{
// 		partialsum += d_a[k];
// 		v[threadIdx.x] = partialsum;
// 	}
// 	__syncthreads();

// 	/*
// 	The loop starts with `s` equal to half the block size (`blockDim.x`).
// 	In each iteration of the loop, each thread with an index less than `s` adds the element at position `threadIdx.x + s` to the element at position `threadIdx.x` in the array `v`.
// 	The operation `s>>=1` halves `s` in each iteration, effectively reducing the active size of the array by half in each step.
// 	After each step, `__syncthreads()` is called to ensure that all threads have completed their computations before the next iteration begins. This is necessary because in the next iteration, some threads will be working with results computed by other threads in the current iteration.
// 	This process continues until `s` becomes 0, at which point all elements of the array have been added together and the total is stored in `v[0]`.
// 	*/
// 	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
// 	{
// 		if (threadIdx.x < s)
// 		{
// 			v[threadIdx.x] += v[threadIdx.x + s];
// 		}
// 		__syncthreads();
// 	}
// 	if (threadIdx.x == 0)
// 	{
// 		d_result[0] = v[0];
// 	}
// }

// template <class T>
// void run_array_sum_kernel3(T* d_a, T* d_result, int size, int block_size)
// {
// 	// (dividend + divisor - 1) / divisor
// 	int num_blocks = (size + block_size - 1) / block_size;
// 	array_sum_kernel3 << <1, num_blocks, block_size * sizeof(T) >> > (d_a, d_result, size);
// 	cudaCheck(cudaGetLastError());
// }

// /*
// * @brief
// * This will include the model parameters like weights and bias for each layer
// */
// #define NUM_PARAMETER_ARRAYS 4
// #define NUM_ACTIVATION_ARRAYS 6
// typedef struct
// {
// 	float* ln1w; // linear layer 1 weights (H1 x N)
// 	float* ln1b; // linear layer 1 bias (H1)
// 	float* ln2w; // linear layer 2 weights (C x H1)
// 	float* ln2b; // linear layer 2 bias (C)
// } ModelParameters;

// /*
// * @brief
// * This will include the model activations like the output of each layer
// */
// typedef struct
// {
// 	float* ln1;          // linear layer 1 output (B x H1)
// 	float* a1;           // activation 1 output (B x H1)
// 	float* ln2;          // linear layer 2 output (B x C) -- C is the number of classes = C
// 	float* sm;           // softmax output (B x C)
// 	float* loss;         // loss (B)
// 	float* reduced_loss; // reduced loss (1)
// } ModelActivation;

// typedef struct
// {
// 	unsigned long param_sizes[NUM_PARAMETER_ARRAYS];
// 	ModelParameters params;
// 	float* params_memory;

// 	unsigned long activation_sizes[NUM_ACTIVATION_ARRAYS];
// 	ModelActivation activations;
// 	float* activations_memory;
// } TwoLayerModel;

// typedef enum
// {
// 	PARAMETERS_TYPE,
// 	ACTIVATIONS_TYPE
// } DATA_TYPE;

// typedef enum
// {
// 	ZEROS_V,
// 	ONES_V,
// 	RANDOM_V
// } INITIAL_VALUE_TYPE;

// float* float_cpu_malloc_and_point(void* data, unsigned long* sizes, int num_arrays, DATA_TYPE type, INITIAL_VALUE_TYPE initial_value = ZEROS_V)
// {
// 	unsigned long total_size = 0;
// 	for (int i = 0; i < num_arrays; i++)
// 	{
// 		total_size += sizes[i];
// 	}
// 	float* memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

// 	switch (initial_value)
// 	{
// 	case ZEROS_V:
// 		memory = make_zeros_float(total_size);
// 		break;
// 	case ONES_V:
// 		memory = make_ones_float(total_size);
// 		break;
// 	case RANDOM_V:
// 		memory = make_random_float(total_size);
// 		break;
// 	}
// 	if (memory == NULL)
// 	{
// 		// Handle allocation failure
// 		return NULL;
// 	}
// 	switch (type)
// 	{
// 	case PARAMETERS_TYPE: // ModelParameters
// 	{
// 		ModelParameters* params = (ModelParameters*)data;
// 		float** ptrs[] = { &params->ln1w,
// 						  &params->ln1b,
// 						  &params->ln2w,
// 						  &params->ln2b };
// 		float* memory_iterator = memory;
// 		for (int i = 0; i < num_arrays; i++)
// 		{
// 			*(ptrs[i]) = memory_iterator;
// 			memory_iterator += sizes[i];
// 		}
// 	}
// 	break;
// 	case ACTIVATIONS_TYPE: // ModelActivation
// 	{
// 		ModelActivation* activations = (ModelActivation*)data;
// 		float** ptrs[] = { &activations->ln1,
// 						  &activations->a1,
// 						  &activations->ln2,
// 						  &activations->sm,
// 						  &activations->loss,
// 						  &activations->reduced_loss };
// 		float* memory_iterator = memory;
// 		for (int i = 0; i < num_arrays; i++)
// 		{
// 			*(ptrs[i]) = memory_iterator;
// 			memory_iterator += sizes[i];
// 		}
// 	}
// 	break;
// 	}
// 	return memory;
// }

// void model_to_cuda(TwoLayerModel* d_model, TwoLayerModel* h_model)
// {
// 	// copy parameters
// 	unsigned long total_param_size = 0;
// 	for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
// 	{
// 		total_param_size += h_model->param_sizes[i];
// 		d_model->param_sizes[i] = h_model->param_sizes[i];
// 	}
// 	cudaCheck(cudaMalloc(&d_model->params_memory, total_param_size * sizeof(float)));
// 	cudaCheck(cudaMemcpy(d_model->params_memory, h_model->params_memory, total_param_size * sizeof(float), cudaMemcpyHostToDevice));
// 	// point to the copied memory
// 	float* memory_iterator = (float*)d_model->params_memory;

// 	ModelParameters* params = &d_model->params;
// 	float** ptrs[] = { &params->ln1w,
// 					  &params->ln1b,
// 					  &params->ln2w,
// 					  &params->ln2b };
// 	for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
// 	{
// 		*(ptrs[i]) = memory_iterator;
// 		memory_iterator += h_model->param_sizes[i];
// 	}

// 	// copy activations
// 	unsigned long total_activation_size = 0;
// 	for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
// 	{
// 		total_activation_size += h_model->activation_sizes[i];
// 		d_model->activation_sizes[i] = h_model->activation_sizes[i];
// 	}
// 	cudaCheck(cudaMalloc(&d_model->activations_memory, total_activation_size * sizeof(float)));
// 	cudaCheck(cudaMemcpy(d_model->activations_memory, h_model->activations_memory, total_activation_size * sizeof(float), cudaMemcpyHostToDevice));

// 	memory_iterator = (float*)d_model->activations_memory;
// 	ModelActivation* activations = &d_model->activations;
// 	float** ptrs2[] = { &activations->ln1,
// 					   &activations->a1,
// 					   &activations->ln2,
// 					   &activations->sm,
// 					   &activations->loss,
// 					   &activations->reduced_loss };
// 	for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
// 	{
// 		*(ptrs2[i]) = memory_iterator;
// 		memory_iterator += h_model->activation_sizes[i];
// 	}
// }

// void clean_model_cpu(TwoLayerModel* model)
// {
// 	free(model->params_memory);
// 	free(model->activations_memory);
// }
// void clean_model_gpu(TwoLayerModel* model)
// {
// 	cudaCheck(cudaFree(model->params_memory));
// 	cudaCheck(cudaFree(model->activations_memory));
// }
// int main()
// {
// 	unsigned long input_dim = 3;
// 	unsigned long B = 30;
// 	unsigned long H1 = 100;
// 	unsigned long C = 10;
// 	TwoLayerModel model;

// 	model.param_sizes[0] = H1 * input_dim; // ln1w
// 	model.param_sizes[1] = H1;             // ln1b
// 	model.param_sizes[2] = C * H1;        // ln2w
// 	model.param_sizes[3] = C;             // ln2b
// 	model.params_memory = float_cpu_malloc_and_point(&(model.params), model.param_sizes, NUM_PARAMETER_ARRAYS, PARAMETERS_TYPE, RANDOM_V);

// 	if (model.params_memory == NULL)
// 	{
// 		// Handle allocation failure
// 		printf("Allocation failure\n");
// 		return 1;
// 	}
// 	// Now Activations
// 	model.activation_sizes[0] = B * H1; // ln1
// 	model.activation_sizes[1] = B * H1; // a1
// 	model.activation_sizes[2] = B * C; // ln2
// 	model.activation_sizes[3] = B * C;  // sm
// 	model.activation_sizes[4] = B;      // loss
// 	model.activation_sizes[5] = 1;      // reduced_loss
// 	model.activations_memory = float_cpu_malloc_and_point(&model.activations, model.activation_sizes, NUM_ACTIVATION_ARRAYS, ACTIVATIONS_TYPE, RANDOM_V);

// 	int deviceIdx = 0;
// 	cudaCheck(cudaSetDevice(deviceIdx));

// 	// create host memory of random numbers

// 	float* inp = make_random_float(B * input_dim);
// 	int* target = make_random_int(B, int(C));

// #if TEST_PYTORTH
// 	write_npy("all-model\\X_c.npy", inp, 2, new unsigned long[2]{ B, input_dim });
// 	write_npy("all-model\\target.npy", target, 1, new unsigned long[1]{ B });
// 	write_npy("all-model\\ln1w.npy", model.params.ln1w, 2, new unsigned long[2]{ H1, input_dim });
// 	write_npy("all-model\\ln1b.npy", model.params.ln1b, 1, new unsigned long[1]{ H1 });
// 	write_npy("all-model\\ln2w.npy", model.params.ln2w, 2, new unsigned long[2]{ C, H1 });
// 	write_npy("all-model\\ln2b.npy", model.params.ln2b, 1, new unsigned long[1]{ C });
// #endif

// 	// move to GPU
// 	TwoLayerModel d_model;
// 	model_to_cuda(&d_model, &model);

// 	// move input and target to GPU
// 	float* d_inp;
// 	int* d_target;
// 	cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(float)));
// 	cudaCheck(cudaMalloc(&d_target, B * sizeof(int)));
// 	cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(float), cudaMemcpyHostToDevice));
// 	cudaCheck(cudaMemcpy(d_target, target, B * sizeof(int), cudaMemcpyHostToDevice));

// 	// run the model
// 	linear_layer(d_inp, d_model.params.ln1w, d_model.params.ln1b, d_model.activations.ln1, B, input_dim, H1, 32);
// 	run_relu_kernel(d_model.activations.ln1, d_model.activations.a1, B, H1, 32);
// 	linear_layer(d_model.activations.a1, d_model.params.ln2w, d_model.params.ln2b, d_model.activations.ln2, B, H1, C, 32);
// 	run_softmax_kernel(d_model.activations.ln2, d_model.activations.sm, B, C, 32);
// 	run_cross_entropy_kernel(d_model.activations.loss, d_model.activations.sm, d_target, B, C, 32);
// 	run_array_sum_kernel3(d_model.activations.loss, d_model.activations.reduced_loss, B, 32);

// 	// cudasynchronize();
// 	cudaCheck(cudaDeviceSynchronize());
// 	// copy the loss to the host
// 	float* reduced_loss = (float*)malloc(sizeof(float));
// 	cudaCheck(cudaMemcpy(reduced_loss, d_model.activations.reduced_loss, sizeof(float), cudaMemcpyDeviceToHost));
// 	printf("Loss: %f\n", *reduced_loss);
// }