#include "common.cuh"
#include <math.h>
#define TEST_PYTORTH true
#include "device_launch_parameters.h"
#include "ModelMemoryHandler.cuh"
#include "ModelLayersKernelsLaunchers.cuh"

template<class T = float>
void model_to_cuda(ModelMemoryHandler<T>* d_model, ModelMemoryHandler<T>* h_model)
{
	// copy parameters
	unsigned long total_param_size = 0;
	for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
	{
		total_param_size += h_model->param_sizes[i];
		d_model->param_sizes[i] = h_model->param_sizes[i];
	}
	cudaCheck(cudaMalloc(&d_model->params_memory, total_param_size * sizeof(float)));
	cudaCheck(cudaMemcpy(d_model->params_memory, h_model->params_memory, total_param_size * sizeof(float), cudaMemcpyHostToDevice));
	// point to the copied memory
	float* memory_iterator = (float*)d_model->params_memory;

	ModelParameters<T>* params = &d_model->params;
	float** ptrs[] = { &params->ln1w,
					  &params->ln1b,
					  &params->ln2w,
					  &params->ln2b };
	for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
	{
		*(ptrs[i]) = memory_iterator;
		memory_iterator += h_model->param_sizes[i];
	}

	// copy activations
	unsigned long total_activation_size = 0;
	for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
	{
		total_activation_size += h_model->activation_sizes[i];
		d_model->activation_sizes[i] = h_model->activation_sizes[i];
	}
	cudaCheck(cudaMalloc(&d_model->activations_memory, total_activation_size * sizeof(float)));
	cudaCheck(cudaMemcpy(d_model->activations_memory, h_model->activations_memory, total_activation_size * sizeof(float), cudaMemcpyHostToDevice));

	memory_iterator = (float*)d_model->activations_memory;
	ModelActivation<T>* activations = &d_model->activations;
	float** ptrs2[] = { &activations->ln1,
					   &activations->a1,
					   &activations->ln2,
					   &activations->sm,
					   &activations->loss,
					   &activations->reduced_loss };
	for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
	{
		*(ptrs2[i]) = memory_iterator;
		memory_iterator += h_model->activation_sizes[i];
	}
}


int main()
{
	unsigned long input_dim = 3;
	unsigned long B = 2;
	unsigned long H1 = 5;
	unsigned long C = 3;
	ModelMemoryHandler<float> h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

	// create host memory of random numbers
	float* inp = make_random_float(B * input_dim);
	int* target = make_random_int(B, int(C));

#if TEST_PYTORTH
	write_npy("all-model\\X_c.npy", inp, 2, new unsigned long[2]{ B, input_dim });
	write_npy("all-model\\target.npy", target, 1, new unsigned long[1]{ B });
	write_npy("all-model\\ln1w.npy", h_model.GetParams().ln1w, 2, new unsigned long[2]{ H1, input_dim });
	write_npy("all-model\\ln1b.npy", h_model.GetParams().ln1b, 1, new unsigned long[1]{ H1 });
	write_npy("all-model\\ln2w.npy", h_model.GetParams().ln2w, 2, new unsigned long[2]{ C, H1 });
	write_npy("all-model\\ln2b.npy", h_model.GetParams().ln2b, 1, new unsigned long[1]{ C });
#endif

	int deviceIdx = 0;
	cudaCheck(cudaSetDevice(deviceIdx));

	// move to GPU
	//ModelMemoryHandler<float> d_model = h_model.model_to_cuda();
	ModelMemoryHandler<float> d_model;
	model_to_cuda(&d_model, &h_model);

	// move input and target to GPU
	float* d_inp;
	int* d_target;
	cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(float)));
	cudaCheck(cudaMalloc(&d_target, B * sizeof(int)));
	cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_target, target, B * sizeof(int), cudaMemcpyHostToDevice));

	// run the model
	ModelLayersKernelsLaunchers::linear_layer(d_inp, d_model.GetParams().ln1w, d_model.GetParams().ln1b, d_model.GetActivations().ln1, B, input_dim, H1, 32);
	ModelLayersKernelsLaunchers::run_relu_kernel(d_model.GetActivations().ln1, d_model.GetActivations().a1, B, H1, 32);
	ModelLayersKernelsLaunchers::linear_layer(d_model.GetActivations().a1, d_model.GetParams().ln2w, d_model.GetParams().ln2b, d_model.GetActivations().ln2, B, H1, C, 32);
	ModelLayersKernelsLaunchers::run_softmax_kernel(d_model.GetActivations().ln2, d_model.GetActivations().sm, B, C, 32);
	ModelLayersKernelsLaunchers::run_cross_entropy_kernel(d_model.GetActivations().loss, d_model.GetActivations().sm, d_target, B, C, 32);
	ModelLayersKernelsLaunchers::run_array_sum_kernel3(d_model.GetActivations().loss, d_model.GetActivations().reduced_loss, B, 32);

	// cudasynchronize();
	cudaCheck(cudaDeviceSynchronize());
	// copy the loss to the host
	float* reduced_loss = (float*)malloc(sizeof(float));
	cudaCheck(cudaMemcpy(reduced_loss, d_model.GetActivations().reduced_loss, sizeof(float), cudaMemcpyDeviceToHost));
	printf("Loss: %f\n", *reduced_loss);
}