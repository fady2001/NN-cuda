#include "ModelMemoryHandler.hpp"
#include "common.cuh"

void ModelMemoryHandler::InitializeModelParametersSizes(unsigned long input_dim, unsigned long H1, unsigned long C)
{
	param_sizes[0] = H1 * input_dim; // ln1w
	param_sizes[1] = H1;			 // ln1b
	param_sizes[2] = C * H1;		 // ln2w
	param_sizes[3] = C;				 // ln2b
}

void ModelMemoryHandler::InitializeModelActivationSizes(unsigned long B, unsigned long H1, unsigned long C)
{
	activation_sizes[0] = B * H1; // ln1
	activation_sizes[1] = B * H1; // a1
	activation_sizes[2] = B * C;  // ln2
	activation_sizes[3] = B * C;  // sm
	activation_sizes[4] = B;	  // loss
	activation_sizes[5] = 1;	  // reduced_loss
}

bool ModelMemoryHandler::InitParametersMemory(INITIAL_VALUE_TYPE initial_value)
{
	unsigned long total_size = 0;
	for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
	{
		total_size += param_sizes[i];
	}
	float* memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

	switch (initial_value)
	{
	case ZEROS_V:
		memory = make_zeros_float(total_size);
		break;
	case ONES_V:
		memory = make_ones_float(total_size);
		break;
	case RANDOM_V:
		memory = make_random_float(total_size);
		break;
	}
	if (memory == nullptr)
	{
		// Handle allocation failure
		return false;
	}
	params_memory = memory;
	return true;
}

bool ModelMemoryHandler::InitActivationsMemory(INITIAL_VALUE_TYPE initial_value)
{
	unsigned long total_size = 0;
	for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
	{
		total_size += activation_sizes[i];
	}
	float* memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

	switch (initial_value)
	{
	case ZEROS_V:
		memory = make_zeros_float(total_size);
		break;
	case ONES_V:
		memory = make_ones_float(total_size);
		break;
	case RANDOM_V:
		memory = make_random_float(total_size);
		break;
	}
	if (memory == nullptr)
	{
		// Handle allocation failure
		return false;
	}
	activations_memory = memory;
	return true;
}

ModelMemoryHandler::ModelMemoryHandler(unsigned long input_dim, unsigned long B, unsigned long H1, unsigned long C, INITIAL_VALUE_TYPE PARAMETERS_INIT, INITIAL_VALUE_TYPE ACTIVATION_INIT)
{
	InitializeModelParametersSizes(input_dim, H1, C);
	InitializeModelActivationSizes(B, H1, C);
	InitParametersMemory(PARAMETERS_INIT);
	AssignParamsMemory();
	InitActivationsMemory(ACTIVATION_INIT);
	AssignActivationsMemory();
}

ModelParameters ModelMemoryHandler::GetParams()
{
	return params;
}

ModelActivation ModelMemoryHandler::GetActivations()
{
	return activations;
}

void ModelMemoryHandler::AssignParamsMemory()
{
	float** ptrs[] = {&params.ln1w,
					  &params.ln1b,
					  &params.ln2w,
					  &params.ln2b };
	float* memory_iterator = params_memory;
	for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
	{
		*ptrs[i] = memory_iterator;
		memory_iterator += param_sizes[i];
	}
}

void ModelMemoryHandler::AssignActivationsMemory()
{
	float** ptrs[] = { &activations.ln1,
					  &activations.a1,
					  &activations.ln2,
					  &activations.sm,
					  &activations.loss,
					  &activations.reduced_loss };
	float* memory_iterator = activations_memory;
	for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
	{
		*(ptrs[i]) = memory_iterator;
		memory_iterator += activation_sizes[i];
	}
}

ModelMemoryHandler::~ModelMemoryHandler()
{
	free(ModelMemoryHandler::params_memory);
	free(ModelMemoryHandler::activations_memory);
}
