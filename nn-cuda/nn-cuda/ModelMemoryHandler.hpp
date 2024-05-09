#pragma once
#define NUM_PARAMETER_ARRAYS 4
#define NUM_ACTIVATION_ARRAYS 6

enum INITIAL_VALUE_TYPE
{
	ZEROS_V,
	ONES_V,
	RANDOM_V
};
template<class T>
struct ModelParameters
{
	/* data */
	T* ln1w; // linear layer 1 weights (H1 x N)
	T* ln1b; // linear layer 1 bias (H1)
	T* ln2w; // linear layer 2 weights (C x H1)
	T* ln2b; // linear layer 2 bias (C)
};

/*
 * @brief
 * This will include the model activations like the output of each layer
 */
template<class T>
struct ModelActivation
{
	/* data */
	T* ln1;			 // linear layer 1 output (B x H1)
	T* a1;			 // activation 1 output (B x H1)
	T* ln2;			 // linear layer 2 output (B x C) -- C is the number of classes = C
	T* sm;			 // softmax output (B x C)
	T* loss;		 // loss (B)
	T* reduced_loss; // reduced loss (1)
};

template<class T>
class ModelMemoryHandler
{
private:
	unsigned long param_sizes[NUM_PARAMETER_ARRAYS];
	ModelParameters<T> params;
	T* params_memory;

	unsigned long activation_sizes[NUM_ACTIVATION_ARRAYS];
	ModelActivation<T> activations;
	T* activations_memory;

	void InitializeModelParametersSizes(unsigned long input_dim, unsigned long H1, unsigned long C)
	{
		param_sizes[0] = H1 * input_dim; // ln1w
		param_sizes[1] = H1;			 // ln1b
		param_sizes[2] = C * H1;		 // ln2w
		param_sizes[3] = C;				 // ln2b
	}
	void InitializeModelActivationSizes(unsigned long B, unsigned long H1, unsigned long C)
	{
		activation_sizes[0] = B * H1; // ln1
		activation_sizes[1] = B * H1; // a1
		activation_sizes[2] = B * C;  // ln2
		activation_sizes[3] = B * C;  // sm
		activation_sizes[4] = B;	  // loss
		activation_sizes[5] = 1;	  // reduced_loss
	}
	bool InitParametersMemory(INITIAL_VALUE_TYPE initial_value)
	{
		unsigned long total_size = 0;
		for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
		{
			total_size += param_sizes[i];
		}
		T* memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

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
	bool InitActivationsMemory(INITIAL_VALUE_TYPE initial_value)
	{
		unsigned long total_size = 0;
		for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
		{
			total_size += activation_sizes[i];
		}
		T* memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

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

public:
	ModelMemoryHandler(unsigned long input_dim = 3, unsigned long B = 30, unsigned long H1 = 100, unsigned long C = 10, INITIAL_VALUE_TYPE PARAMETERS_INIT = ZEROS_V, INITIAL_VALUE_TYPE ACTIVATION_INIT = ZEROS_V)
	{
		InitializeModelParametersSizes(input_dim, H1, C);
		InitializeModelActivationSizes(B, H1, C);
		InitParametersMemory(PARAMETERS_INIT);
		AssignParamsMemory();
		InitActivationsMemory(ACTIVATION_INIT);
		AssignActivationsMemory();
	}
	ModelParameters<T> GetParams() { return params; }
	ModelActivation<T> GetActivations() {return activations;}
	void AssignParamsMemory()
	{
		T** ptrs[] = { &params.ln1w,
					  &params.ln1b,
					  &params.ln2w,
					  &params.ln2b };
		T* memory_iterator = params_memory;
		for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
		{
			*ptrs[i] = memory_iterator;
			memory_iterator += param_sizes[i];
		}
	}
	void AssignActivationsMemory()
	{
		T** ptrs[] = { &activations.ln1,
					  &activations.a1,
					  &activations.ln2,
					  &activations.sm,
					  &activations.loss,
					  &activations.reduced_loss };
		T* memory_iterator = activations_memory;
		for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
		{
			*(ptrs[i]) = memory_iterator;
			memory_iterator += activation_sizes[i];
		}
	}
	~ModelMemoryHandler()
	{
		free(ModelMemoryHandler::params_memory);
		free(ModelMemoryHandler::activations_memory);
	}
};