#pragma once
#define NUM_PARAMETER_ARRAYS 4
#define NUM_ACTIVATION_ARRAYS 6

enum INITIAL_VALUE_TYPE
{
	ZEROS_V,
	ONES_V,
	RANDOM_V
};

struct ModelParameters
{
	/* data */
	float *ln1w; // linear layer 1 weights (H1 x N)
	float *ln1b; // linear layer 1 bias (H1)
	float *ln2w; // linear layer 2 weights (C x H1)
	float *ln2b; // linear layer 2 bias (C)
};

/*
 * @brief
 * This will include the model activations like the output of each layer
 */
struct ModelActivation
{
	/* data */
	float *ln1;			 // linear layer 1 output (B x H1)
	float *a1;			 // activation 1 output (B x H1)
	float *ln2;			 // linear layer 2 output (B x C) -- C is the number of classes = C
	float *sm;			 // softmax output (B x C)
	float *loss;		 // loss (B)
	float *reduced_loss; // reduced loss (1)
};

class ModelMemoryHandler
{
private:
	unsigned long param_sizes[NUM_PARAMETER_ARRAYS];
	ModelParameters params;
	float *params_memory;

	unsigned long activation_sizes[NUM_ACTIVATION_ARRAYS];
	ModelActivation activations;
	float *activations_memory;

	void InitializeModelParametersSizes(unsigned long input_dim, unsigned long H1, unsigned long C);
	void InitializeModelActivationSizes(unsigned long B, unsigned long H1, unsigned long C);
	bool InitParametersMemory(INITIAL_VALUE_TYPE initial_value);
	bool InitActivationsMemory(INITIAL_VALUE_TYPE initial_value);

public:
	ModelMemoryHandler(unsigned long input_dim = 3, unsigned long B = 30, unsigned long H1 = 100, unsigned long C = 10, INITIAL_VALUE_TYPE PARAMETERS_INIT = ZEROS_V, INITIAL_VALUE_TYPE ACTIVATION_INIT = ZEROS_V);
	ModelParameters GetParams();
	ModelActivation GetActivations();
	void AssignParamsMemory();
	void AssignActivationsMemory();
	~ModelMemoryHandler();
};