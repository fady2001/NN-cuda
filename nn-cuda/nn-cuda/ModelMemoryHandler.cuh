#pragma once
#include "common.hpp"
#define NUM_PARAMETER_ARRAYS 4
#define NUM_ACTIVATION_ARRAYS 6
#define NUM_GRADIENT_ARRAYS 4
#define NUM_DOWNSTREAM_GRADIENT_ARRAYS 4

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

struct ParametersGradients
{
  /* data */
  float *ln1w_grad; // linear layer 1 weights gradient (H1 x N)
  float *ln1b_grad; // linear layer 1 bias gradient (H1)
  float *ln2w_grad; // linear layer 2 weights gradient (C x H1)
  float *ln2b_grad; // linear layer 2 bias gradient (C)
};

/*
 * @brief
 * This will include the model activations like the output of each layer
 */
struct ModelActivation
{
  /* data */
  float *ln1;          // linear layer 1 output (B x H1)
  float *a1;           // activation 1 output (B x H1)
  float *ln2;          // linear layer 2 output (B x C) -- C is the number of classes = C
  float *sm;           // softmax output (B x C)
  float *loss;         // loss (B)
  float *reduced_loss; // reduced loss (1)
};

struct downstreamGradients
{
  float *dln1; // no need for it (B x N)
  float *da1;  // from activation (B x H1)
  float *dln2; // from ln2 (B x H1)
  float *dsm;  // from softmax (B x C)
};

class ModelMemoryHandler
{
private:
  uint param_sizes[NUM_PARAMETER_ARRAYS]{};
  ModelParameters params{};
  float *params_memory{};

  uint activation_sizes[NUM_ACTIVATION_ARRAYS]{};
  ModelActivation activations{};
  float *activations_memory{};

  uint gradient_sizes[NUM_GRADIENT_ARRAYS]{};
  ParametersGradients gradients{};
  float *gradients_memory{};

  uint downstream_gradient_sizes[NUM_DOWNSTREAM_GRADIENT_ARRAYS]{};
  downstreamGradients downstream_gradients{};
  float *downstream_gradients_memory{};

  bool isCuda = false;

  void InitializeModelParametersSizes(uint input_dim, uint H1, uint C)
  {
    param_sizes[0] = H1 * input_dim; // ln1w
    param_sizes[1] = H1;             // ln1b
    param_sizes[2] = C * H1;         // ln2w
    param_sizes[3] = C;              // ln2b
  }

  uint InitializeModelParametersSizes(ModelMemoryHandler *h_model)
  {
    uint total_param_size = 0;
    for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
    {
      total_param_size += h_model->param_sizes[i];
      param_sizes[i] = h_model->param_sizes[i];
    }
    return total_param_size;
  }

  void InitializeParameterGradientsSizes(uint input_dim, uint H1, uint C)
  {
    gradient_sizes[0] = H1 * input_dim; // ln1w gradients
    gradient_sizes[1] = H1;             // ln1b gradients
    gradient_sizes[2] = C * H1;         // ln2w gradients
    gradient_sizes[3] = C;              // ln2b gradients
  }

  uint InitializeParameterGradientsSizes(ModelMemoryHandler *h_model)
  {
    uint total_param_size = 0;
    for (int i = 0; i < NUM_GRADIENT_ARRAYS; i++)
    {
      total_param_size += h_model->gradient_sizes[i];
      gradient_sizes[i] = h_model->gradient_sizes[i];
    }
    return total_param_size;
  }

  void InitializeModelActivationSizes(uint B, uint H1, uint C)
  {
    activation_sizes[0] = B * H1; // ln1
    activation_sizes[1] = B * H1; // a1
    activation_sizes[2] = B * C;  // ln2
    activation_sizes[3] = B * C;  // sm
    activation_sizes[4] = B;      // loss
    activation_sizes[5] = 1;      // reduced_loss
  }

  uint InitializeModelActivationSizes(ModelMemoryHandler *h_model)
  {
    uint total_activation_size = 0;
    for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
    {
      total_activation_size += h_model->activation_sizes[i];
      activation_sizes[i] = h_model->activation_sizes[i];
    }
    return total_activation_size;
  }

  void InitializeDownstreamGradientSizes(uint input_dim, uint B, uint H1, uint C)
  {
    downstream_gradient_sizes[0] = B * input_dim; // dln1
    downstream_gradient_sizes[1] = B * H1;        // da1
    downstream_gradient_sizes[2] = B * H1;        // dln2
    downstream_gradient_sizes[3] = B * C;         // dsm
  }

  uint InitializeDownstreamGradientSizes(ModelMemoryHandler *h_model)
  {
    uint total_downstream_gradient_size = 0;
    for (int i = 0; i < NUM_DOWNSTREAM_GRADIENT_ARRAYS; i++)
    {
      total_downstream_gradient_size += h_model->downstream_gradient_sizes[i];
      downstream_gradient_sizes[i] = h_model->downstream_gradient_sizes[i];
    }
    return total_downstream_gradient_size;
  }

  float *InitMemory(INITIAL_VALUE_TYPE initial_value, uint *sizes, int num_arrays)
  {
    uint total_size = 0;

    for (int i = 0; i < num_arrays; i++)
    {
      total_size += sizes[i];
    }

    float *memory = nullptr;

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
      perror("Memory allocation failed\n");
      exit(EXIT_FAILURE); // Or return false if you prefer non-terminating error handling
    }

    // Assign the memory to the appropriate global pointer
    return memory; // Optionally return the allocated memory instead of true
  }

  void AssignParamsMemory()
  {
    float **ptrs[] = {&params.ln1w, &params.ln1b, &params.ln2w, &params.ln2b};
    float *memory_iterator = params_memory;
    for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++)
    {
      *ptrs[i] = memory_iterator;
      memory_iterator += param_sizes[i];
    }
  }

  void AssignActivationsMemory()
  {
    float **ptrs[] = {&activations.ln1, &activations.a1,
                      &activations.ln2, &activations.sm,
                      &activations.loss, &activations.reduced_loss};
    float *memory_iterator = activations_memory;
    for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++)
    {
      *(ptrs[i]) = memory_iterator;
      memory_iterator += activation_sizes[i];
    }
  }

  void AssignParamsGradientsMemory()
  {
    float **ptrs[] = {&gradients.ln1w_grad, &gradients.ln1b_grad,
                      &gradients.ln2w_grad, &gradients.ln2b_grad};
    float *memory_iterator = gradients_memory;
    for (int i = 0; i < NUM_GRADIENT_ARRAYS; i++)
    {
      *(ptrs[i]) = memory_iterator;
      memory_iterator += gradient_sizes[i];
    }
  }

  void AssignDownstreamGradientsMemory()
  {
    float **ptrs[] = {&downstream_gradients.dln1, &downstream_gradients.da1,
                      &downstream_gradients.dln2, &downstream_gradients.dsm};
    float *memory_iterator = downstream_gradients_memory;
    for (int i = 0; i < NUM_DOWNSTREAM_GRADIENT_ARRAYS; i++)
    {
      *(ptrs[i]) = memory_iterator;
      memory_iterator += downstream_gradient_sizes[i];
    }
  }

public:
  ModelMemoryHandler() {}

  ModelMemoryHandler(uint input_dim, uint B, uint H1, uint C, INITIAL_VALUE_TYPE PARAMETERS_INIT = ZEROS_V, INITIAL_VALUE_TYPE ACTIVATION_INIT = ZEROS_V)
  {
    // params memory
    InitializeModelParametersSizes(input_dim, H1, C);
    params_memory = InitMemory(PARAMETERS_INIT, param_sizes, NUM_PARAMETER_ARRAYS);
    AssignParamsMemory();

    // activation memory
    InitializeModelActivationSizes(B, H1, C);
    activations_memory = InitMemory(ACTIVATION_INIT, activation_sizes, NUM_ACTIVATION_ARRAYS);
    AssignActivationsMemory();

    // Gradients
    InitializeParameterGradientsSizes(input_dim, H1, C);
    gradients_memory = InitMemory(PARAMETERS_INIT, gradient_sizes, NUM_GRADIENT_ARRAYS);
    AssignParamsGradientsMemory();

    // Downstream Gradients
    InitializeDownstreamGradientSizes(input_dim, B, H1, C);
    downstream_gradients_memory = InitMemory(PARAMETERS_INIT, downstream_gradient_sizes, NUM_DOWNSTREAM_GRADIENT_ARRAYS);
    AssignDownstreamGradientsMemory();
  }
  float *GetParamsMemory() { return params_memory; }
  float *GetActivationsMemory() { return activations_memory; }
  float *GetGradientsMemory() { return gradients_memory; }
  float *GetDownstreamGradientsMemory() { return downstream_gradients_memory; }

  uint get_num_parameters()
  {
    return param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3];
  }

  ModelParameters GetParams() { return params; }

  ModelActivation GetActivations() { return activations; }

  ParametersGradients GetGradients() { return gradients; }

  downstreamGradients GetDownstreamGradients() { return downstream_gradients; }

  void model_to_cuda(ModelMemoryHandler *d_model)
  {
    d_model->isCuda = true;

    uint total_param_size = d_model->InitializeModelParametersSizes(this);
    cudaCheck(cudaMalloc(&d_model->params_memory, total_param_size * sizeof(float)));
    cudaCheck(cudaMemcpy(d_model->params_memory, this->params_memory, total_param_size * sizeof(float), cudaMemcpyHostToDevice));
    d_model->AssignParamsMemory();

    // copy activations
    uint total_activation_size = d_model->InitializeModelActivationSizes(this);
    cudaCheck(cudaMalloc(&d_model->activations_memory, total_activation_size * sizeof(float)));
    cudaCheck(cudaMemcpy(d_model->activations_memory, this->activations_memory, total_activation_size * sizeof(float), cudaMemcpyHostToDevice));
    d_model->AssignActivationsMemory();

    // memory for gradients
    uint total_gradient_size = d_model->InitializeParameterGradientsSizes(this);
    cudaCheck(cudaMalloc(&d_model->gradients_memory, total_gradient_size * sizeof(float)));
    d_model->AssignParamsGradientsMemory();

    // memory for downstream gradients
    uint total_downstream_gradient_size = d_model->InitializeDownstreamGradientSizes(this);
    cudaCheck(cudaMalloc(&d_model->downstream_gradients_memory, total_downstream_gradient_size * sizeof(float)));
    d_model->AssignDownstreamGradientsMemory();
  }

  ~ModelMemoryHandler()
  {
    if (isCuda)
    {
      cudaFree(params_memory);
      cudaFree(activations_memory);
      cudaFree(gradients_memory);
      cudaFree(downstream_gradients_memory);
    }
    else
    {
      free(params_memory);
      free(activations_memory);
      free(gradients_memory);
      free(downstream_gradients_memory);
    }
  }
};