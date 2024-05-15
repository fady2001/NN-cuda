#pragma once
#include "common.hpp"
#define NUM_PARAMETER_ARRAYS 4
#define NUM_ACTIVATION_ARRAYS 6

enum INITIAL_VALUE_TYPE { ZEROS_V, ONES_V, RANDOM_V };
struct ModelParameters {
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
struct ModelActivation {
  /* data */
  float *ln1; // linear layer 1 output (B x H1)
  float *a1;  // activation 1 output (B x H1)
  float *ln2; // linear layer 2 output (B x C) -- C is the number of classes = C
  float *sm;  // softmax output (B x C)
  float *loss;         // loss (B)
  float *reduced_loss; // reduced loss (1)
};

class ModelMemoryHandler {
public:
  unsigned long param_sizes[NUM_PARAMETER_ARRAYS];
  ModelParameters params;
  float *params_memory;

  unsigned long activation_sizes[NUM_ACTIVATION_ARRAYS];
  ModelActivation activations;
  float *activations_memory;

  bool isCuda;

  void InitializeModelParametersSizes(unsigned long input_dim, unsigned long H1,
                                      unsigned long C) {
    param_sizes[0] = H1 * input_dim; // ln1w
    param_sizes[1] = H1;             // ln1b
    param_sizes[2] = C * H1;         // ln2w
    param_sizes[3] = C;              // ln2b
  }

  unsigned long InitializeModelParametersSizes(ModelMemoryHandler *h_model) {
    unsigned long total_param_size = 0;
    for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++) {
      total_param_size += h_model->param_sizes[i];
      param_sizes[i] = h_model->param_sizes[i];
    }
    return total_param_size;
  }

  void InitializeModelActivationSizes(unsigned long B, unsigned long H1,
                                      unsigned long C) {
    activation_sizes[0] = B * H1; // ln1
    activation_sizes[1] = B * H1; // a1
    activation_sizes[2] = B * C;  // ln2
    activation_sizes[3] = B * C;  // sm
    activation_sizes[4] = B;      // loss
    activation_sizes[5] = 1;      // reduced_loss
  }

  unsigned long InitializeModelActivationSizes(ModelMemoryHandler *h_model) {
    unsigned long total_activation_size = 0;
    for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++) {
      total_activation_size += h_model->activation_sizes[i];
      activation_sizes[i] = h_model->activation_sizes[i];
    }
    return total_activation_size;
  }

  bool InitParametersMemory(INITIAL_VALUE_TYPE initial_value) {
    unsigned long total_size = 0;
    for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++) {
      total_size += param_sizes[i];
    }
    float *memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

    switch (initial_value) {
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
    if (memory == nullptr) {
      // Handle allocation failure
      return false;
    }
    params_memory = memory;
    return true;
  }

  bool InitActivationsMemory(INITIAL_VALUE_TYPE initial_value) {
    unsigned long total_size = 0;
    for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++) {
      total_size += activation_sizes[i];
    }
    float *memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

    switch (initial_value) {
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
    if (memory == nullptr) {
      // Handle allocation failure
      return false;
    }
    activations_memory = memory;
    return true;
  }

  // public:
  ModelMemoryHandler() { isCuda = false; }

  ModelMemoryHandler(unsigned long input_dim, unsigned long B, unsigned long H1,
                     unsigned long C,
                     INITIAL_VALUE_TYPE PARAMETERS_INIT = ZEROS_V,
                     INITIAL_VALUE_TYPE ACTIVATION_INIT = ZEROS_V) {
    isCuda = false; // default
    InitializeModelParametersSizes(input_dim, H1, C);
    InitializeModelActivationSizes(B, H1, C);
    InitParametersMemory(PARAMETERS_INIT);
    AssignParamsMemory();
    InitActivationsMemory(ACTIVATION_INIT);
    AssignActivationsMemory();
  }

  ModelParameters GetParams() { return params; }

  ModelActivation GetActivations() { return activations; }

  void AssignParamsMemory() {
    float **ptrs[] = {&params.ln1w, &params.ln1b, &params.ln2w, &params.ln2b};
    float *memory_iterator = params_memory;
    for (int i = 0; i < NUM_PARAMETER_ARRAYS; i++) {
      *ptrs[i] = memory_iterator;
      memory_iterator += param_sizes[i];
    }
  }

  void AssignActivationsMemory() {
    float **ptrs[] = {&activations.ln1,  &activations.a1,
                      &activations.ln2,  &activations.sm,
                      &activations.loss, &activations.reduced_loss};
    float *memory_iterator = activations_memory;
    for (int i = 0; i < NUM_ACTIVATION_ARRAYS; i++) {
      *(ptrs[i]) = memory_iterator;
      memory_iterator += activation_sizes[i];
    }
  }

  void model_to_cuda(ModelMemoryHandler *d_model) {
    d_model->isCuda = true;
    unsigned long total_param_size =
        d_model->InitializeModelParametersSizes(this);

    cudaCheck(
        cudaMalloc(&d_model->params_memory, total_param_size * sizeof(float)));
    cudaCheck(cudaMemcpy(d_model->params_memory, this->params_memory,
                         total_param_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    d_model->AssignParamsMemory();

    // copy activations
    unsigned long total_activation_size =
        d_model->InitializeModelActivationSizes(this);

    cudaCheck(cudaMalloc(&d_model->activations_memory,
                         total_activation_size * sizeof(float)));
    cudaCheck(cudaMemcpy(d_model->activations_memory, this->activations_memory,
                         total_activation_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    d_model->AssignActivationsMemory();
  }

  ~ModelMemoryHandler() {
    if (isCuda) {
      cudaFree(params_memory);
      cudaFree(activations_memory);
    } else {
      free(params_memory);
      free(activations_memory);
    }
  }
};