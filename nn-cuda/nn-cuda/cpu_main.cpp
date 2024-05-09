#include <iostream>
#include <cmath>
#include "common.cuh"
#define TEST_PYTORTH true

void linear_layer_forward_cpu(float *X, float *W, float *bias, float *y, int B, int N, int M)
{

  for (int i = 0; i < B; i++)
  {
      for (int j = 0; j < M; j++)
      {
          y[i * M + j] = bias[j];
          for (int k = 0; k < N; k++)
          {
              y[i * M + j] += X[i * N + k] * W[j * N + k];
          }
      }
  }
}

void relu_forward_cpu(float *input, float *output, int B, int N)
{
   for (int i = 0; i < B; i++)
   {
       for (int j = 0; j < N; j++)
       {
           int idx = i * N + j;
           output[idx] = fmaxf(0.0f, input[idx]);
       }
   }
}

template <class T>
void softmax_cpu(const T *in, T *out, int N, int C)
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

template <class T>
void cross_entropy_cpu(T *losses, const T *input, const int *targets, int N, int C)
{
   // output: losses is (N) of the individual losses for each batch
   // input: input are (N,C) of the probabilities from softmax
   // input: targets is (N) of integers giving the correct index in logits
   for (int i = 0; i < N; i++)
   {
       losses[i] = -log(input[i * C + targets[i]]);
   }
}

template <class T>
void array_sum_cpu(T* out, const T* in, int N) {
   T sum = 0;
   for (int i = 0; i < N; ++i) {
       sum += in[i];
   }
   *out = sum;
}

/*
* @brief
* This will include the model parameters like weights and bias for each layer
*/
#define NUM_PARAMETER_ARRAYS 4
#define NUM_ACTIVATION_ARRAYS 6
typedef struct
{
   float *ln1w; // linear layer 1 weights (H1 x N)
   float *ln1b; // linear layer 1 bias (H1)
   float *ln2w; // linear layer 2 weights (H2 x H1)
   float *ln2b; // linear layer 2 bias (H2)
} ModelParameters;

/*
* @brief
* This will include the model activations like the output of each layer
*/
typedef struct
{
   float *ln1;          // linear layer 1 output (B x H1)
   float *a1;           // activation 1 output (B x H1)
   float *ln2;          // linear layer 2 output (B x H2) -- H2 is the number of classes = C
   float *sm;           // softmax output (B x C)
   float *loss;         // loss (B)
   float *reduced_loss; // reduced loss (1)
} ModelActivation;

typedef struct
{
   unsigned long param_sizes[NUM_PARAMETER_ARRAYS];
   ModelParameters params;
   float *params_memory;

   unsigned long activation_sizes[NUM_ACTIVATION_ARRAYS];
   ModelActivation activations;
   float *activations_memory;
} TwoLayerModel;

typedef enum
{
   PARAMETERS_TYPE,
   ACTIVATIONS_TYPE
} DataType;

typedef enum
{
   ZEROS_V,
   ONES_V,
   RANDOM_V
} INITIAL_VALUE_TYPE;

float *float_cpu_malloc_and_point(void *data, unsigned long *sizes, int num_arrays, DataType type, INITIAL_VALUE_TYPE initial_value = ZEROS_V)
{
   unsigned long total_size = 0;
   for (int i = 0; i < num_arrays; i++)
   {
       total_size += sizes[i];
   }
   float *memory = nullptr; // = (float*)malloc(total_size * sizeof(float));

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
   if (memory == NULL)
   {
       // Handle allocation failure
       return NULL;
   }
   switch (type)
   {
   case PARAMETERS_TYPE: // ModelParameters
   {
       ModelParameters *params = (ModelParameters *)data;
       float **ptrs[] = {&params->ln1w,
                         &params->ln1b,
                         &params->ln2w,
                         &params->ln2b};
       float *memory_iterator = memory;
       for (int i = 0; i < num_arrays; i++)
       {
           *(ptrs[i]) = memory_iterator;
           memory_iterator += sizes[i];
       }
   break;
   }
   case ACTIVATIONS_TYPE: // ModelActivation
   {
       ModelActivation *activations = (ModelActivation *)data;
       float **ptrs[] = {&activations->ln1,
                         &activations->a1,
                         &activations->ln2,
                         &activations->sm,
                         &activations->loss,
                         &activations->reduced_loss};
       float *memory_iterator = memory;
       for (int i = 0; i < num_arrays; i++)
       {
           *(ptrs[i]) = memory_iterator;
           memory_iterator += sizes[i];
       }
   break;
   }
   }
   return memory;
}


void clean_model_cpu(TwoLayerModel *model)
{
   free(model->params_memory);
   free(model->activations_memory);
}

int main()
{
   unsigned long input_dim = 3;
   unsigned long B = 2;
   unsigned long H1 = 3;
   unsigned long H2 = 5;
   unsigned long C = 3;
   TwoLayerModel model;

   model.param_sizes[0] = H1 * input_dim; // ln1w
   model.param_sizes[1] = H1;             // ln1b
   model.param_sizes[2] = H2 * H1;        // ln2w
   model.param_sizes[3] = H2;             // ln2b
   model.params_memory = float_cpu_malloc_and_point(&(model.params), model.param_sizes, NUM_PARAMETER_ARRAYS, PARAMETERS_TYPE, RANDOM_V);

   if (model.params_memory == NULL)
   {
       // Handle allocation failure
       printf("Allocation failure\n");
       return 1;
   }
   // Now Activations
   model.activation_sizes[0] = B * H1; // ln1
   model.activation_sizes[1] = B * H1; // a1
   model.activation_sizes[2] = B * H2; // ln2
   model.activation_sizes[3] = B * C;  // sm
   model.activation_sizes[4] = B;      // loss
   model.activation_sizes[5] = 1;      // reduced_loss
   model.activations_memory = float_cpu_malloc_and_point(&model.activations, model.activation_sizes, NUM_ACTIVATION_ARRAYS, ACTIVATIONS_TYPE,ZEROS_V);
   if (model.activations_memory == NULL)
   {
	   // Handle allocation failure
	   printf("Allocation failure\n");
	   return 1;
   }
   // create host memory of random numbers
   float *inp = make_random_float(B * input_dim);
   int *target = make_random_int(B, int(C));

#if TEST_PYTORTH
   write_npy("all-model-cpu\\X_c.npy", inp, 2, new unsigned long[2]{B, input_dim});
   write_npy("all-model-cpu\\target.npy", target, 1, new unsigned long[1]{B});
   write_npy("all-model-cpu\\ln1w.npy", model.params.ln1w, 2, new unsigned long[2]{H1, input_dim});
   write_npy("all-model-cpu\\ln1b.npy", model.params.ln1b, 1, new unsigned long[1]{H1});
   write_npy("all-model-cpu\\ln2w.npy", model.params.ln2w, 2, new unsigned long[2]{H2, H1});
   write_npy("all-model-cpu\\ln2b.npy", model.params.ln2b, 1, new unsigned long[1]{H2});
#endif

// run CPU
    linear_layer_forward_cpu(inp, model.params.ln1w, model.params.ln1b, model.activations.ln1, B, input_dim, H1);
    relu_forward_cpu(model.activations.ln1, model.activations.a1, B, H1);
    linear_layer_forward_cpu(model.activations.a1, model.params.ln2w, model.params.ln2b, model.activations.ln2, B, H1, H2);
    softmax_cpu(model.activations.ln2, model.activations.sm, B, H2);
    cross_entropy_cpu(model.activations.loss, model.activations.sm, target, B, C);
    array_sum_cpu(model.activations.reduced_loss, model.activations.loss, B);

// print results
    printf("Reduced Loss: %f\n", *model.activations.reduced_loss);
	clean_model_cpu(&model);
	return 0;
}