#include "ModelLayersKernelsLaunchers.cuh"
#include "ModelMemoryHandler.cuh"
#include "common.hpp"
#include "device_launch_parameters.h"
#include <math.h>
#define TEST_PYTORTH true
#define TYPE float
int main() {
  unsigned long input_dim = 3;
  unsigned long B = 2;
  unsigned long H1 = 5;
  unsigned long C = 3;
  ModelMemoryHandler<TYPE> h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

  // create host memory of random numbers
  TYPE *inp = make_random_float<TYPE>(B * input_dim);
  int *target = make_random_int(B, int(C));

#if TEST_PYTORTH
  write_npy("all-model\\X_c.npy", inp, 2, new unsigned long[2]{B, input_dim});
  write_npy("all-model\\target.npy", target, 1, new unsigned long[1]{B});
  write_npy("all-model\\ln1w.npy", h_model.GetParams().ln1w, 2,
            new unsigned long[2]{H1, input_dim});
  write_npy("all-model\\ln1b.npy", h_model.GetParams().ln1b, 1,
            new unsigned long[1]{H1});
  write_npy("all-model\\ln2w.npy", h_model.GetParams().ln2w, 2,
            new unsigned long[2]{C, H1});
  write_npy("all-model\\ln2b.npy", h_model.GetParams().ln2b, 1,
            new unsigned long[1]{C});
#endif

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // move to GPU
  ModelMemoryHandler<TYPE> d_model;
  h_model.model_to_cuda(&d_model);

  // move input and target to GPU
  TYPE *d_inp;
  int *d_target;
  cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(TYPE)));
  cudaCheck(cudaMalloc(&d_target, B * sizeof(int)));
  cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(TYPE),
                       cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_target, target, B * sizeof(int), cudaMemcpyHostToDevice));

  // run the model
  ModelLayersKernelsLaunchers::linear_layer(
      d_inp, d_model.GetParams().ln1w, d_model.GetParams().ln1b,
      d_model.GetActivations().ln1, B, input_dim, H1, 32);
  ModelLayersKernelsLaunchers::run_relu_kernel(
      d_model.GetActivations().ln1, d_model.GetActivations().a1, B, H1, 32);
  ModelLayersKernelsLaunchers::linear_layer(
      d_model.GetActivations().a1, d_model.GetParams().ln2w,
      d_model.GetParams().ln2b, d_model.GetActivations().ln2, B, H1, C, 32);
  ModelLayersKernelsLaunchers::run_softmax_kernel(
      d_model.GetActivations().ln2, d_model.GetActivations().sm, B, C, 32);
  ModelLayersKernelsLaunchers::run_cross_entropy_kernel(
      d_model.GetActivations().loss, d_model.GetActivations().sm, d_target, B,
      C, 32);
  ModelLayersKernelsLaunchers::run_array_sum_kernel3(
      d_model.GetActivations().loss, d_model.GetActivations().reduced_loss, B,
      32);

  // cudasynchronize();
  cudaCheck(cudaDeviceSynchronize());
  // copy the loss to the host
  TYPE *reduced_loss = (TYPE *)malloc(sizeof(TYPE));
  cudaCheck(cudaMemcpy(reduced_loss, d_model.GetActivations().reduced_loss,
                       sizeof(TYPE), cudaMemcpyDeviceToHost));
  printf("Loss: %f\n", *reduced_loss);

  return 0;
}