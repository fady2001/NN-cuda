#include "ModelMemoryHandler.cuh"
#include "common.hpp"
#include "kernels_launchers.cuh"
#include <cmath>
#define TEST_PYTORTH true

int main() {
  uint input_dim = 200;
  uint B = 320;
  uint H1 = 2150;
  uint C = 14;
  ModelMemoryHandler h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

  // create host memory of random numbers
  float *inp = make_random_float(B * input_dim);
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
  ModelMemoryHandler d_model;
  h_model.model_to_cuda(&d_model);

  // move input and target to GPU
  float *d_inp;
  uint *d_target;
  cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(float)));
  cudaCheck(cudaMalloc(&d_target, B * sizeof(uint)));
  cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(float),
                       cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_target, target, B * sizeof(int), cudaMemcpyHostToDevice));

  // run the model
  KernelsLaunchers::linear_layer(
      d_inp, d_model.GetParams().ln1w, d_model.GetParams().ln1b,
      d_model.GetActivations().ln1, B, input_dim, H1, 32);
  save_2d("all-model\\ln1.npy", d_model.GetActivations().ln1, B, H1);
  KernelsLaunchers::run_relu_kernel(d_model.GetActivations().ln1,
                                    d_model.GetActivations().a1, B, H1, 32);
  save_2d("all-model\\a1.npy", d_model.GetActivations().a1, B, H1);
  KernelsLaunchers::linear_layer(
      d_model.GetActivations().a1, d_model.GetParams().ln2w,
      d_model.GetParams().ln2b, d_model.GetActivations().ln2, B, H1, C, 32);
  save_2d("all-model\\ln2.npy", d_model.GetActivations().ln2, B, C);
  KernelsLaunchers::run_softmax_kernel(d_model.GetActivations().ln2,
                                       d_model.GetActivations().sm, B, C, 32);
  save_2d("all-model\\sm.npy", d_model.GetActivations().sm, B, C);
  KernelsLaunchers::run_cross_entropy_kernel(d_model.GetActivations().loss,
                                             d_model.GetActivations().sm,
                                             d_target, B, C, 32);
  save_1d("all-model\\loss.npy", d_model.GetActivations().loss, B);
  KernelsLaunchers::run_reduce_kernel3(d_model.GetActivations().loss,
                                       d_model.GetActivations().reduced_loss, B,
                                       REDUCTION::MEAN, 32);

  // cuda synchronize();
  cudaCheck(cudaDeviceSynchronize());
  // copy the loss to the host
  float *reduced_loss = (float *)malloc(sizeof(float));
  cudaCheck(cudaMemcpy(reduced_loss, d_model.GetActivations().reduced_loss,
                       sizeof(float), cudaMemcpyDeviceToHost));
  printf("Loss: %f\n", *reduced_loss);

  // backpropagation
  KernelsLaunchers::run_crossentropy_softmax_backward(
      d_model.GetDownstreamGradients().dsm, d_model.GetActivations().sm,
      d_target, B, C, 32);
  save_2d("all-model\\dsm.npy", d_model.GetDownstreamGradients().dsm, B, C);
  // get_from_gpu_and_print("dsm", d_model.GetDownstreamGradients().dsm, B * C);

  KernelsLaunchers::runLinearBackward(
      d_model.GetActivations().a1, d_model.GetParams().ln2w,
      d_model.GetDownstreamGradients().dsm, d_model.GetGradients().ln2w_grad,
      d_model.GetGradients().ln2b_grad, d_model.GetDownstreamGradients().dln2,
      B, H1, C, 32);
  save_2d("all-model\\dln2.npy", d_model.GetDownstreamGradients().dln2, B, H1);
  save_2d("all-model\\ln2w_grad.npy", d_model.GetGradients().ln2w_grad, C, H1);
  save_1d("all-model\\ln2b_grad.npy", d_model.GetGradients().ln2b_grad, C);

  // get_from_gpu_and_print("dln2", d_model.GetDownstreamGradients().dln2, B *
  // H1); get_from_gpu_and_print("ln2w_grad", d_model.GetGradients().ln2w_grad,
  // C * H1); get_from_gpu_and_print("ln2b_grad",
  // d_model.GetGradients().ln2b_grad, C);

  KernelsLaunchers::runReluBackward(
      d_model.GetActivations().ln1, d_model.GetDownstreamGradients().dln2,
      d_model.GetDownstreamGradients().da1, B, H1, 32);
  save_2d("all-model\\da1.npy", d_model.GetDownstreamGradients().da1, B, H1);
  // get_from_gpu_and_print("da1", d_model.GetDownstreamGradients().da1, B *
  // H1);
  KernelsLaunchers::runLinearBackward(
      d_inp, d_model.GetParams().ln1w, d_model.GetDownstreamGradients().da1,
      d_model.GetGradients().ln1w_grad, d_model.GetGradients().ln1b_grad,
      d_model.GetDownstreamGradients().dln1, B, input_dim, H1, 32);
  save_2d("all-model\\dln1.npy", d_model.GetDownstreamGradients().dln1, B,
          input_dim);
  save_2d("all-model\\ln1w_grad.npy", d_model.GetGradients().ln1w_grad, H1,
          input_dim);
  save_1d("all-model\\ln1b_grad.npy", d_model.GetGradients().ln1b_grad, H1);
  // get_from_gpu_and_print("dln1", d_model.GetDownstreamGradients().dln1,B *
  // input_dim);

  // Magic optimizer
  KernelsLaunchers::SGD_run_kernel(d_model.GetParamsMemory(),
                                   d_model.GetGradientsMemory(),
                                   d_model.get_num_parameters(), 0.01, 0.0, 32);

#if TEST_PYTORTH
  save_2d("all-model\\updated_ln1w.npy", d_model.GetParams().ln1w, H1,
          input_dim);
  save_1d("all-model\\updated_ln1b.npy", d_model.GetParams().ln1b, H1);
  save_2d("all-model\\updated_ln2w.npy", d_model.GetParams().ln2w, C, H1);
  save_1d("all-model\\updated_ln2b.npy", d_model.GetParams().ln2b, C);
#endif
  return 0;
}
