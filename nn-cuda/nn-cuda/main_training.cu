#include "ModelMemoryHandler.cuh"
#include "common.hpp"
#include "kernels_launchers.cuh"
#include <cmath>
#include <vector>

#define TEST_PYTORTH true

// Function to simulate loading a batch of data
void load_batch(trainData &data, float **curr_input,
                unsigned int **current_target, unsigned int B,
                unsigned int batch_index) {

  // Load the batch from the dataset
  //  for (int i = 0; i < B; i++) {
  //    //    load input vector
  //    for (int j = 0; j < data.input_shape[1]; j++) {
  //      curr_input[i * data.input_shape[1] + j] =
  //          data.inp[(batch_index * B + i) * data.input_shape[1] + j];
  //    }
  //    current_target[i] = data.target[batch_index * B + i];
  //  }
  *curr_input = data.inp + batch_index * B * data.input_shape[1];
  *current_target = data.target + batch_index * B;
}

// Function to perform a single training step
void train_step(ModelMemoryHandler &d_model, float *inp, uint *target, uint B,
                uint input_dim, uint H1, uint C, bool my_turn,
                cudaEvent_t events[2], cudaStream_t stream = nullptr) {

  float *d_inp;
  uint *d_target;
  //  cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(float)));
  cudaCheck(cudaMallocAsync(&d_inp, B * input_dim * sizeof(float), stream));
  //  cudaCheck(cudaMalloc(&d_target, B * sizeof(uint)));
  cudaCheck(cudaMallocAsync(&d_target, B * sizeof(uint), stream));

  //  cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_inp, inp, B * input_dim * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
  //  cudaCheck(
  //      cudaMemcpy(d_target, target, B * sizeof(uint),
  //      cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_target, target, B * sizeof(uint),
                            cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaStreamWaitEvent(stream, events[!my_turn]));
  // run the model
  KernelsLaunchers::linear_layer(
      d_inp, d_model.GetParams().ln1w, d_model.GetParams().ln1b,
      d_model.GetActivations().ln1, B, input_dim, H1, 32, stream);
  KernelsLaunchers::run_relu_kernel(d_model.GetActivations().ln1,
                                    d_model.GetActivations().a1, B, H1, 32,
                                    stream);
  KernelsLaunchers::linear_layer(
      d_model.GetActivations().a1, d_model.GetParams().ln2w,
      d_model.GetParams().ln2b, d_model.GetActivations().ln2, B, H1, C, 32,
      stream);
  KernelsLaunchers::run_cross_entropy_loss_kernel(
      d_model.GetActivations().ln2, d_target, d_model.GetActivations().sm,
      d_model.GetActivations().loss, B, C, 32, stream);

  KernelsLaunchers::run_reduce_kernel3(d_model.GetActivations().loss,
                                       d_model.GetActivations().reduced_loss, B,
                                       REDUCTION::MEAN, 32, stream);

  // copy the loss to the host

  // cudaCheck(cudaMemcpyAsync(&reduced_loss,
  //                           d_model.GetActivations().reduced_loss,
  //                           sizeof(float), cudaMemcpyDeviceToHost, stream));

  //  printf("Loss: %f\n", *reduced_loss);

  // backpropagation
  KernelsLaunchers::run_crossentropy_softmax_backward(
      d_model.GetDownstreamGradients().dsm, d_model.GetActivations().sm,
      d_target, B, C, 32, REDUCTION::MEAN, stream);
  // get_from_gpu_and_print("dsm", d_model.GetDownstreamGradients().dsm, B * C);

  KernelsLaunchers::runLinearBackward(
      d_model.GetActivations().a1, d_model.GetParams().ln2w,
      d_model.GetDownstreamGradients().dsm, d_model.GetGradients().ln2w_grad,
      d_model.GetGradients().ln2b_grad, d_model.GetDownstreamGradients().dln2,
      B, H1, C, 32, stream);

  // get_from_gpu_and_print("dln2", d_model.GetDownstreamGradients().dln2, B *
  // H1); get_from_gpu_and_print("ln2w_grad", d_model.GetGradients().ln2w_grad,
  // C * H1); get_from_gpu_and_print("ln2b_grad",
  // d_model.GetGradients().ln2b_grad, C);

  KernelsLaunchers::runReluBackward(
      d_model.GetActivations().ln1, d_model.GetDownstreamGradients().dln2,
      d_model.GetDownstreamGradients().da1, B, H1, 32, stream);
  // get_from_gpu_and_print("da1", d_model.GetDownstreamGradients().da1, B *
  // H1);
  KernelsLaunchers::runLinearBackward(
      d_inp, d_model.GetParams().ln1w, d_model.GetDownstreamGradients().da1,
      d_model.GetGradients().ln1w_grad, d_model.GetGradients().ln1b_grad,
      d_model.GetDownstreamGradients().dln1, B, input_dim, H1, 32, stream);
  // Magic optimizer
  KernelsLaunchers::SGD_run_kernel(
      d_model.GetParamsMemory(), d_model.GetGradientsMemory(),
      d_model.get_num_parameters(), 0.01, 0.0, 32, stream);
  cudaCheck(cudaEventRecord(events[my_turn], stream));

  // Free allocated memory on device
  //  cudaCheck(cudaFree(d_inp));
  //  cudaCheck(cudaFree(d_target));
  //  cudaCheck(cudaFreeAsync(d_inp, stream));
  //  cudaCheck(cudaFreeAsync(d_target, stream));

  //  sync stream and print loss
  //  cudaCheck(cudaStreamSynchronize(stream));
  //  printf("Loss: %f\n", reduced_loss);
}

int main() {
  uint B = 1024;
  trainData td{};
  read_training_data("../dataset/x_train.npy", "../dataset/y_train.npy", td,
                     true, true);
  uint input_dim = td.input_shape[1];
  uint H1 = 256;
  uint C = 16;
  const int EPOCHS = 5; // Number of epochs to train
  uint NUM_BATCHES = (td.input_shape[0] + B - 1) / B;
  ModelMemoryHandler h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
#if TEST_PYTORTH
  write_npy("trained-model\\ln1w.npy", h_model.GetParams().ln1w, 2,
            new unsigned long[2]{H1, input_dim});
  write_npy("trained-model\\ln1b.npy", h_model.GetParams().ln1b, 1,
            new unsigned long[1]{H1});
  write_npy("trained-model\\ln2w.npy", h_model.GetParams().ln2w, 2,
            new unsigned long[2]{C, H1});
  write_npy("trained-model\\ln2b.npy", h_model.GetParams().ln2b, 1,
            new unsigned long[1]{C});
#endif
  // move to GPU
  ModelMemoryHandler d_model;
  h_model.model_to_cuda(&d_model);
  cudaEvent_t start_event, stop_event;
  cudaCheck(cudaEventCreate(&start_event));
  cudaCheck(cudaEventCreate(&stop_event));

  cudaEvent_t events[2];
  cudaStream_t streams[2];
  for (int i = 0; i < 2; i++) {
    cudaCheck(cudaEventCreate(&events[i]));
    cudaCheck(cudaStreamCreate(&streams[i]));
  }
  float total_time = 0;
  bool turn = false;
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    printf("Epoch %d\n", epoch + 1);
    cudaCheck(cudaEventRecord(start_event));
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
      // Load batch
      float *inp;
      uint *target;
      uint batch_to_load = std::min(B, td.input_shape[0] - batch * B);
      load_batch(td, &inp, &target, batch_to_load, batch);
      // Train on batch
      train_step(d_model, inp, target, batch_to_load, input_dim, H1, C, turn,
                 events, streams[turn]);
      turn = !turn;
    }
    cudaCheck(cudaEventRecord(stop_event));
    cudaCheck(cudaEventSynchronize(stop_event));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("Epoch time: %f\n", elapsed_time);
    total_time += elapsed_time;
  }
  printf("Total time: %f\n", total_time);
  printf("Average time per epoch: %f\n", total_time / EPOCHS);
  cudaCheck(cudaDeviceSynchronize());
  float reduced_loss;
  cudaCheck(cudaMemcpy(&reduced_loss, d_model.GetActivations().reduced_loss,
                       sizeof(float), cudaMemcpyDeviceToHost));
  printf("Final Loss: %f\n", reduced_loss);
  for (int i = 0; i < 2; i++) {
    cudaCheck(cudaEventDestroy(events[i]));
    cudaCheck(cudaStreamDestroy(streams[i]));
  }
  cudaCheck(cudaEventDestroy(start_event));
  cudaCheck(cudaEventDestroy(stop_event));
  return 0;
}
