//#include "ModelMemoryHandler.cuh"
//#include "common.hpp"
//#include "device_launch_parameters.h"
//#include "kernels_launchers.cuh"
//#include <cmath>
//#include <vector>
//
//#define TEST_PYTORTH true
//
//// Function to simulate loading a batch of data
//void load_batch(trainData &data, float *curr_input, uint *current_target,
//              uint B, uint batch_index) {
//
//// Load the batch from the dataset
//for (int i = 0; i < B; i++) {
//  //    load input vector
//  for (int j = 0; j < data.input_shape[1]; j++) {
//    curr_input[i * data.input_shape[1] + j] =
//        data.inp[(batch_index * B + i) * data.input_shape[1] + j];
//  }
//  current_target[i] = data.target[batch_index * B + i];
//}
//}
//
//// Function to perform a single training step
//void train_step(ModelMemoryHandler &d_model, float *inp, uint *target, uint B,
//              uint input_dim, uint H1, uint C) {
//
//int deviceIdx = 0;
//cudaCheck(cudaSetDevice(deviceIdx));
//
//// move input and target to GPU
//float *d_inp;
//uint *d_target;
//cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(float)));
//cudaCheck(cudaMalloc(&d_target, B * sizeof(uint)));
//cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(float),
//                     cudaMemcpyHostToDevice));
//cudaCheck(
//    cudaMemcpy(d_target, target, B * sizeof(uint), cudaMemcpyHostToDevice));
//// run the model
//KernelsLaunchers::linear_layer(
//    d_inp, d_model.GetParams().ln1w, d_model.GetParams().ln1b,
//    d_model.GetActivations().ln1, B, input_dim, H1, 32);
//KernelsLaunchers::run_relu_kernel(d_model.GetActivations().ln1,
//                                  d_model.GetActivations().a1, B, H1, 32);
//KernelsLaunchers::linear_layer(
//    d_model.GetActivations().a1, d_model.GetParams().ln2w,
//    d_model.GetParams().ln2b, d_model.GetActivations().ln2, B, H1, C, 32);
//    KernelsLaunchers::run_cross_entropy_loss_kernel(d_model.GetActivations().ln2, d_target, d_model.GetActivations().sm,d_model.GetActivations().loss, B, C, 32);
//KernelsLaunchers::run_reduce_kernel3(d_model.GetActivations().loss,
//                                     d_model.GetActivations().reduced_loss, B,
//                                     REDUCTION::MEAN, 32);
//
//// cuda synchronize();
//cudaCheck(cudaDeviceSynchronize());
//// copy the loss to the host
//float *reduced_loss = (float *)malloc(sizeof(float));
//cudaCheck(cudaMemcpy(reduced_loss, d_model.GetActivations().reduced_loss,
//                     sizeof(float), cudaMemcpyDeviceToHost));
//printf("Loss: %f\n", *reduced_loss);
//
//// backpropagation
//KernelsLaunchers::run_crossentropy_softmax_backward(
//    d_model.GetDownstreamGradients().dsm, d_model.GetActivations().sm,
//    d_target, B, C, 32, REDUCTION::MEAN);
//// get_from_gpu_and_print("dsm", d_model.GetDownstreamGradients().dsm, B * C);
//
//KernelsLaunchers::runLinearBackward(
//    d_model.GetActivations().a1, d_model.GetParams().ln2w,
//    d_model.GetDownstreamGradients().dsm, d_model.GetGradients().ln2w_grad,
//    d_model.GetGradients().ln2b_grad, d_model.GetDownstreamGradients().dln2,
//    B, H1, C, 32);
//
//// get_from_gpu_and_print("dln2", d_model.GetDownstreamGradients().dln2, B *
//// H1); get_from_gpu_and_print("ln2w_grad", d_model.GetGradients().ln2w_grad,
//// C * H1); get_from_gpu_and_print("ln2b_grad",
//// d_model.GetGradients().ln2b_grad, C);
//
//KernelsLaunchers::runReluBackward(
//    d_model.GetActivations().ln1, d_model.GetDownstreamGradients().dln2,
//    d_model.GetDownstreamGradients().da1, B, H1, 32);
//// get_from_gpu_and_print("da1", d_model.GetDownstreamGradients().da1, B *
//// H1);
//KernelsLaunchers::runLinearBackward(
//    d_inp, d_model.GetParams().ln1w, d_model.GetDownstreamGradients().da1,
//    d_model.GetGradients().ln1w_grad, d_model.GetGradients().ln1b_grad,
//    d_model.GetDownstreamGradients().dln1, B, input_dim, H1, 32);
//// Magic optimizer
//KernelsLaunchers::SGD_run_kernel(d_model.GetParamsMemory(),
//                                 d_model.GetGradientsMemory(),
//                                 d_model.get_num_parameters(), 0.01, 0.0, 32);
//
//// Free allocated memory on device
//cudaCheck(cudaFree(d_inp));
//cudaCheck(cudaFree(d_target));
//free(reduced_loss);
//}
//
//int main() {
//uint B = 32;
//trainData td{};
//read_training_data("../dataset/x_train.npy", "../dataset/y_train.npy", td,
//                   true);
//uint input_dim = td.input_shape[1];
//uint H1 = 256;
//uint C = 16;
//const int EPOCHS = 10; // Number of epochs to train
//uint NUM_BATCHES = (td.input_shape[0] + B - 1) / B;
//ModelMemoryHandler h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);
//
//int deviceIdx = 0;
//cudaCheck(cudaSetDevice(deviceIdx));
//#if TEST_PYTORTH
//write_npy("trained-model\\ln1w.npy", h_model.GetParams().ln1w, 2,
//          new unsigned long[2]{H1, input_dim});
//write_npy("trained-model\\ln1b.npy", h_model.GetParams().ln1b, 1,
//          new unsigned long[1]{H1});
//write_npy("trained-model\\ln2w.npy", h_model.GetParams().ln2w, 2,
//          new unsigned long[2]{C, H1});
//write_npy("trained-model\\ln2b.npy", h_model.GetParams().ln2b, 1,
//          new unsigned long[1]{C});
//#endif
//// move to GPU
//ModelMemoryHandler d_model;
//h_model.model_to_cuda(&d_model);
//
//for (int epoch = 0; epoch < EPOCHS; epoch++) {
//  printf("Epoch %d\n", epoch + 1);
//  for (int batch = 0; batch < NUM_BATCHES; batch++) {
//    // Load batch
//    float *inp = (float *)malloc(B * input_dim * sizeof(float));
//    uint *target = (uint *)malloc(B * sizeof(uint));
//    uint batch_to_load = std::min(B, td.input_shape[0] - batch * B);
//    load_batch(td, inp, target, batch_to_load, batch);
//
//    // Train on batch
//    train_step(d_model, inp, target, batch_to_load, input_dim, H1, C);
//
//    // Free allocated host memory for the batch
//    free(inp);
//    free(target);
//  }
//}
//
//return 0;
//}
