#include "ModelMemoryHandler.cuh"
#include "common.hpp"
#include "device_launch_parameters.h"
#include "kernels_launchers.cuh"
#include <cmath>
#include <vector>

#define TEST_PYTORTH true

// Define some constants for the dataset
const int NUM_BATCHES = 100; // Number of batches to simulate
const int EPOCHS = 10;       // Number of epochs to train

// Function to simulate loading a batch of data
void load_batch(float *&inp, int *&target, int B, int input_dim, int C)
{
    inp = make_random_float(B * input_dim);
    target = make_random_int(B, int(C));
}

// Function to perform a single training step
void train_step(ModelMemoryHandler &d_model, float *inp, int *target, int B, int input_dim, int H1, int C)
{
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // move input and target to GPU
    float *d_inp;
    uint *d_target;
    cudaCheck(cudaMalloc(&d_inp, B * input_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_target, B * sizeof(uint)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_target, target, B * sizeof(int), cudaMemcpyHostToDevice));

    // run the model
    ModelLayersKernelsLaunchers::linear_layer(
        d_inp, d_model.GetParams().ln1w, d_model.GetParams().ln1b, d_model.GetActivations().ln1, B, input_dim, H1, 32);
    ModelLayersKernelsLaunchers::run_relu_kernel(d_model.GetActivations().ln1, d_model.GetActivations().a1, B, H1, 32);
    ModelLayersKernelsLaunchers::linear_layer(
        d_model.GetActivations().a1, d_model.GetParams().ln2w, d_model.GetParams().ln2b, d_model.GetActivations().ln2, B, H1, C, 32);
    ModelLayersKernelsLaunchers::run_softmax_kernel(d_model.GetActivations().ln2, d_model.GetActivations().sm, B, C, 32);
    ModelLayersKernelsLaunchers::run_cross_entropy_kernel(d_model.GetActivations().loss, d_model.GetActivations().sm, d_target, B, C, 32);
    ModelLayersKernelsLaunchers::run_array_sum_kernel3(d_model.GetActivations().loss, d_model.GetActivations().reduced_loss, B, 32);

    // cuda synchronize();
    cudaCheck(cudaDeviceSynchronize());
    // copy the loss to the host
    float *reduced_loss = (float *)malloc(sizeof(float));
    cudaCheck(cudaMemcpy(reduced_loss, d_model.GetActivations().reduced_loss, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Loss: %f\n", *reduced_loss);

    // backpropagation
    ModelLayersKernelsLaunchers::run_crossentropy_softmax_backward(
        d_model.GetDownstreamGradients().dsm, d_model.GetActivations().sm, d_target, B, C, 32);
    ModelLayersKernelsLaunchers::runLinearBackward(
        d_model.GetActivations().a1, d_model.GetParams().ln2w, d_model.GetDownstreamGradients().dsm, d_model.GetGradients().ln2w_grad,
        d_model.GetGradients().ln2b_grad, d_model.GetDownstreamGradients().dln2, B, H1, C, 32);
    ModelLayersKernelsLaunchers::runReluBackward(
        d_model.GetActivations().ln1, d_model.GetDownstreamGradients().dln2, d_model.GetDownstreamGradients().da1, B, H1, 32);
    ModelLayersKernelsLaunchers::runLinearBackward(
        d_inp, d_model.GetParams().ln1w, d_model.GetDownstreamGradients().da1, d_model.GetGradients().ln1w_grad, d_model.GetGradients().ln1b_grad,
        d_model.GetDownstreamGradients().dln1, B, input_dim, H1, 32);

    // Update the parameters
    ModelLayersKernelsLaunchers::SGD_run_kernel(d_model.GetParamsMemory(), d_model.GetGradientsMemory(), d_model.get_num_parameters(), 0.01, 0.0, 32);

    // Free allocated memory on device
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_target));
    free(reduced_loss);
}

int main()
{
    uint input_dim = 20;
    uint B = 32;
    uint H1 = 215;
    uint C = 4;
    ModelMemoryHandler h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // move to GPU
    ModelMemoryHandler d_model;
    h_model.model_to_cuda(&d_model);

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        printf("Epoch %d\n", epoch + 1);
        for (int batch = 0; batch < NUM_BATCHES; batch++)
        {
            // Load batch
            float *inp;
            int *target;
            load_batch(inp, target, B, input_dim, C);

            // Train on batch
            train_step(d_model, inp, target, B, input_dim, H1, C);

            // Free allocated host memory for the batch
            free(inp);
            free(target);
        }
    }

    return 0;
}
