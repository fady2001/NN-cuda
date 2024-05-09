#include <iostream>
#include <cmath>
#include "common.cuh"
#include "ModelLayers.hpp"
#include "ModelMemoryHandler.hpp"
#define TEST_PYTORTH true

int main()
{
    unsigned long input_dim = 3;
    unsigned long B = 30;
    unsigned long H1 = 100;
    unsigned long C = 10;
    ModelMemoryHandler<float> model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

    // create host memory of random numbers
    float *inp = make_random_float(B * input_dim);
    int *target = make_random_int(B, int(C));

#if TEST_PYTORTH
    write_npy("all-model-cpu\\X_c.npy", inp, 2, new unsigned long[2]{B, input_dim});
    write_npy("all-model-cpu\\target.npy", target, 1, new unsigned long[1]{B});
    write_npy("all-model-cpu\\ln1w.npy", model.GetParams().ln1w, 2, new unsigned long[2]{H1, input_dim});
    write_npy("all-model-cpu\\ln1b.npy", model.GetParams().ln1b, 1, new unsigned long[1]{H1});
    write_npy("all-model-cpu\\ln2w.npy", model.GetParams().ln2w, 2, new unsigned long[2]{C, H1});
    write_npy("all-model-cpu\\ln2b.npy", model.GetParams().ln2b, 1, new unsigned long[1]{C});
#endif

    // run CPU
    ModelLayers::linear_layer_forward_cpu<float>(inp, model.GetParams().ln1w, model.GetParams().ln1b, model.GetActivations().ln1, B, input_dim, H1);
    ModelLayers::relu_forward_cpu<float>(model.GetActivations().ln1, model.GetActivations().a1, B, H1);
    ModelLayers::linear_layer_forward_cpu<float>(model.GetActivations().a1, model.GetParams().ln2w, model.GetParams().ln2b, model.GetActivations().ln2, B, H1, C);
    ModelLayers::softmax_cpu<float>(model.GetActivations().ln2, model.GetActivations().sm, B, C);
    ModelLayers::cross_entropy_cpu<float>(model.GetActivations().loss, model.GetActivations().sm, target, B, C);
    ModelLayers::array_sum_cpu<float>(model.GetActivations().reduced_loss, model.GetActivations().loss, B);

    // print results
    printf("Reduced Loss: %f\n", *model.GetActivations().reduced_loss);
    return 0;
}