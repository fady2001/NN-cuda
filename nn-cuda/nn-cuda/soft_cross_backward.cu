#include "common.hpp"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#define TEST_PYTORTH true
/**
 * @brief
 *
 * @tparam T
 * @param down_grads: output tensor of shape (N, C)
 * @param probs: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
 * @param targets : target tensor of shape (N) contains number from 0 to C-1
 * @param N : number of rows
 * @param C : number of columns
 * @return __global__
 */
template <class T>
void crossentropy_softmax_backward_cpu(T *down_grads, const T *probs, const int *targets, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < C; j++)
        {
            down_grads[j + i * C] = probs[j + i * C] - ((j == targets[i]) ? 1.0f : 0.0f);
        }
    }
}

template <class T>
__global__ void crossentropy_softmax_backward_kernel(T *down_grads, const T *probs, const int *targets, int N, int C)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        for (int j = 0; j < C; j++)
        {
            down_grads[j + i * C] = probs[j + i * C] - ((j == targets[i]) ? 1.0f : 0.0f);
        }
    }
}

// kernel launcher
template <class T>
void run_crossentropy_softmax_backward(T *down_grads, T *probs, int *targets, int N, int C, int block_size)
{
    const int grid_size = (N + block_size - 1) / block_size;
    crossentropy_softmax_backward_kernel<<<grid_size, block_size>>>(down_grads, probs, targets, N, C);
    cudaCheck(cudaGetLastError());
}

int main()
{
    srand(0);
    float *down_grads;
    float *h_probs;
    int *h_targets;
    const unsigned long C = 3;
    const unsigned long N = 3;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    down_grads = make_random_float(N * C);
    h_targets = make_random_int(N, C);
    h_probs = make_random_float(N * C);

#if TEST_PYTORTH
    write_npy("cross-entropy-backward\\down_grads.npy", down_grads, 2, new unsigned long[2]{N, C});
    write_npy("cross-entropy-backward\\h_targets.npy", h_targets, 1, new unsigned long[1]{N});
    write_npy("cross-entropy-backward\\h_probs.npy", h_probs, 2, new unsigned long[2]{N, C});
#endif

    // CPU
    crossentropy_softmax_backward_cpu(down_grads, h_probs, h_targets, N, C);

#if TEST_PYTORTH
    write_npy("cross-entropy-backward\\h_dlogits_after.npy", down_grads, 2, new unsigned long[2]{N, C});
#endif

    // GPU
    float *d_down_grads;
    float *d_probs;
    int *d_targets;
    cudaCheck(cudaMalloc(&d_down_grads, N * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs, N * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, N * sizeof(int)));

    cudaCheck(cudaMemcpy(d_down_grads, down_grads, N * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_probs, h_probs, N * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, h_targets, N * sizeof(int), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        run_crossentropy_softmax_backward(d_down_grads, d_probs, d_targets, N, C, block_size);
        validate_result(d_down_grads, down_grads, "out", N * C, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, run_crossentropy_softmax_backward<float>, d_down_grads, d_probs, d_targets, N, C, block_sizes[j]);

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (N * C));
    }

    // free memory
    free(down_grads);
    free(h_targets);
    free(h_probs);
    cudaFree(d_down_grads);
    cudaFree(d_probs);
    cudaFree(d_targets);
    return 0;
}