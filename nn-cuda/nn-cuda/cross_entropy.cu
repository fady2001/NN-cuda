#include "common.hpp"
#include <stdio.h>
#define TEST_PYTORTH true

// ------------------------------- cpu version -------------------------------
/**
 * @brief
 *  this a template function to be suitable for float and double numbers to
 * apply cross entropy
 *
 * @param in: input tensor of shape (N, C) where N is the batch size (number of
 * rows) and C (number of columns) is the number of classes
 * @param targets: target tensor of shape (N) contains number from 0 to C-1
 * @param softmaxed: output tensor of shape (N, C) contains the softmaxed values
 * @param losses: output tensor of shape (N)
 * @param N: number of rows
 * @param C: number of columns
 */

template <class T>
void cross_entropy_cpu(T *in, uint *targets, T *softmaxed, T *losses, int N,
                       int C) {
  // loop over each row. each row will get softmaxed
  for (int i = 0; i < N; i++) {
    // assume that the first element in the row is the maximum
    T max_val = in[i * C];
    // loop to get the maximum value of each row
    for (int j = 1; j < C; j++) {
      if (in[i * C + j] > max_val) {
        max_val = in[i * C + j];
      }
    }

    T sum = 0;
    // loop over the row to calculate the sum and apply normalization
    for (int j = 0; j < C; j++) {
      // apply normalization step to ensure that the maximum value will be 0 to
      // avoid overflow
      in[i * C + j] = in[i * C + j] - max_val;
      sum += exp(in[i * C + j]);
    }
    // output softmaxed values
    for (int j = 0; j < C; j++) {
      softmaxed[i * C + j] = in[i * C + j] - log(sum);
    }
    // calculate the loss
    losses[i] = -softmaxed[i * C + targets[i]];
  }
}

// ------------------------------- gpu version -------------------------------

/* each thread will process only one row */
template <class T>
__global__ void cross_entropy_kernel(T *in, uint *targets, T *softmaxed,
                                     T *losses, int N, int C) {
  // input dimension (N,C)
  // output dimension (N,C)
  // get actual index in in_h and out_d
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    T max_val = in[i * C];
    for (int j = 1; j < C; j++) {
      if (in[i * C + j] > max_val) {
        max_val = in[i * C + j];
      }
    }

    T sum = 0;
    for (int j = 0; j < C; j++) {
      // apply normalization step to ensure that the maximum value will be 0 to
      // avoid overflow
      in[i * C + j] = in[i * C + j] - max_val;
      sum += exp(in[i * C + j]);
    }
    // output softmaxed values
    for (int j = 0; j < C; j++) {
      softmaxed[i * C + j] = in[i * C + j] - log(sum);
    }
    // calculate the loss
    losses[i] = -softmaxed[i * C + targets[i]];
  }
}

template <class T>
void run_kernel1(T *in, uint *targets, T *softmaxed, T *losses, int N, int C,
                 int block_size) {
  int num_blocks = ceil_div(N, block_size);
  cross_entropy_kernel<<<num_blocks, block_size>>>(in, targets, softmaxed,
                                                   losses, N, C);
}

int main() {
  srand(0);
  const unsigned long N = 10000, C = 10000;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // create host memory of random numbers
  float *h_inp = make_random_float_01(N * C);
  uint *h_targets = make_random_int(N, C);
  float *h_softmaxed = (float *)malloc(N * C * sizeof(float));
  float *h_losses = (float *)malloc(N * sizeof(float));

#if TEST_PYTORTH
  write_npy("cross-entropy-layer\\h_inp.npy", h_inp, 2,
            new unsigned long[2]{N, C});
  write_npy<uint>("cross-entropy-layer\\h_targets.npy", h_targets, 1,
                  new unsigned long[1]{N});
#endif

  // move to GPU
  float *d_inp;
  uint *d_targets;
  float *d_softmaxed;
  float *d_losses;
  cudaCheck(cudaMalloc(&d_inp, N * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_targets, N * sizeof(uint)));
  cudaCheck(cudaMalloc(&d_softmaxed, N * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_losses, N * sizeof(float)));
  cudaCheck(
      cudaMemcpy(d_inp, h_inp, N * C * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_targets, h_targets, N * sizeof(uint),
                       cudaMemcpyHostToDevice));

  measureExecutionTime(cross_entropy_cpu<float>, h_inp, h_targets, h_softmaxed,
                       h_losses, N, C);
#if TEST_PYTORTH
  write_npy("cross-entropy-layer\\h_softmaxed.npy", h_softmaxed, 2,
            new unsigned long[2]{N, C});
  write_npy("cross-entropy-layer\\h_losses.npy", h_losses, 1,
            new unsigned long[1]{N});
#endif

  int block_sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
  // first check the correctness of the kernel
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    run_kernel1<float>(d_inp, d_targets, d_softmaxed, d_losses, N, C,
                       block_sizes[j]);
    validate_result(d_softmaxed, h_softmaxed, "out", N, 1e-4f);
    validate_result(d_losses, h_losses, "out", N, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];

    int repeat_times = 100;
    float elapsed_time =
        benchmark_kernel(repeat_times, run_kernel1<float>, d_inp, d_targets,
                         d_softmaxed, d_losses, N, C, block_sizes[j]);

    printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size,
           elapsed_time, elapsed_time * 1'000 / (N * C));
  }

  // free memory
  free(h_inp);
  free(h_targets);
  free(h_softmaxed);
  free(h_losses);
  cudaFree(d_inp);
  cudaFree(d_targets);
  cudaFree(d_softmaxed);
  cudaFree(d_losses);
  return 0;
}