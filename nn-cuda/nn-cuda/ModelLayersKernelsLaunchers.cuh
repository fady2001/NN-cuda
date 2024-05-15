#include "ModelLayersKernels.cuh"
#include "common.hpp"

class ModelLayersKernelsLaunchers {
public:
  static void linear_layer(float *X, float *W, float *bias, float *y, int B,
                           int N, int M, int sqrt_block_size) {
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((B + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    linear_layer_forward_naive<<<grid, block>>>(X, W, bias, y, B, N, M);
    cudaCheck(cudaDeviceSynchronize());
  }

  static void run_relu_kernel(float *input, float *output, int B, int N,
                              int sqrt_block_size) {
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((B + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    relu_forward<<<grid, block>>>(input, output, B, N);
    cudaCheck(cudaDeviceSynchronize());
  }

  static void run_softmax_kernel(const float *input, float *output, int N,
                                 int C, int block_size) {
    int num_blocks = (N + block_size - 1) / block_size;
    softmax_kernel<<<num_blocks, block_size>>>(input, output, N, C);
    cudaCheck(cudaDeviceSynchronize());
  }

  static void run_cross_entropy_kernel(float *losses, const float *probs,
                                       const int *targets, int N, int C,
                                       const int block_size) {
    //(dividend + divisor - 1) / divisor
    const int grid_size = (N + block_size - 1) / block_size;
    cross_entropy_kernel<<<grid_size, block_size>>>(losses, probs, targets, N,
                                                    C);
    cudaCheck(cudaGetLastError());
  }

  static void run_array_sum_kernel3(float *d_a, float *d_result, int size,
                                    int block_size) {
    // (dividend + divisor - 1) / divisor
    int num_blocks = (size + block_size - 1) / block_size;
    array_sum_kernel3<<<1, num_blocks, block_size * sizeof(float)>>>(
        d_a, d_result, size);
    cudaCheck(cudaGetLastError());
  }
};