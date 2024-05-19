#include "common.hpp"
#include "kernels.cuh"
#include "matmuls_kernels.cuh"

class KernelsLaunchers {
public:
  static void linear_layer(float *X, float *W, float *bias, float *y, uint B,
                           uint N, uint M, uint sqrt_block_size,
                           cudaStream_t stream = nullptr) {
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((B + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    linear_layer_forward_naive<<<grid, block, 0, stream>>>(X, W, bias, y, B, N,
                                                           M);
    //    cudaCheck(cudaDeviceSynchronize());
  }

  static void run_relu_kernel(float *input, float *output, uint B, uint N,
                              uint sqrt_block_size,
                              cudaStream_t stream = nullptr) {
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((N + block.x - 1) / block.x, (B + block.y - 1) / block.y);
    relu_forward<<<grid, block, 0, stream>>>(input, output, B, N);
    //    cudaCheck(cudaDeviceSynchronize());
  }

  static void runReluBackward(float *input, float *upGrad, float *downGrad,
                              uint B, uint N, uint sqrt_block_size,
                              cudaStream_t stream = nullptr) {
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((N + block.x - 1) / block.x, (B + block.y - 1) / block.y);
    relu_backward<<<grid, block, 0, stream>>>(input, upGrad, downGrad, B, N);
    //    cudaCheck(cudaDeviceSynchronize());
  }
  static void run_log_softmax_kernel(float *input, float *output, uint N,
                                     uint C, uint block_size,
                                     cudaStream_t stream = nullptr) {
    int num_blocks = (N + block_size - 1) / block_size;
    log_softmax_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, N,
                                                              C);
    //    cudaCheck(cudaDeviceSynchronize());
  }

  template <class T>
  static void run_cross_entropy_loss_kernel(T *in, uint *targets, T *softmaxed,
                                            T *losses, int N, int C,
                                            int block_size,
                                            cudaStream_t stream = nullptr) {
    int num_blocks = ceil_div(N, block_size);
    cross_entropy_kernel<<<num_blocks, block_size, 0, stream>>>(
        in, targets, softmaxed, losses, N, C);
  }

  static void run_reduce_kernel3(float *d_a, float *d_result, uint size,
                                 REDUCTION reduction, uint block_size,
                                 cudaStream_t stream = nullptr) {
    // (dividend + divisor - 1) / divisor
    int num_blocks = (size + block_size - 1) / block_size;
    reduce_kernel3<<<1, num_blocks, block_size * sizeof(float), stream>>>(
        d_a, d_result, size, reduction);
    //    cudaCheck(cudaGetLastError());
  }

  static void run_crossentropy_softmax_backward(float *down_grads, float *probs,
                                                uint *targets, uint N, uint C,
                                                uint block_size,
                                                REDUCTION reduction = MEAN,
                                                cudaStream_t stream = nullptr) {
    const int grid_size = (N + block_size - 1) / block_size;
    crossentropy_softmax_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        down_grads, probs, targets, N, C, reduction);
    cudaCheck(cudaGetLastError());
  }

  static void runMatMull(float *A, float *B, float *C, uint N, uint L, uint M,
                         bool is_first_T, bool is_second_T, int sqrt_block_size,
                         cudaStream_t stream = nullptr) {
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    mat_mul_naive<<<grid, block, 0, stream>>>(A, B, C, N, L, M, is_first_T,
                                              is_second_T);
    //    cudaCheck(cudaDeviceSynchronize());
  }

  static void runReduceOnAxisKernel(float *d_A, float *d_out, uint N, uint M,
                                    int block_size, bool take_avg = false,
                                    cudaStream_t stream = nullptr) {
    // we will use 256 threads per block
    uint grid_size = (M + block_size - 1) / block_size;
    reduce_on_axis<<<grid_size, block_size, 0, stream>>>(d_A, d_out, N, M,
                                                         take_avg);
    //    cudaCheck(cudaPeekAtLastError());
    //    cudaCheck(cudaDeviceSynchronize());
  }

  static void runLinearBackward(float *d_inp, float *d_weight, float *d_up_grad,
                                float *d_dLdw, float *d_dLdb, float *d_dLdx,
                                uint B, uint N, uint M, int sqrt_block_size,
                                cudaStream_t stream = nullptr) {
    // compute dL/dW = (dL/dout).T * x
    //    runMatMull(d_up_grad, d_inp, d_dLdw, M, B, N, true, false,
    //    sqrt_block_size);
    mat_mul_dispatcher(d_up_grad, d_inp, d_dLdw, M, B, N, true, false,
                       sqrt_block_size, 0, stream);
    // compute dL/db = avg(dL/dout, axis=0)
    runReduceOnAxisKernel(d_up_grad, d_dLdb, B, M, sqrt_block_size, false,
                          stream);
    // compute dL/dx = dL/dout * W
    //    runMatMull(d_up_grad, d_weight, d_dLdx, B, M, N, false, false,
    //               sqrt_block_size);
    mat_mul_dispatcher(d_up_grad, d_weight, d_dLdx, B, M, N, false, false,
                       sqrt_block_size, 0, stream);
  }

  static void SGD_run_kernel(float *params_memory, const float *grads_memory,
                             uint num_parameters, float learning_rate = 1e-3,
                             float weight_decay = 0.0, int block_size = 16,
                             cudaStream_t stream = nullptr) {
    int num_blocks = ceil_div<float>(num_parameters, block_size);
    SGD_kernel<<<num_blocks, block_size, 0, stream>>>(
        params_memory, grads_memory, num_parameters, learning_rate,
        weight_decay);
  }
};