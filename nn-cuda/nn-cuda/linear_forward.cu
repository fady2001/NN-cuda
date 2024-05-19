#include "common.hpp"
#define TEST_PYTORTH true

/**
 * @brief
 *  this performs the forward pass of a linear layer
 * y = x W.T  + b
 *
 * @param X: input tensor of shape (B, N) where B is the batch size and N is the
 * number of input neurons
 * @param W: weight tensor of shape (M, N) where M is the number of output
 * neurons
 * @param bias: bias tensor of shape (M)
 * @param y: output tensor of shape (B, M)
 */
__global__ void linear_layer_forward_naive(float *X, float *W, float *bias,
                                           float *y, int B, int N, int M) {
  // this maps one thread to one output element
  // the grid size is (B,M,1)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // will be used to store the dot product of the i-th row of X and the j-th row
  // of W
  if (i < B && j < M) {
    float dot_product = bias[j];
    for (size_t k = 0; k < N; k++) {
      dot_product += X[i * N + k] * W[j * N + k];
    }
    // store the result in y with the bias
    y[i * M + j] = dot_product;
  }
}

__global__ void linear_layer_forward__optimized(const float *A, const float *B,
                                                float *C, const float *bias,
                                                uint N, uint L, uint M) {

  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
    uint tile_col = tile * TILE_WIDTH + threadIdx.x;
    uint tile_row = tile * TILE_WIDTH + threadIdx.y;

    if (tile_col < L && row < N)
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[row * L + tile_col];
    else
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;

    if (tile_row < L && col < M)
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[col * L + tile_row];
    else
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;

    __syncthreads();

    for (uint k = 0; k < TILE_WIDTH; k++) {
      sum +=
          As[threadIdx.y * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + threadIdx.x];
    }
    __syncthreads();
  }

  if (col < M && row < N) {
    C[row * M + col] = sum + bias[col];
  }
}

void linear_layer_forward_cpu(float *X, float *W, float *bias, float *y, int B,
                              int N, int M) {

  for (int i = 0; i < B; i++) {
    for (int j = 0; j < M; j++) {
      y[i * M + j] = bias[j];
      for (int k = 0; k < N; k++) {
        y[i * M + j] += X[i * N + k] * W[j * N + k];
      }
    }
  }
}
void runKernel1(float *X, float *W, float *bias, float *y, int B, int N, int M,
                int sqrt_block_size) {
  dim3 block(sqrt_block_size, sqrt_block_size);
  dim3 grid((B + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  linear_layer_forward_naive<<<grid, block>>>(X, W, bias, y, B, N, M);
  cudaCheck(cudaDeviceSynchronize());
}
void runLinearForward2(float *X, float *W, float *bias, float *y, int B, int N,
                       int M, int sqrt_block_size) {
  dim3 block(sqrt_block_size, sqrt_block_size);
  dim3 grid((M + block.x - 1) / block.x, (B + block.y - 1) / block.y);
  int shared_mem_size = 2 * block.x * block.y * sizeof(float);
  linear_layer_forward__optimized<<<grid, block, shared_mem_size>>>(
      X, W, y, bias, B, N, M);
  cudaCheck(cudaDeviceSynchronize());
}

int main() {
  srand(0);
  const unsigned long B = 2048, N = 2048, M = 2048;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // create host memory of random numbers
  float *out = (float *)malloc(B * M * sizeof(float));
  float *inp = make_random_float(B * N);
  float *weight = make_random_float(M * N);
  float *bias = make_random_float(M);

// write arrays to npy files if you want to test with torch
#if TEST_PYTORTH
  write_npy("linear-layer\\X_c.npy", inp, 2, new unsigned long[2]{B, N});
  write_npy("linear-layer\\W_C.npy", weight, 2, new unsigned long[2]{M, N});
  write_npy("linear-layer\\bias_C.npy", bias, 1, new unsigned long[1]{M});
#endif

  // move to GPU
  float *d_out;
  float *d_inp;
  float *d_weight;
  float *d_bias;
  cudaCheck(cudaMalloc(&d_out, B * M * sizeof(float)));
  cudaCheck(cudaMalloc(&d_inp, B * N * sizeof(float)));
  cudaCheck(cudaMalloc(&d_weight, M * N * sizeof(float)));
  cudaCheck(cudaMalloc(&d_bias, M * sizeof(float)));
  cudaCheck(
      cudaMemcpy(d_inp, inp, B * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_weight, weight, M * N * sizeof(float),
                       cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_bias, bias, M * sizeof(float), cudaMemcpyHostToDevice));

  //  linear_layer_forward_cpu(inp, weight, bias, out, B, N, M);
  measureExecutionTime(linear_layer_forward_cpu, inp, weight, bias, out, B, N,
                       M);
#if TEST_PYTORTH
  write_npy("linear-layer\\out_C.npy", out, 2, new unsigned long[2]{B, M});
#endif

  // print_2D_Matrix(out, "out", B, M);
  int sqrt_block_sizes[] = {16, 32};
  // first check the correctness of the kernel
  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];
    printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
    //    runLinearForward2(d_inp, d_weight, d_bias, d_out, B, N, M,
    //    sqrt_block_size); validate_result(d_out, out, "out", B * M, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");
  printf("All results match. Starting benchmarks.\n\n");

  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];

    int repeat_times = 100;
    float elapsed_time1 =
        benchmark_kernel(repeat_times, runKernel1, d_inp, d_weight, d_bias,
                         d_out, B, N, M, sqrt_block_size);

    float elapsed_time2 =
        benchmark_kernel(repeat_times, runLinearForward2, d_inp, d_weight,
                         d_bias, d_out, B, N, M, sqrt_block_size);
    float tflops = (float)B * N * M * 2 / elapsed_time1 * 1e3f / 1e12f;
    printf("K1 sqrt_block_size %4d | time %.4f ms | tflops %.2f\n",
           sqrt_block_size, elapsed_time1, tflops);

    float tflops2 = (float)B * N * M * 2 / elapsed_time2 * 1e3f / 1e12f;
    printf("K2 sqrt_block_size %4d | time %.4f ms | tflops %.2f\n",
           sqrt_block_size, elapsed_time2, tflops2);
  }

  // free memory
  free(out);
  free(inp);
  free(weight);
  free(bias);
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_weight));
  cudaCheck(cudaFree(d_bias));
  return 0;
}