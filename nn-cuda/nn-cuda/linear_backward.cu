
#include "common.hpp"
#define TEST_PYTORTH true
#define FLOAT_TYPE float
#define uint unsigned int
/**
 * @brief
 *  This performs generic matrix multiplication
 *  C = A * B
 *  where A is of shape (N,L)
 *  B is of shape (L,M)
 *  C is of shape (N,M)
 *  The kernel is generic and can be used for any matrix multiplication
 *
 *  @param A: input matrix A of shape (N,L) or (L,N) if is_first_T is set
 *  @param B: input matrix B of shape (L,M) or (M,L) if is_second_T is set
 *  @param C: output matrix C of shape (N,M)
 *  @param is_first_T: if true, A is transposed so the multiplication is A^T * B
 *  @param is_second_T: if true, B is transposed
 *
 */
__global__ void mat_mul(const float *A, const float *B, float *C, uint N,
                        uint L, uint M, bool is_first_T, bool is_second_T) {
  // this maps one thread to one output element
  // the grid size is (B,M,1)
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < M) {
    float sum = 0;
    for (uint k = 0; k < L; k++) {
      float a = is_first_T ? A[k * N + i] : A[i * L + k];
      float b = is_second_T ? B[j * L + k] : B[k * M + j];
      sum += a * b;
    }
    C[i * M + j] = sum;
  }
}

void mat_mul_cpu(const float *A, const float *B, float *C, uint N, uint L,
                 uint M) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < M; j++) {
      float sum = 0;
      for (uint k = 0; k < L; k++) {
        sum += A[i * L + k] * B[k * M + j];
      }
      C[i * M + j] = sum;
    }
  }
}
float *transpose(const float *A, uint N, uint M) {
  float *out = (float *)malloc(N * M * sizeof(float));
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < M; j++) {
      out[j * N + i] = A[i * M + j];
    }
  }
  return out;
}

void runMatMull(float *A, float *B, float *C, uint N, uint L, uint M,
                bool is_first_T, bool is_second_T, int sqrt_block_size) {
  dim3 block(sqrt_block_size, sqrt_block_size);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  mat_mul<<<grid, block>>>(A, B, C, N, L, M, is_first_T, is_second_T);
  cudaCheck(cudaDeviceSynchronize());
}

int main() {
  srand(0);
  const size_t B = 100, N = 100, M = 30;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // create host memory of random numbers
  FLOAT_TYPE *out = (FLOAT_TYPE *)malloc(B * M * sizeof(float));
  FLOAT_TYPE *inp = make_random_float<FLOAT_TYPE>(B * N);
  FLOAT_TYPE *weight = make_random_float<FLOAT_TYPE>(M * N);
  FLOAT_TYPE *bias = make_random_float<FLOAT_TYPE>(M);

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

  linear_layer_forward_cpu(inp, weight, bias, out, B, N, M);

#if TEST_PYTORTH
  write_npy("linear-layer\\out_C.npy", out, 2, new unsigned long[2]{B, M});
#endif

  // print_2D_Matrix(out, "out", B, M);
  int sqrt_block_sizes[] = {4, 8, 16, 32};
  // first check the correctness of the kernel
  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];
    printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
    runKernel1(d_inp, d_weight, d_bias, d_out, B, N, M, sqrt_block_size);
    //        validate_result(d_out, out, "out", B * M, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");
  //    printf("All results match. Starting benchmarks.\n\n");

  //    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++)
  //    {
  //        int sqrt_block_size = sqrt_block_sizes[j];
  //
  //        int repeat_times = 100;
  //        float elapsed_time = benchmark_kernel(repeat_times, runKernel1,
  //        d_inp, d_weight, d_bias, d_out, B, N, M, sqrt_block_size);
  //
  //        // napkin math: estimate the flops achieved
  //        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
  //        float tflops = (float)B * N * M * 2 / elapsed_time * 1e3f / 1e12f;
  //        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n",
  //        sqrt_block_size, elapsed_time, tflops);
  //    }

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