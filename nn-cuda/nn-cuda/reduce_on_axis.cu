
#include "common.hpp"
#include <cassert>
#define TEST_PYTORTH true
#define uint unsigned int
/**
 * @brief
 *  This performs reduction on a matrix
 *  Matrix is B*M
 *  The output is M
 */
__global__ void reduce_on_axis(const float *A, float *out, uint N, uint M,
                               bool take_avg = false) {
  // this maps one thread per column
  // the grid size is (B,1,1)
  uint j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < M) {
    float sum = 0;
    for (uint k = 0; k < N; k++) {
      sum += A[k * M + j];
    }
    out[j] = sum;
    if (take_avg) {
      out[j] /= (float)N;
    }
  }
}

void reduce_on_axis_cpu(const float *A, float *out, uint N, uint M) {
  for (uint j = 0; j < M; j++) {
    float sum = 0;
    for (uint k = 0; k < N; k++) {
      sum += A[k * M + j];
    }
    out[j] = sum;
  }
}

void runReduceOnAxisKernel(float *d_A, float *d_out, uint N, uint M,
                           int block_size) {
  // we will use 256 threads per block
  uint grid_size = (M + block_size - 1) / block_size;
  reduce_on_axis<<<grid_size, block_size>>>(d_A, d_out, N, M);
  cudaCheck(cudaPeekAtLastError());
  cudaCheck(cudaDeviceSynchronize());
}

int main() {
  srand(0);
  uint B = 10000;
  uint M = 10000;
  float *inp = make_random_float(B * M);
  float *out = (float *)malloc(M * sizeof(float));
  float *d_out;
  float *d_inp;
  cudaCheck(cudaMalloc(&d_out, B * M * sizeof(float)));
  cudaCheck(cudaMalloc(&d_inp, B * M * sizeof(float)));
  cudaCheck(
      cudaMemcpy(d_inp, inp, B * M * sizeof(float), cudaMemcpyHostToDevice));

  reduce_on_axis_cpu(inp, out, B, M);

#if TEST_PYTORTH
  write_npy("reduce-on-axis\\inp.npy", inp, 2, new unsigned long[2]{B, M});
  write_npy("reduce-on-axis\\out.npy", out, 1, new unsigned long[1]{M});
#endif

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    runReduceOnAxisKernel(d_inp, d_out, B, M, block_size);
    validate_result(d_out, out, "out", M, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];

    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, runReduceOnAxisKernel,
                                          d_inp, d_out, B, M, block_sizes[j]);

    printf("block_size %4d | time %.4f ms \n", block_size, elapsed_time);
  }

  free(inp);
  free(out);
  cudaFree(d_out);
  cudaFree(d_inp);
  return 0;
}