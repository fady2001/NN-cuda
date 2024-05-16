
#include "common.hpp"
#include "matmuls_kernels.cuh"
#include <cassert>
#define TEST_PYTORTH true
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
__global__ void mat_mul_naive(const float *A, const float *B, float *C, uint N,
                              uint L, uint M, bool is_first_T,
                              bool is_second_T) {
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

__global__ void mat_mul_K2(const float *A, const float *B, float *C, uint N,
                           uint L, uint M, bool is_first_T, bool is_second_T) {
  extern __shared__ float shared_mem[];
  // the grid size is (B,M,1)
  // this maps one thread to one output element
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
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[tile_row * L + col];
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
    C[row * M + col] = sum;
  }
}
void mat_mul_dispatcher(float *A, float *B, float *C, uint N, uint L, uint M,
                        bool is_first_T, bool is_second_T,
                        int sqrt_block_size) {

  size_t shared_mem_size =
      2 * sqrt_block_size * sqrt_block_size * sizeof(float);

  dim3 grid_dim((M + sqrt_block_size - 1) / sqrt_block_size,
                (N + sqrt_block_size - 1) / sqrt_block_size);
  dim3 block_dim(sqrt_block_size, sqrt_block_size);

  if (!is_first_T && !is_second_T) {
    mat_mul_K2_none<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N, L, M);
  } else if (is_first_T && !is_second_T) {
    mat_mul_K2_first_T<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N, L,
                                                                 M);
  } else if (!is_first_T && is_second_T) {
    mat_mul_K2_second_T<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N, L,
                                                                  M);
  } else if (is_first_T && is_second_T) {
    mat_mul_K2_both_T<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N, L,
                                                                M);
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
  mat_mul_naive<<<grid, block>>>(A, B, C, N, L, M, is_first_T, is_second_T);
  cudaCheck(cudaDeviceSynchronize());
}

void runMatMull_K2(float *A, float *B, float *C, uint N, uint L, uint M,
                   bool is_first_T, bool is_second_T, int sqrt_block_size) {
  dim3 block(sqrt_block_size, sqrt_block_size);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  uint shmemSize = 2 * sqrt_block_size * sqrt_block_size * sizeof(float);
  mat_mul_K2<<<grid, block, shmemSize>>>(A, B, C, N, L, M, is_first_T,
                                         is_second_T);
  cudaCheck(cudaDeviceSynchronize());
}

int main() {
  srand(0);
  uint A_d1 = 1024;
  uint A_d2 = 128;
  uint B_d1 = 1024;
  uint B_d2 = 128;

  bool is_f_T = false;
  bool is_s_T = true;

  // create host memory of random numbers
  float *A = make_random_float(A_d1 * A_d2);
  float *B = make_random_float(B_d1 * B_d2);
  float *A_T = A;
  float *B_T = B;
  if (is_f_T) {
    A_T = transpose(A, A_d1, A_d2);
  }
  if (is_s_T) {
    B_T = transpose(B, B_d1, B_d2);
  }

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // N*L*M => N*M
  uint N = is_f_T ? A_d2 : A_d1;
  uint L = is_f_T ? A_d1 : A_d2;
  uint M = is_s_T ? B_d1 : B_d2;
  printf("N %d L %d M %d\n", N, L, M);
  float *C = (float *)malloc(N * M * sizeof(float));

  //  assert A_d2 == B_d1
  //  assert(A_d2 == B_d1);

  // just run cpu
  mat_mul_cpu(A_T, B_T, C, N, L, M);

// write arrays to npy files if you want to test with torch
#if TEST_PYTORTH
#endif

  // move to GPU
  float *d_A;
  float *d_B;
  float *d_C;
  cudaCheck(cudaMalloc(&d_A, A_d1 * A_d2 * sizeof(float)));
  cudaCheck(cudaMalloc(&d_B, B_d1 * B_d2 * sizeof(float)));
  cudaCheck(cudaMalloc(&d_C, N * M * sizeof(float)));
  cudaCheck(
      cudaMemcpy(d_A, A, A_d1 * A_d2 * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_B, B, B_d1 * B_d2 * sizeof(float), cudaMemcpyHostToDevice));

#if TEST_PYTORTH
  write_npy("matmul\\A.npy", A, 2, new unsigned long[2]{A_d1, A_d2});
  write_npy("matmul\\B.npy", B, 2, new unsigned long[2]{B_d1, B_d2});
  write_npy("matmul\\C.npy", C, 2, new unsigned long[2]{N, M});
#endif

  // print_2D_Matrix(out, "out", B, B_d2);
  int sqrt_block_sizes[] = {4, 8, 16, 32};
  // first check the correctness of the kernel
  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];
    printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
    mat_mul_dispatcher(d_A, d_B, d_C, N, L, M, is_f_T, is_s_T, sqrt_block_size);
    validate_result(d_C, C, "out", size_t(N) * M, 1e-3f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];

    int repeat_times = 100;
    float elapsed_time =
        benchmark_kernel(repeat_times, mat_mul_dispatcher, d_A, d_B, d_C, A_d1,
                         A_d2, B_d2, is_f_T, is_s_T, sqrt_block_size);
    // napkin math: estimate the flops achieved
    // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
    float tflops = (float)N * L * M * 2 / elapsed_time * 1e3f / 1e12f;
    printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n",
           sqrt_block_size, elapsed_time, tflops);
  }

  // free memory
  free(A);
  free(B);
  free(C);
  //  free(A_T);
  //  free(B_T);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}