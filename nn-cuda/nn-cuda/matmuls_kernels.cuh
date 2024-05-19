#pragma once
#include "common.hpp"
#include <cuda_runtime.h>
#define uint unsigned int

__global__ void mat_mul_K2_none(const float *A, const float *B, float *C,
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
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[tile_row * M + col];
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

__global__ void mat_mul_K2_optimized(const float *A, const float *B, float *C,
                                     uint N, uint L, uint M) {
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0f;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
    uint tile_col = tile * TILE_WIDTH + threadIdx.x;
    uint tile_row = tile * TILE_WIDTH + threadIdx.y;

    // Coalesced loading of data into shared memory
    if (tile_col < L && row < N)
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[row * L + tile_col];
    else
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

    if (tile_row < L && col < M)
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[tile_row * M + col];
    else
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

    __syncthreads();

#pragma unroll
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

template <uint N_ELEM_PER_THREAD>
__global__ void mat_mul_K2_optimized_v2(const float *A, const float *B,
                                        float *C, uint N, uint L, uint M) {
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint tx = threadIdx.x;
  uint ty = threadIdx.y;
  uint row = blockIdx.y * TILE_WIDTH + ty;
  uint col_start = blockIdx.x * TILE_WIDTH + tx;

  float sum[N_ELEM_PER_THREAD] = {0.0f};

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
    uint tile_col = tile * TILE_WIDTH + tx;

    // Load tiles into shared memory
    if (row < N && tile_col < L) {
      As[ty * TILE_WIDTH + tx] = A[row * L + tile_col];
    } else {
      As[ty * TILE_WIDTH + tx] = 0.0f;
    }

    for (uint e = 0; e < N_ELEM_PER_THREAD; e++) {
      uint current_col = col_start + e * blockDim.x;
      if (current_col < M && tile_col < L) {
        Bs[ty * TILE_WIDTH + tx] = B[tile_col * M + current_col];
      } else {
        Bs[ty * TILE_WIDTH + tx] = 0.0f;
      }

      __syncthreads();

#pragma unroll
      for (uint k = 0; k < TILE_WIDTH; k++) {
        sum[e] += As[ty * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + tx];
      }

      __syncthreads();
    }
  }

  for (uint e = 0; e < N_ELEM_PER_THREAD; e++) {
    uint current_col = col_start + e * blockDim.x;
    if (row < N && current_col < M) {
      C[row * M + current_col] = sum[e];
    }
  }
}

__global__ void mat_mul_K2_first_T(const float *A, const float *B, float *C,
                                   uint N, uint L, uint M) {
  extern __shared__ float shared_mem[];
  float *As = shared_mem;
  uint TILE_WIDTH = blockDim.x;

  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
    uint tile_col = tile * TILE_WIDTH + threadIdx.x;
    uint tile_row = tile * TILE_WIDTH + threadIdx.y;

    if (tile_col < L && row < N)
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[tile_col * N + row];
    else
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;

    if (tile_row < L && col < M)
      Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[tile_row * M + col];
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

__global__ void mat_mul_K2_second_T(const float *A, const float *B, float *C,
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
    C[row * M + col] = sum;
  }
}

__global__ void mat_mul_K2_both_T(const float *A, const float *B, float *C,
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
      As[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[tile_col * N + row];
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
    C[row * M + col] = sum;
  }
}
void mat_mul_dispatcher(float *A, float *B, float *C, uint N, uint L, uint M,
                        bool is_first_T, bool is_second_T, int sqrt_block_size,
                        int version = 0, cudaStream_t stream = nullptr) {
  size_t shared_mem_size =
      2 * sqrt_block_size * sqrt_block_size * sizeof(float);

  dim3 grid_dim((M + sqrt_block_size - 1) / sqrt_block_size,
                (N + sqrt_block_size - 1) / sqrt_block_size);
  dim3 block_dim(sqrt_block_size, sqrt_block_size);

  if (!is_first_T && !is_second_T) {
    if (version > 2) {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_none<<<grid_dim, block_dim, shared_mem_size, stream>>>(
          A, B, C, N, L, M);
    if (version == 1)
      mat_mul_K2_optimized<<<grid_dim, block_dim, shared_mem_size, stream>>>(
          A, B, C, N, L, M);
    else if (version == 2) {
      const int TILE_WIDTH = 32;
      constexpr uint N_ELEM_PER_THREAD = 2; // Set your desired value here
      dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
      dim3 dimGrid((M + TILE_WIDTH * N_ELEM_PER_THREAD - 1) /
                       (TILE_WIDTH * N_ELEM_PER_THREAD),
                   (N + TILE_WIDTH - 1) / TILE_WIDTH);
      size_t shared_mem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

      mat_mul_K2_optimized_v2<N_ELEM_PER_THREAD>
          <<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, N, L, M);
    }
  } else if (is_first_T && !is_second_T) {
    if (version > 0) {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_first_T<<<grid_dim, block_dim, shared_mem_size, stream>>>(
          A, B, C, N, L, M);
  } else if (!is_first_T && is_second_T) {
    if (version > 0) {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_second_T<<<grid_dim, block_dim, shared_mem_size, stream>>>(
          A, B, C, N, L, M);
  } else if (is_first_T && is_second_T) {
    if (version > 0) {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_both_T<<<grid_dim, block_dim, shared_mem_size, stream>>>(
          A, B, C, N, L, M);
  }
}