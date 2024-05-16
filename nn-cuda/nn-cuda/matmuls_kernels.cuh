#pragma once
#include "common.hpp"
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