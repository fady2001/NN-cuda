#pragma once
#include "common.hpp"
#include <cuda_runtime.h>
#define uint unsigned int

__global__ void mat_mul_K2_none(const float *A, const float *B, float *C,
                                uint N, uint L, uint M)
{
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
  {
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

    for (uint k = 0; k < TILE_WIDTH; k++)
    {
      sum +=
          As[threadIdx.y * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + threadIdx.x];
    }
    __syncthreads();
  }

  if (col < M && row < N)
  {
    C[row * M + col] = sum;
  }
}
#define index(row, col, M) (row * M + col)

__global__ void mat_mul_K2_optimized(const float *A, const float *B, float *C,
                                     uint N, uint L, uint M)
{
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0f;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
  {
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
    for (uint k = 0; k < TILE_WIDTH; k++)
    {
      sum +=
          As[threadIdx.y * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + threadIdx.x];
    }
    __syncthreads();
  }

  if (col < M && row < N)
  {
    C[row * M + col] = sum;
  }
}
__global__ void mat_mul_K2_optimized_v2(const float *A, const float *B,
                                        float *C, uint N, uint L, uint M)
{
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  const int TM = 2; // Number of elements each thread will compute

  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  uint row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  uint innerColA, innerRowA, innerColB, innerRowB;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < (L + TILE_WIDTH - 1) / TILE_WIDTH; ++bkIdx)
  {
    // Populate the shared memory caches
    innerRowA = threadIdx.y;
    innerColA = threadIdx.x;
    innerRowB = threadIdx.y;
    innerColB = threadIdx.x;

    for (int i = 0; i < TM; ++i)
    {
      uint tile_row = row + i * TILE_WIDTH;
      if ((bkIdx * TILE_WIDTH + innerColA) < L && tile_row < N)
      {
        As[innerRowA * TILE_WIDTH + innerColA] =
            A[tile_row * L + bkIdx * TILE_WIDTH + innerColA];
      }
      else
      {
        As[innerRowA * TILE_WIDTH + innerColA] = 0.0f;
      }

      if ((bkIdx * TILE_WIDTH + innerRowB) < L && col < M)
      {
        Bs[innerRowB * TILE_WIDTH + innerColB] =
            B[(bkIdx * TILE_WIDTH + innerRowB) * M + col];
      }
      else
      {
        Bs[innerRowB * TILE_WIDTH + innerColB] = 0.0f;
      }
    }
    __syncthreads();

    // Advance block tile for outer loop
    A += TILE_WIDTH;
    B += TILE_WIDTH * M;

    // Calculate per-thread results
    for (uint dotIdx = 0; dotIdx < TILE_WIDTH; ++dotIdx)
    {
      float Btmp = Bs[dotIdx * TILE_WIDTH + threadIdx.x];
      for (uint resIdx = 0; resIdx < TM; ++resIdx)
      {
        threadResults[resIdx] +=
            As[(threadIdx.y + resIdx * TILE_WIDTH) * TILE_WIDTH + dotIdx] *
            Btmp;
      }
    }
    __syncthreads();
  }

  // Write the results to global memory
  for (int i = 0; i < TM; ++i)
  {
    uint tile_row = row + i * TILE_WIDTH;
    if (tile_row < N && col < M)
    {
      C[tile_row * M + col] = threadResults[i];
    }
  }
}

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA)
    {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
    {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // block into registers
      for (uint i = 0; i < TM; ++i)
      {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i)
      {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
      {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
        {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
  {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
    {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}
__global__ void mat_mul_K2_first_T(const float *A, const float *B, float *C,
                                   uint N, uint L, uint M)
{
  extern __shared__ float shared_mem[];
  float *As = shared_mem;
  uint TILE_WIDTH = blockDim.x;

  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
  {
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

    for (uint k = 0; k < TILE_WIDTH; k++)
    {
      sum +=
          As[threadIdx.y * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + threadIdx.x];
    }
    __syncthreads();
  }

  if (col < M && row < N)
  {
    C[row * M + col] = sum;
  }
}

__global__ void mat_mul_K2_second_T(const float *A, const float *B, float *C,
                                    uint N, uint L, uint M)
{
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
  {
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

    for (uint k = 0; k < TILE_WIDTH; k++)
    {
      sum +=
          As[threadIdx.y * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + threadIdx.x];
    }
    __syncthreads();
  }

  if (col < M && row < N)
  {
    C[row * M + col] = sum;
  }
}

__global__ void mat_mul_K2_both_T(const float *A, const float *B, float *C,
                                  uint N, uint L, uint M)
{
  extern __shared__ float shared_mem[];
  uint TILE_WIDTH = blockDim.x;
  float *As = shared_mem;
  float *Bs = shared_mem + TILE_WIDTH * TILE_WIDTH;

  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;

  for (uint tile = 0; tile < (L + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
  {
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

    for (uint k = 0; k < TILE_WIDTH; k++)
    {
      sum +=
          As[threadIdx.y * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + threadIdx.x];
    }
    __syncthreads();
  }

  if (col < M && row < N)
  {
    C[row * M + col] = sum;
  }
}
void mat_mul_dispatcher(float *A, float *B, float *C, uint N, uint L, uint M,
                        bool is_first_T, bool is_second_T, int sqrt_block_size,
                        int version = 0)
{
  size_t shared_mem_size =
      2 * sqrt_block_size * sqrt_block_size * sizeof(float);

  dim3 grid_dim((M + sqrt_block_size - 1) / sqrt_block_size,
                (N + sqrt_block_size - 1) / sqrt_block_size);
  dim3 block_dim(sqrt_block_size, sqrt_block_size);

  if (!is_first_T && !is_second_T)
  {
    if (version > 2)
    {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_none<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N, L,
                                                                M);
    if (version == 1)
      mat_mul_K2_optimized<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N,
                                                                     L, M);
    else if (version == 2)
    {
      const int BM = 128;
      const int BN = 128;
      const int BK = 8;
      const int TM = 4;
      const int TN = 4;
      dim3 blockDim((BM * BN) / (TM * TN));
      dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));

      // Calculate the shared memory size
      size_t sharedMemSize = (BM * BK + BK * BN) * sizeof(float);

      // Launch the kernel
      sgemm2DBlocktiling<BM, BN, BK, TM, TN>
          <<<gridDim, blockDim, sharedMemSize>>>(M, N, L, 1, A, B, 0, C);
    }
  }
  else if (is_first_T && !is_second_T)
  {
    if (version > 0)
    {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_first_T<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N,
                                                                   L, M);
  }
  else if (!is_first_T && is_second_T)
  {
    if (version > 0)
    {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_second_T<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N,
                                                                    L, M);
  }
  else if (is_first_T && is_second_T)
  {
    if (version > 0)
    {
      perror("Invalid Kernel version");
      exit(EXIT_FAILURE);
    }
    if (version == 0)
      mat_mul_K2_both_T<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, N, L,
                                                                  M);
  }
}