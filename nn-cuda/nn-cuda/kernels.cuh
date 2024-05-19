#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <iostream>
#define uint unsigned int

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
template <class T>
__global__ void linear_layer_forward_naive(T *X, T *W, T *bias, T *y, uint B,
                                           uint N, uint M) {
  // this maps o  ne thread to one output element
  // the grid size is (B,M,1)
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint j = blockIdx.y * blockDim.y + threadIdx.y;

  // will be used to store the dot product of the i-th row of X and the j-th row
  // of W
  if (i < B && j < M) {
    T dot_product = bias[j];
    for (unsigned long k = 0; k < N; k++) {
      dot_product += X[i * N + k] * W[j * N + k];
    }
    // store the result in y with the bias
    y[i * M + j] = dot_product;
  }
}

/**
 * @brief
 *  This function performs the forward pass of a ReLU activation function.
 *
 * @param input: Input tensor of shape (B, N) where B is the batch size and N is
 * the number of elements per batch.
 * @param output: Output tensor of the same shape as the input.
 */
template <class T>
__global__ void relu_forward(T *input, T *output, uint B, uint N) {
  // This maps one thread to one element in the input.
  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < B) {
    uint idx = row * N + col;
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

/* each thread will process only one row */
template <class T>
__global__ void log_softmax_kernel(T *in_h, T *out_d, int N, int C) {
  // input dimension (N,C)
  // output dimension (N,C)
  // get actual index in in_h and out_d
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    T max_val = in_h[i * C];
    for (int j = 1; j < C; j++) {
      if (in_h[i * C + j] > max_val) {
        max_val = in_h[i * C + j];
      }
    }

    T sum = 0;
    for (int j = 0; j < C; j++) {
      // apply normalization step to ensure that the maximum value will be 0 to
      // avoid overflow
      in_h[i * C + j] = in_h[i * C + j] - max_val;
      sum += exp(in_h[i * C + j]);
    }
    // output softmaxed values
    for (int j = 0; j < C; j++) {
      out_d[i * C + j] = in_h[i * C + j] - log(sum);
    }
  }
}

// kernel for cross_entropy
template <class T>
__global__ void nll_loss_kernel(T *losses, const T *input, const uint *targets,
                                uint N, uint C) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    losses[i] = -input[i * C + targets[i]];
  }
}

template <class T>
__global__ void reduce_kernel3(T *d_a, T *d_result, int size,
                               REDUCTION reduction) {
  extern __shared__ T v[];
  int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
  int start_index = threadIdx.x * amount_per_thread;
  int end_index = min(start_index + amount_per_thread, size);
  T partialsum = (reduction == MAX) ? d_a[start_index] : 0.0f;

  for (int k = start_index; k < end_index; k++) {
    if (reduction == MAX) {
      partialsum = max(partialsum, d_a[k]);
    } else {
      partialsum += d_a[k];
    }
  }
  v[threadIdx.x] = partialsum;

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (reduction == MAX) {
        v[threadIdx.x] = max(v[threadIdx.x], v[threadIdx.x + s]);
      } else {
        v[threadIdx.x] += v[threadIdx.x + s];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (reduction == MEAN) {
      d_result[0] = v[0] / size;
    } else {
      d_result[0] = v[0];
    }
  }
}

/*###################################################################################
#								BACK PROPAGATION
#
#####################################################################################*/

/**
 * @brief
 *
 * @tparam T
 * @param down_grads: output tensor of shape (N, C)
 * @param probs: input tensor of shape (N, C) where N is the batch size (number
 * of rows) and C (number of columns) is the number of classes
 * @param targets : target tensor of shape (N) contains number from 0 to C-1
 * @param N : number of rows
 * @param C : number of columns
 * @return __global__
 */
template <class T>
__global__ void
crossentropy_softmax_backward_kernel(T *down_grads, const T *probs,
                                     uint *targets, int N, int C,
                                     REDUCTION reduction = MEAN) {
  if (reduction == MEAN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
      for (int j = 0; j < C; j++) {
        down_grads[j + i * C] =
            (exp(probs[j + i * C]) - ((j == targets[i]) ? 1.0f : 0.0f)) / N;
      }
    }
  } else {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
      for (int j = 0; j < C; j++) {
        down_grads[j + i * C] =
            exp(probs[j + i * C]) - ((j == targets[i]) ? 1.0f : 0.0f);
      }
    }
  }
}

/**
 * @brief
 *  This function performs the backward pass of a ReLU activation function.
 *
 * @param input: Input tensor of shape (B, N) from the forward pass.
 * @param grad_output: Gradient tensor from the next layer.
 * @param grad_input: Gradient tensor to propagate back.
 */
__global__ void relu_backward(float *input, float *up_grad, float *down_grad,
                              uint B, uint N) {
  // This maps one thread to one element in the input.
  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < B) {
    uint idx = row * N + col;
    down_grad[idx] = input[idx] > 0 ? up_grad[idx] : 0;
  }
}

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

__global__ void SGD_kernel(float *params_memory, const float *grads_memory,
                           long num_parameters, float learning_rate,
                           float weight_decay) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_parameters) {
    params_memory[i] -=
        learning_rate * (grads_memory[i] + weight_decay * params_memory[i]);
  }
}

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