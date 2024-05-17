//
//#include "common.hpp"
//#define TEST_PYTORTH true
//#define uint unsigned int
///**
// * @brief
// *  This performs generic matrix multiplication
// *  C = A * B
// *  where A is of shape (N,L)
// *  B is of shape (L,M)
// *  C is of shape (N,M)
// *  The kernel is generic and can be used for any matrix multiplication
// *
// *  @param A: input matrix A of shape (N,L) or (L,N) if is_first_T is set
// *  @param B: input matrix B of shape (L,M) or (M,L) if is_second_T is set
// *  @param C: output matrix C of shape (N,M)
// *  @param is_first_T: if true, A is transposed so the multiplication is A^T * B
// *  @param is_second_T: if true, B is transposed
// *
// */
//__global__ void mat_mul_naive(const float *A, const float *B, float *C, uint N,
//                              uint L, uint M, bool is_first_T,
//                              bool is_second_T) {
//  // this maps one thread to one output element
//  // the grid size is (B,M,1)
//  uint i = blockIdx.x * blockDim.x + threadIdx.x;
//  uint j = blockIdx.y * blockDim.y + threadIdx.y;
//  if (i < N && j < M) {
//    float sum = 0;
//    for (uint k = 0; k < L; k++) {
//      float a = is_first_T ? A[k * N + i] : A[i * L + k];
//      float b = is_second_T ? B[j * L + k] : B[k * M + j];
//      sum += a * b;
//    }
//    C[i * M + j] = sum;
//  }
//}
//
///**
// * @brief
// *  This performs reduction on a matrix
// *  Matrix is B*M
// *  The output is M
// */
//__global__ void reduce_on_axis(const float *A, float *out, uint N, uint M,
//                               bool take_avg = false) {
//  // this maps one thread per column
//  // the grid size is (B,1,1)
//  uint j = threadIdx.x + blockIdx.x * blockDim.x;
//  if (j < M) {
//    float sum = 0;
//    for (uint k = 0; k < N; k++) {
//      sum += A[k * M + j];
//    }
//    out[j] = sum;
//    if (take_avg) {
//      out[j] /= (float)N;
//    }
//  }
//}
//
//void runMatMull(float *A, float *B, float *C, uint N, uint L, uint M,
//                bool is_first_T, bool is_second_T, int sqrt_block_size) {
//  dim3 block(sqrt_block_size, sqrt_block_size);
//  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//  mat_mul_naive<<<grid, block>>>(A, B, C, N, L, M, is_first_T, is_second_T);
//  cudaCheck(cudaDeviceSynchronize());
//}
//
//void runReduceOnAxisKernel(float *d_A, float *d_out, uint N, uint M,
//                           int block_size, bool take_avg = false) {
//  // we will use 256 threads per block
//  uint grid_size = (M + block_size - 1) / block_size;
//  reduce_on_axis<<<grid_size, block_size>>>(d_A, d_out, N, M, take_avg);
//  cudaCheck(cudaPeekAtLastError());
//  cudaCheck(cudaDeviceSynchronize());
//}
//void reduce_on_axis_cpu(const float *A, float *out, uint N, uint M) {
//  for (uint j = 0; j < M; j++) {
//    float sum = 0;
//    for (uint k = 0; k < N; k++) {
//      sum += A[k * M + j];
//    }
//    out[j] = sum;
//  }
//}
//void runBackward(float *d_inp, float *d_weight, float *d_up_grad, float *d_dLdw,
//                 float *d_dLdb, float *d_dLdx, uint B, uint N, uint M,
//                 int sqrt_block_size) {
//  // compute dL/dW = (dL/dout).T * x
//  runMatMull(d_up_grad, d_inp, d_dLdw, M, B, N, true, false, sqrt_block_size);
//  // compute dL/db = avg(dL/dout, axis=0)
//  runReduceOnAxisKernel(d_up_grad, d_dLdb, B, M, sqrt_block_size, false);
//  // compute dL/dx = dL/dout * W
//  runMatMull(d_up_grad, d_weight, d_dLdx, B, M, N, false, false,
//             sqrt_block_size);
//}
//
//void mat_mul_cpu(const float *A, const float *B, float *C, uint N, uint L,
//                 uint M) {
//  for (uint i = 0; i < N; i++) {
//    for (uint j = 0; j < M; j++) {
//      float sum = 0;
//      for (uint k = 0; k < L; k++) {
//        sum += A[i * L + k] * B[k * M + j];
//      }
//      C[i * M + j] = sum;
//    }
//  }
//}
//float *transpose(const float *A, uint N, uint M) {
//  float *out = (float *)malloc(N * M * sizeof(float));
//  for (uint i = 0; i < N; i++) {
//    for (uint j = 0; j < M; j++) {
//      out[j * N + i] = A[i * M + j];
//    }
//  }
//  return out;
//}
//void run_mat_mul_cpu(const float *A, const float *B, float *C, uint N, uint L,
//                     uint M, bool is_f_T, bool is_s_T) {
//  const float *A_T = A;
//  const float *B_T = B;
//  if (is_f_T) {
//    A_T = transpose(A, N, L);
//  }
//  if (is_s_T) {
//    B_T = transpose(B, L, M);
//  }
//  mat_mul_cpu(A_T, B_T, C, N, L, M);
//  if (is_f_T) {
//    free((void *)A_T);
//  }
//  if (is_s_T) {
//    free((void *)B_T);
//  }
//}
//void run_linear_backward_cpu(const float *inp, const float *weight,
//                             const float *up_grad, float *dLdw, float *dLdb,
//                             float *dLdx, uint B, uint N, uint M) {
//  float *up_grad_T = transpose(up_grad, B, M);
//  float *inp_T = transpose(inp, B, N);
//  float *weight_T = transpose(weight, M, N);
//  mat_mul_cpu(up_grad_T, inp, dLdw, M, B, N);
//  reduce_on_axis_cpu(up_grad, dLdb, B, M);
//  mat_mul_cpu(up_grad, weight, dLdx, B, M, N);
//  free(up_grad_T);
//  free(inp_T);
//  free(weight_T);
//}
//
//void read_cuda(float *d_out, uint N, uint M, const char *name) {
//  float *out = (float *)malloc(N * M * sizeof(float));
//  cudaCheck(
//      cudaMemcpy(out, d_out, N * M * sizeof(float), cudaMemcpyDeviceToHost));
//  write_npy(name, out, 2, new unsigned long[2]{N, M});
//  free(out);
//}
//
//int main() {
//  srand(0);
//  const size_t B = 10, N = 1000, M = 350;
//
//  int deviceIdx = 0;
//  cudaCheck(cudaSetDevice(deviceIdx));
//
//  float *inp = make_random_float(B * N);
//  float *weight = make_random_float(M * N);
//  float *bias = make_random_float(M);
//  float *up_grad = make_random_float(B * M); // dL/dout
//
//// write arrays to npy files if you want to test with torch
//#if TEST_PYTORTH
//  write_npy("linear-backward\\X_c.npy", inp, 2, new unsigned long[2]{B, N});
//  write_npy("linear-backward\\W_C.npy", weight, 2, new unsigned long[2]{M, N});
//  write_npy("linear-backward\\bias_C.npy", bias, 1, new unsigned long[1]{M});
//  write_npy("linear-backward\\up_grad.npy", up_grad, 2,
//            new unsigned long[2]{B, M});
//#endif
//  float *dldw_cpu = (float *)malloc(M * N * sizeof(float));
//  float *dldb_cpu = (float *)malloc(M * sizeof(float));
//  float *dldx_cpu = (float *)malloc(B * N * sizeof(float));
//  run_linear_backward_cpu(inp, weight, up_grad, dldw_cpu, dldb_cpu, dldx_cpu, B,
//                          N, M);
//  // move to GPU
//  float *d_inp;
//  float *d_weight;
//  //  float *d_bias;
//  float *d_up_grad;
//  // three gradients we need to compute
//
//  float *d_dLdw;
//  float *d_dLdb;
//  float *d_dLdx;
//
//  cudaCheck(cudaMalloc(&d_inp, B * N * sizeof(float)));
//  cudaCheck(cudaMalloc(&d_weight, M * N * sizeof(float)));
//  //  cudaCheck(cudaMalloc(&d_bias, M * sizeof(float)));
//  cudaCheck(cudaMalloc(&d_up_grad, B * M * sizeof(float)));
//  cudaCheck(
//      cudaMemcpy(d_inp, inp, B * N * sizeof(float), cudaMemcpyHostToDevice));
//  cudaCheck(cudaMemcpy(d_weight, weight, M * N * sizeof(float),
//                       cudaMemcpyHostToDevice));
//  //  cudaCheck(
//  //      cudaMemcpy(d_bias, bias, M * sizeof(float), cudaMemcpyHostToDevice));
//  cudaCheck(cudaMemcpy(d_up_grad, up_grad, B * M * sizeof(float),
//                       cudaMemcpyHostToDevice));
//
//  cudaCheck(cudaMalloc(&d_dLdw, M * N * sizeof(float)));
//  cudaCheck(cudaMalloc(&d_dLdb, M * sizeof(float)));
//  cudaCheck(cudaMalloc(&d_dLdx, B * N * sizeof(float)));
//
//  runBackward(d_inp, d_weight, d_up_grad, d_dLdw, d_dLdb, d_dLdx, B, N, M, 32);
//
//  validate_result(d_dLdw, dldw_cpu, "dLdw", M * N, 1e-4f);
//  validate_result(d_dLdb, dldb_cpu, "dLdb", M, 1e-4f);
//  validate_result(d_dLdx, dldx_cpu, "dLdx", B * N, 1e-4f);
//
//#if TEST_PYTORTH
//  read_cuda(d_dLdw, M, N, "linear-backward\\dLdw.npy");
//  read_cuda(d_dLdb, M, 1, "linear-backward\\dLdb.npy");
//  read_cuda(d_dLdx, B, N, "linear-backward\\dLdx.npy");
//#endif
//
//  return 0;
//}