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
//__global__ void mat_mul_naïve(const float *A, const float *B, float *C, uint N,
//                              uint L, uint M, bool is_first_T, bool is_second_T)
//{
//  // this maps one thread to one output element
//  // the grid size is (B,M,1)
//  uint i = blockIdx.x * blockDim.x + threadIdx.x;
//  uint j = blockIdx.y * blockDim.y + threadIdx.y;
//  if (i < N && j < M)
//  {
//    float sum = 0;
//    for (uint k = 0; k < L; k++)
//    {
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
//                               bool take_avg = false)
//{
//  // this maps one thread per column
//  // the grid size is (B,1,1)
//  uint j = threadIdx.x + blockIdx.x * blockDim.x;
//  if (j < M)
//  {
//    float sum = 0;
//    for (uint k = 0; k < N; k++)
//    {
//      sum += A[k * M + j];
//    }
//    out[j] = sum;
//    if (take_avg)
//    {
//      out[j] /= (float)N;
//    }
//  }
//}
//
//void runMatMull(float *A, float *B, float *C, uint N, uint L, uint M,
//                bool is_first_T, bool is_second_T, int sqrt_block_size)
//{
//  dim3 block(sqrt_block_size, sqrt_block_size);
//  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//  mat_mul_naïve<<<grid, block>>>(A, B, C, N, L, M, is_first_T, is_second_T);
//  cudaCheck(cudaDeviceSynchronize());
//}
//
//void runReduceOnAxisKernel(float *d_A, float *d_out, uint N, uint M,
//                           int block_size, bool take_avg = false)
//{
//  // we will use 256 threads per block
//  uint grid_size = (M + block_size - 1) / block_size;
//  reduce_on_axis<<<grid_size, block_size>>>(d_A, d_out, N, M, take_avg);
//  cudaCheck(cudaPeekAtLastError());
//  cudaCheck(cudaDeviceSynchronize());
//}
//
//void runBackward(float *d_inp, float *d_weight, float *d_up_grad, float *d_dLdw,
//                 float *d_dLdb, float *d_dLdx, uint B, uint N, uint M,
//                 int sqrt_block_size)
//{
//  // compute dL/dW = (dL/dout).T * x
//  runMatMull(d_up_grad, d_inp, d_dLdw, M, B, N, true, false, sqrt_block_size);
//  // compute dL/db = avg(dL/dout, axis=0)
//  runReduceOnAxisKernel(d_up_grad, d_dLdb, B, M, sqrt_block_size, false);
//  // compute dL/dx = dL/dout * W
//  runMatMull(d_up_grad, d_weight, d_dLdx, B, M, N, false, false,
//             sqrt_block_size);
//}
//
//void read_cuda(float *d_out, uint N, uint M, const char *name)
//{
//  float *out = (float *)malloc(N * M * sizeof(float));
//  cudaCheck(
//      cudaMemcpy(out, d_out, N * M * sizeof(float), cudaMemcpyDeviceToHost));
//  write_npy(name, out, 2, new unsigned long[2]{N, M});
//  free(out);
//}
//
//int main()
//{
//  srand(0);
//  const size_t B = 300, N = 1000, M = 350;
//
////   int deviceIdx = 0;
////   cudaCheck(cudaSetDevice(deviceIdx));
//
////   float *inp = make_random_float(B * N);
////   float *weight = make_random_float(M * N);
////   float *bias = make_random_float(M);
////   float *up_grad = make_random_float(B * M); // dL/dout
//
//// // write arrays to npy files if you want to test with torch
//// #if TEST_PYTORTH
////   write_npy("linear-backward\\X_c.npy", inp, 2, new unsigned long[2]{B, N});
////   write_npy("linear-backward\\W_C.npy", weight, 2, new unsigned long[2]{M, N});
////   write_npy("linear-backward\\bias_C.npy", bias, 1, new unsigned long[1]{M});
////   write_npy("linear-backward\\up_grad.npy", up_grad, 2,
////             new unsigned long[2]{B, M});
//// #endif
//
////   // move to GPU
////   float *d_inp;
////   float *d_weight;
////   //  float *d_bias;
////   float *d_up_grad;
////   // three gradients we need to compute
//
////   float *d_dLdw;
////   float *d_dLdb;
////   float *d_dLdx;
//
////   cudaCheck(cudaMalloc(&d_inp, B * N * sizeof(float)));
////   cudaCheck(cudaMalloc(&d_weight, M * N * sizeof(float)));
////   //  cudaCheck(cudaMalloc(&d_bias, M * sizeof(float)));
////   cudaCheck(cudaMalloc(&d_up_grad, B * M * sizeof(float)));
////   cudaCheck(
////       cudaMemcpy(d_inp, inp, B * N * sizeof(float), cudaMemcpyHostToDevice));
////   cudaCheck(cudaMemcpy(d_weight, weight, M * N * sizeof(float),
////                        cudaMemcpyHostToDevice));
////   //  cudaCheck(
////   //      cudaMemcpy(d_bias, bias, M * sizeof(float), cudaMemcpyHostToDevice));
////   cudaCheck(cudaMemcpy(d_up_grad, up_grad, B * M * sizeof(float),
////                        cudaMemcpyHostToDevice));
//
////   cudaCheck(cudaMalloc(&d_dLdw, M * N * sizeof(float)));
////   cudaCheck(cudaMalloc(&d_dLdb, M * sizeof(float)));
////   cudaCheck(cudaMalloc(&d_dLdx, B * N * sizeof(float)));
//
////   runBackward(d_inp, d_weight, d_up_grad, d_dLdw, d_dLdb, d_dLdx, B, N, M, 16);
//// #if TEST_PYTORTH
////   read_cuda(d_dLdw, M, N, "linear-backward\\dLdw.npy");
////   read_cuda(d_dLdb, M, 1, "linear-backward\\dLdb.npy");
////   read_cuda(d_dLdx, B, N, "linear-backward\\dLdx.npy");
//// #endif
//
////   return 0;
//// }