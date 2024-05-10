#include "ModelLayersKernels.cuh"

class ModelLayersKernelsLaunchers {
public:
    template<class T>
    void linear_layer(T* X, T* W, T* bias, T* y, int B, int N, int M, int sqrt_block_size)
    {
 	dim3 block(sqrt_block_size, sqrt_block_size);
 	dim3 grid((B + block.x - 1) / block.x, (M + block.y - 1) / block.y);
 	ModelLayersKernels::linear_layer_forward_naive << <grid, block >> > (X, W, bias, y, B, N, M);
 	cudaCheck(cudaDeviceSynchronize());
    }

    template<class T>
    void run_relu_kernel(T* input, T* output, int B, int N, int sqrt_block_size)
 {
 	dim3 block(sqrt_block_size, sqrt_block_size);
 	dim3 grid((B + block.x - 1) / block.x, (N + block.y - 1) / block.y);
 	ModelLayersKernels::relu_forward << <grid, block >> > (input, output, B, N);
 	cudaCheck(cudaDeviceSynchronize());
 }

 template <class T>
 void run_softmax_kernel(const T* input, T* output, int N, int C, int block_size)
 {
 	int num_blocks = (N + block_size - 1) / block_size;
 	ModelLayersKernels::softmax_kernel << <num_blocks, block_size >> > (input, output, N, C);
    cudaCheck(cudaDeviceSynchronize());
 }

 template <class T>
 void run_cross_entropy_kernel(T* losses, const T* probs, const int* targets, int N, int C, const int block_size)
 {
 	//(dividend + divisor - 1) / divisor
 	const int grid_size = (N + block_size - 1) / block_size;
 	ModelLayersKernels::cross_entropy_kernel << <grid_size, block_size >> > (losses, probs, targets, N, C);
 	cudaCheck(cudaGetLastError());
 }

 template <class T>
 void run_array_sum_kernel3(T* d_a, T* d_result, int size, int block_size)
 {
 	// (dividend + divisor - 1) / divisor
 	int num_blocks = (size + block_size - 1) / block_size;
 	ModelLayersKernels::array_sum_kernel3 << <1, num_blocks, block_size * sizeof(T) >> > (d_a, d_result, size);
 	cudaCheck(cudaGetLastError());
 }
}