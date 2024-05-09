//#include "common.cuh"
//#include "device_launch_parameters.h"
//#include <cmath>
//#include <iostream>
//#define TEST_PYTORTH true
//
//template <class T>
//void crossentropy_softmax_backward_cpu(T* dlogits, const T* dlosses, const T* probs, const int* targets, int N, int C)
//{
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < C; j++) {
//			const T kronecker = (j == targets[i]) ? 1.0f : 0.0f;
//			dlogits[j + i * C] += (probs[j + i * C] - kronecker) * dlosses[i];
//		}
//	}
//}
//
//// kernel for previous function
//template<class T>
//__global__ void crossentropy_softmax_backward_kernel(T* dlogits,
//	const T* dlosses, const T* probs, const int* targets,
//	int N, int C) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < N) {
//		for (int j = 0; j < C; j++) {
//			const T kronecker = (j == targets[i]) ? 1.0f : 0.0f;
//			dlogits[j + i * C] += (probs[j + i * C] - kronecker) * dlosses[i];
//		}
//	}
//}
//
//// kernel launcher
//template<class T>
//void crossentropy_softmax_backward(T* dlogits, const T* dlosses, const T* probs, const int* targets, int N, int C, const int block_size) {
//	const int grid_size = (N + block_size - 1) / block_size;
//	crossentropy_softmax_backward_kernel << <grid_size, block_size >> > (dlogits, dlosses, probs, targets, N, C);
//	cudaCheck(cudaGetLastError());
//}
//
//int main()
//{
//	srand(0);
//	float* h_dlogits;
//	float* h_dlosses;
//	float* h_probs;
//	int* h_targets;
//	const unsigned long C = 5;
//	const unsigned long N = 3;
//	
//	int deviceIdx = 0;
//	cudaCheck(cudaSetDevice(deviceIdx));
//
//	h_dlogits = make_random_float(N * C);
//	h_targets = make_random_int(N, C);
//	h_probs = make_random_float(N * C);
//	h_dlosses = make_random_float(N * 1);
//
//#if TEST_PYTORTH
//   write_npy("cross-entropy-backward\\h_dlogits_before.npy", h_dlogits, 2, new size_t[2]{N, C});
//   write_npy("cross-entropy-backward\\h_targets.npy", h_targets, 1, new size_t[1]{N});
//   write_npy("cross-entropy-backward\\h_probs.npy", h_probs, 2, new size_t[2]{N, C});
//   write_npy("cross-entropy-backward\\h_dlosses.npy", h_dlosses, 1, new size_t[1]{N});
//#endif
//
//	// CPU
//	crossentropy_softmax_backward_cpu(h_dlogits, h_dlosses, h_probs, h_targets, N, C);
//
//#if TEST_PYTORTH
//write_npy("cross-entropy-backward\\h_dlogits_after.npy", h_dlogits, 2, new size_t[2]{N, C});
//#endif
//
//	free(h_dlogits);
//	free(h_dlosses);
//	free(h_probs);
//	free(h_targets);
//	return 0;
//}