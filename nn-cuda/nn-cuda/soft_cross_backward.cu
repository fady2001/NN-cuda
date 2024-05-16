#include "common.hpp"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#define TEST_PYTORTH true

template <class T>
void crossentropy_softmax_backward_cpu(T* down_grads,const T* probs, const int* targets, int N, int C)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < C; j++) {
			down_grads[j + i * C] = probs[j + i * C] - ((j == targets[i]) ? 1.0f : 0.0f);
		}
	}
}

template <class T>
__global__ void crossentropy_softmax_backward_kernel(T* down_grads, const T* probs, const int* targets, int N, int C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		for (int j = 0; j < C; j++) {
			down_grads[j + i * C] = probs[j + i * C] - ((j == targets[i]) ? 1.0f : 0.0f);
		}
	}
}

// kernel launcher
template<class T>
void crossentropy_softmax_backward(T* down_grads, const T* probs, const T* targets, int N, int C, const int block_size) {
	const int grid_size = (N + block_size - 1) / block_size;
	crossentropy_softmax_backward_kernel << <grid_size, block_size >> > (down_grads, probs, targets, N, C);
	cudaCheck(cudaGetLastError());
}

int main()
{
	srand(0);
	float* down_grads;
	float* h_probs;
	int *h_targets;
	const unsigned long C = 3;
	const unsigned long N = 3;
	
	int deviceIdx = 0;
	cudaCheck(cudaSetDevice(deviceIdx));

	down_grads = make_random_float(N * C);
	h_targets = make_random_int(N, C);
	h_probs = make_random_float(N * C);

#if TEST_PYTORTH
  write_npy("cross-entropy-backward\\down_grads.npy", down_grads, 2, new unsigned long[2]{N, C});
  write_npy("cross-entropy-backward\\h_targets.npy", h_targets, 1, new unsigned long[1]{N});
  write_npy("cross-entropy-backward\\h_probs.npy", h_probs, 2, new unsigned long[2]{N, C});
#endif

	// CPU
	crossentropy_softmax_backward_cpu(down_grads, h_probs, h_targets, N, C);

#if TEST_PYTORTH
write_npy("cross-entropy-backward\\h_dlogits_after.npy", down_grads, 2, new unsigned long[2]{N, C});
#endif

	free(down_grads);
	free(h_targets);
	free(h_probs);
	return 0;
}