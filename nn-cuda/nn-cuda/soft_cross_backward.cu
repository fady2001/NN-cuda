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

// kernel for previous function
template<class T>
__global__ void crossentropy_softmax_backward_kernel(T* dlogits,
	const T* dlosses, const T* probs, const int* targets,
	int N, int C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		for (int j = 0; j < C; j++) {
			const T kronecker = (j == targets[i]) ? 1.0f : 0.0f;
			dlogits[j + i * C] += (probs[j + i * C] - kronecker) * dlosses[i];
		}
	}
}

// kernel launcher
template<class T>
void crossentropy_softmax_backward(T* dlogits, const T* dlosses, const T* probs, const int* targets, int N, int C, const int block_size) {
	const int grid_size = (N + block_size - 1) / block_size;
	crossentropy_softmax_backward_kernel << <grid_size, block_size >> > (dlogits, dlosses, probs, targets, N, C);
	cudaCheck(cudaGetLastError());
}

int main()
{
	srand(0);
	float h_dlogits[] = { 0.0000000e+00 ,1.5702642e-03 ,9.9842972e-01,4.0329954e-01 ,5.9670043e-01 ,0.0000000e+00,2.1250953e-05 ,9.9997878e-01 ,0.0000000e+00 };
	float h_dlosses[] = { 1.5715e-03, 9.0808e-01, 2.1219e-05 };
	float h_probs[] = { 0, 0.00157026 ,0.9984297,0.40329954 ,0.59670043 ,0,2.1250953e-05 ,9.9997878e-01 ,0.0000000e+00 };
	int h_targets[] = { 2,0,1 };
	const unsigned long C = 3;
	const unsigned long N = 3;
	
	int deviceIdx = 0;
	cudaCheck(cudaSetDevice(deviceIdx));

	//h_dlogits = make_random_float(N * C);
	//h_targets = make_random_int(N, C);
	//h_probs = make_random_float(N * C);
	//h_dlosses = make_random_float(N * 1);

#if TEST_PYTORTH
  write_npy("cross-entropy-backward\\h_dlogits_before.npy", h_dlogits, 2, new unsigned long[2]{N, C});
  write_npy("cross-entropy-backward\\h_targets.npy", h_targets, 1, new unsigned long[1]{N});
  write_npy("cross-entropy-backward\\h_probs.npy", h_probs, 2, new unsigned long[2]{N, C});
  write_npy("cross-entropy-backward\\h_dlosses.npy", h_dlosses, 1, new unsigned long[1]{N});
#endif

	// CPU
	crossentropy_softmax_backward_cpu(h_dlogits, h_probs, h_targets, N, C);

#if TEST_PYTORTH
write_npy("cross-entropy-backward\\h_dlogits_after.npy", h_dlogits, 2, new unsigned long[2]{N, C});
#endif

	free(h_dlogits);
	free(h_dlosses);
	free(h_probs);
	free(h_targets);
	return 0;
}