//#include "common.hpp"
//#include <device_launch_parameters.h>
//#include <device_atomic_functions.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <iostream>
//
//enum REDUCTION { MEAN, SUM, MAX };
//
//template <class T>
//void reduce_cpu(T* out, const T* in, int N, REDUCTION reduction) {
//    if (reduction == MAX) {
//        *out = in[0];
//        for (int i = 1; i < N; ++i) {
//            if (in[i] > *out) {
//                *out = in[i];
//            }
//        }
//        return;
//    }
//    T sum = 0;
//    for (int i = 0; i < N; ++i) {
//        sum += in[i];
//    }
//    *out = sum;
//    if (reduction == MEAN) {
//        *out /= N;
//    }
//}
//
//// kernel 1
//template <class T>
//__global__ void reduce_kernel1(T* out, const T* in, int N, REDUCTION reduction) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < N) {
//		if (reduction == MAX) {
//			atomicMax((int*)out, __float_as_int(in[i])); // Use atomicMax for floats
//		}
//		else {
//			atomicAdd(out, in[i]);
//            __syncthreads();
//            if (threadIdx.x == 0 && reduction == MEAN) {
//                    *out /= N;
//            }
//		}
//	}
//}
//
//
//// kernel 2
//template <class T>
//__global__ void reduce_kernel2(T *d_a, T *d_result, int size, REDUCTION reduction) {
//    extern __shared__ T v[];
//
//    int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
//    int start_index = threadIdx.x * amount_per_thread; 
//    int end_index = min(start_index + amount_per_thread, size); 
//    T partialsum = (reduction == MAX) ? d_a[start_index] : 0.0f;
//
//    for(int k = start_index; k < end_index; k++) {
//        if (reduction == MAX) {
//            partialsum = max(partialsum, d_a[k]);
//        } else {
//            partialsum += d_a[k];
//        }
//    }
//    v[threadIdx.x] = partialsum; 
//
//    __syncthreads();
//
//    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
//        int index = 2 * s * threadIdx.x;
//        if (index < blockDim.x) {
//            if (reduction == MAX) {
//                v[index] = max(v[index], v[index + s]);
//            } else {
//                v[index] += v[index + s];
//            }
//        }
//        __syncthreads();
//    }
//
//    if (threadIdx.x == 0) {
//        if (reduction == MEAN) {
//            d_result[0] = v[0] / size;
//        } else {
//            d_result[0] = v[0];
//        }
//    }
//}
//
//// kernel 3
//template <class T>
//__global__ void reduce_kernel3(T *d_a, T *d_result, int size, REDUCTION reduction) {
//    extern __shared__ T v[];
//    int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
//    int start_index = threadIdx.x * amount_per_thread; 
//    int end_index = min(start_index + amount_per_thread, size); 
//    T partialsum = (reduction == MAX) ? d_a[start_index] : 0.0f;
//
//    for(int k = start_index; k < end_index; k++) {
//        if (reduction == MAX) {
//            partialsum = max(partialsum, d_a[k]);
//        } else {
//            partialsum += d_a[k];
//        }
//    }
//    v[threadIdx.x] = partialsum; 
//
//    __syncthreads();
//
//    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//        if (threadIdx.x < s) {
//            if (reduction == MAX) {
//                v[threadIdx.x] = max(v[threadIdx.x], v[threadIdx.x + s]);
//            } else {
//                v[threadIdx.x] += v[threadIdx.x + s];
//            }
//        }
//        __syncthreads();
//    }
//
//    if (threadIdx.x == 0) {
//        if (reduction == MEAN) {
//            d_result[0] = v[0] / size;
//        } else {
//            d_result[0] = v[0];
//        }
//    }
//}
//
//int main() {
//	// Placeholder for actual test implementation
//	// This should include allocation of device memory, kernel launches, and copying results back to host
//	// Example usage:
//	const int N = 1024;
//	float* h_in = new float[N];
//	float* h_out = new float;
//	float* d_in, * d_out;
//
//	// Initialize input data
//	for (int i = 0; i < N; ++i) {
//		h_in[i] = static_cast<float>(i);
//	}
//
//	cudaMalloc(&d_in, N * sizeof(float));
//	cudaMalloc(&d_out, sizeof(float));
//	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemset(d_out, 0, sizeof(float));
//
//	// Choose reduction type
//	REDUCTION reduction = MAX; // Change to MEAN or MAX as needed
//
//    // Run CPU reduction
//    reduce_cpu(h_out, h_in, N, reduction);
//    std::cout << "Result from CPU: " << *h_out << std::endl;
//
//	// Launch kernel 1
//	reduce_kernel1 << <(N + 255) / 256, 256 >> > (d_out, d_in, N, reduction);
//	cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
//	std::cout << "Result from kernel 1: " << *h_out << std::endl;
//
//	// Reset output
//	cudaMemset(d_out, 0, sizeof(float));
//
//	// Launch kernel 2
//	reduce_kernel2 << <1, 256, 256 * sizeof(float) >> > (d_in, d_out, N, reduction);
//	cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
//	std::cout << "Result from kernel 2: " << *h_out << std::endl;
//
//	// Reset output
//	cudaMemset(d_out, 0, sizeof(float));
//
//	// Launch kernel 3
//	reduce_kernel3 << <1, 256, 256 * sizeof(float) >> > (d_in, d_out, N, reduction);
//	cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
//	std::cout << "Result from kernel 3: " << *h_out << std::endl;
//
//	// Clean up
//	cudaFree(d_in);
//	cudaFree(d_out);
//	delete[] h_in;
//	delete h_out;
//    return 0;
//}
