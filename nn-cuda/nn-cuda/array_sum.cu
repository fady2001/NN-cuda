#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

template <class T>
void array_sum_cpu(T* out, const T* in, int N) {
    T sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += in[i];
    }
    *out = sum;
}

// kernel 1
__global__ void array_sum_kernel(float* out, const float* in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(out, in[i]);
    }
}

// kernel 2
template <class T>
__global__ void array_sum_kernel2(T *d_a, T *d_result, int size)
{
    // shared memory with size of theads
    extern __shared__ T v[];

    // calculate the amount of work each thread should do like lecture
    int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
    int start_index = threadIdx.x*amount_per_thread; 
    int end_index= min(start_index + amount_per_thread,size); 
	T partialsum = 0.0f;
    for(int k = start_index ; k <end_index ; k++ )
    {
        // perform the first reduction step while loading data
        partialsum += d_a[k];
        v[threadIdx.x] = partialsum; 
    }
    // syncronize threads to make sure that all threads have finished the first reduction step and shared memeory is ready to be accessed
    __syncthreads();
    // each thread process two elements depends on s(stride)
    // partial sums are accumilated in index = 0
    for(unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x) 
        {
            v[index] += v[index + s];
        }
        __syncthreads();
    }
    // finally make the output is the index zero in shared memory
    if (threadIdx.x == 0) 
    {
        d_result[0] = v[0];
    }
}

// in this kernel we access sequantially which is faster
// note that the difference in efficiency is not significant between these two kernel as the sequential part is 1024 because we use one block
// it will be noted if we used multiple blocks and very large array
template <class T>
__global__ void array_sum_kernel3(T *d_a, T *d_result, int size)
{
    extern __shared__ T v[];
    int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
    int start_index = threadIdx.x*amount_per_thread; 
    int end_index= min(start_index + amount_per_thread,size); 
    T partialsum = 0.0f;
    for(int k = start_index ; k <end_index ; k++ )
    {
        partialsum += d_a[k];
        v[threadIdx.x] = partialsum; 
    }
    __syncthreads();
    
    /*
    The loop starts with `s` equal to half the block size (`blockDim.x`). 
    In each iteration of the loop, each thread with an index less than `s` adds the element at position `threadIdx.x + s` to the element at position `threadIdx.x` in the array `v`. 
    The operation `s>>=1` halves `s` in each iteration, effectively reducing the active size of the array by half in each step. 
    After each step, `__syncthreads()` is called to ensure that all threads have completed their computations before the next iteration begins. This is necessary because in the next iteration, some threads will be working with results computed by other threads in the current iteration.
    This process continues until `s` becomes 0, at which point all elements of the array have been added together and the total is stored in `v[0]`.
    */
	for(unsigned int s= blockDim.x/2; s>0; s>>=1) 
    { 
        if (threadIdx.x < s) 
        { 
            v[threadIdx.x] += v[threadIdx.x + s]; 
        } 
        __syncthreads(); 
    }
    if (threadIdx.x == 0) 
    {
        d_result[0] = v[0];
    }
}


// main function to test the first kernel
void test_array_sum_kernel() {
    int N = 1 << 20;
    float *h_in = new float[N];
    float *h_out = new float[1];
    float *d_in;
    float *d_out;
	cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    array_sum_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << *h_out << std::endl;
    array_sum_cpu(h_out, h_in, N);
    std::cout << "CPU result: " << *h_out << std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
}

int main()
{
    test_array_sum_kernel();
    return 0;
}