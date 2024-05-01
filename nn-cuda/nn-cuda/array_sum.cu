template <class T>
void array_sum_cpu(T* out, const T* in, int N) {
    T sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += in[i];
    }
    *out = sum;
}

// kernel 1
template <class T>
__global__ void array_sum_kernel(T* out, const T* in, int N) {
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
