#include "common.h"
#define TEST_PYTORTH true

/**
 * @brief
 *  This function performs the forward pass of a ReLU activation function.
 *
 * @param input: Input tensor of shape (B, N) where B is the batch size and N is the number of elements per batch.
 * @param output: Output tensor of the same shape as the input.
 */
__global__ void relu_forward(float *input, float *output, int B, int N)
{
    // This maps one thread to one element in the input.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && j < N)
    {
        int idx = i * N + j;
        output[idx] = fmaxf(0.0f, input[idx]);
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
__global__ void relu_backward(float *input, float *grad_output, float *grad_input, int B, int N)
{
    // This maps one thread to one element in the input.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && j < N)
    {
        int idx = i * N + j;
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
    }
}

void relu_forward_cpu(float *input, float *output, int B, int N)
{
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int idx = i * N + j;
            output[idx] = fmaxf(0.0f, input[idx]);
        }
    }
}

void relu_backward_cpu(float *input, float *grad_output, float *grad_input, int B, int N)
{
    for (int i = 0; i < B; i++)
    {
        for (int j = N - 1; j >= 0; j--)
        {
            int idx = i * N + j;
            grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
        }
    }
}

void runKernel1(float *input, float *output, int B, int N, int sqrt_block_size)
{
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((B + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    relu_forward<<<grid, block>>>(input, output, B, N);
    cudaCheck(cudaDeviceSynchronize());
}

void runKernel2(float *input, float *grad_output, float *grad_input, int B, int N, int sqrt_block_size)
{
    dim3 block(sqrt_block_size, sqrt_block_size);
    dim3 grid((B + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    relu_backward<<<grid, block>>>(input, grad_output, grad_input, B, N);
    cudaCheck(cudaDeviceSynchronize());
}

int main()
{
    srand(0);
    const unsigned long B = 100, N = 100;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // Create host memory of random numbers
    float *out = (float *)malloc(B * N * sizeof(float));
    float *grad_input = (float *)malloc(B * N * sizeof(float));
    float *grad_output = make_random_float(B * N);
    float *inp = make_random_float(B * N);

#if TEST_PYTORTH
    write_npy("relu-layer\\X_relu.npy", inp, 2, new unsigned long[2]{B, N});
    write_npy("relu-layer\\up_grad_relu.npy", grad_output, 2, new unsigned long[2]{B, N});
#endif

    // Move to GPU
    float *d_out, *d_grad_input, *d_grad_output, *d_inp;
    cudaCheck(cudaMalloc(&d_out, B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grad_input, B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grad_output, B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_inp, inp, B * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grad_output, grad_output, B * N * sizeof(float), cudaMemcpyHostToDevice));

    relu_forward_cpu(inp, out, B, N);
    relu_backward_cpu(inp, grad_output, grad_input, B, N);

#if TEST_PYTORTH
    write_npy("relu-layer\\out_relu.npy", out, 2, new unsigned long[2]{B, N});
    write_npy("relu-layer\\down_grad_relu.npy", grad_input, 2, new unsigned long[2]{B, N});
#endif

    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++)
    {
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);

        runKernel1(d_inp, d_out, B, N, sqrt_block_size);
        validate_result(d_out, out, "out", B * N, 1e-4f);

        runKernel2(d_inp, d_grad_output, d_grad_input, B, N, sqrt_block_size);
        validate_result(d_grad_input, grad_input, "grad_in", B * N, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++)
    {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 100;

        float elapsed_time1 = benchmark_kernel(repeat_times, runKernel1, d_inp, d_out, B, N, sqrt_block_size);
        float elapsed_time2 = benchmark_kernel(repeat_times, runKernel2, d_inp, d_grad_output, d_grad_input, B, N, sqrt_block_size);

        float tflops1 = (float)B * N * 1 / elapsed_time1 * 1e3f / 1e12f;
        float tflops2 = (float)B * N * 1 / elapsed_time2 * 1e3f / 1e12f;

        printf("Forward pass: sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time1, tflops1);
        printf("Backward pass: sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time2, tflops2);
    }

    free(out);
    free(inp);
    free(grad_input);
    free(grad_output);

    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_grad_input));
    cudaCheck(cudaFree(d_grad_output));

    return 0;
}
