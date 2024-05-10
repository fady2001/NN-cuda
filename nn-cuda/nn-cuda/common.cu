#include "common.cuh"


void cuda_check(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int *make_random_int(size_t N, int V)
{
    int *arr = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}


void print_2D_Matrix(float *matrix, const char *name, int rows, int cols)
{
    printf("Matrix: %s with size %d x %d: \n", name, rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

template <class T>
void validate_result(T *device_result, const T *cpu_reference, const char *name, std::size_t num_elements, T tolerance)
{
    T *out_gpu = (T *)malloc(num_elements * sizeof(T));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(T), cudaMemcpyDeviceToHost));

    int nfaults = 0;
    for (int i = 0; i < num_elements; i++)
    {
        if (i < 5)
        {
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }
        if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance && !isnan(cpu_reference[i]))
        {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], out_gpu[i]);
            nfaults++;
            if (nfaults >= 10)
            {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }
    free(out_gpu);
}

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs &&...kernel_args)
{
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, nullptr));

    for (int i = 0; i < repeats; i++)
    {
        kernel(std::forward<KernelArgs>(kernel_args)...);
    }

    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));

    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

    return elapsed_time / repeats;
}

void *malloc_check(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}