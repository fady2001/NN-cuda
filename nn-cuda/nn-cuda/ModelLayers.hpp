#pragma once


class ModelLayers
{
public:
    template <class T>
    static void linear_layer_forward_cpu(T *X, T *W, T *bias, T *y, int B, int N, int M)
    {
        for (int i = 0; i < B; i++)
        {
            for (int j = 0; j < M; j++)
            {
                y[i * M + j] = bias[j];
                for (int k = 0; k < N; k++)
                {
                    y[i * M + j] += X[i * N + k] * W[j * N + k];
                }
            }
        }
    }

    template <class T>
    static void relu_forward_cpu(T *input, T *output, int B, int N)
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

    template <class T>
    static void log_softmax_cpu(T *in, T *out, int N, int C)
    {
        // loop over each row. each row will get softmaxed
        for (int i = 0; i < N; i++)
        {
            // assume that the first element in the row is the maximum
            T max_val = in[i * C];
            // loop to get the maximum value of each row
            for (int j = 1; j < C; j++)
            {
                if (in[i * C + j] > max_val)
                {
                    max_val = in[i * C + j];
                }
            }

            T sum = 0;
            // loop over the row to calculate the sum and apply normalization
            for (int j = 0; j < C; j++)
            {
                // apply normalization step to ensure that the maximum value will be 0 to avoid overflow
                in[i * C + j] = in[i * C + j] - max_val;
                sum += exp(in[i * C + j]);
            }
            // output softmaxed values
            for (int j = 0; j < C; j++)
            {
                out[i * C + j] = in[i * C + j] - log(sum);
            }
        }
    }

    template <class T>
    static void cross_entropy_cpu(T *losses, T *input, uint *targets, int N, int C)
    {
        // output: losses is (N) of the individual losses for each batch
        // input: input are (N,C) of the probabilities from softmax
        // input: targets is (N) of integers giving the correct index in logits
        for (int i = 0; i < N; i++)
        {
            losses[i] = -input[i * C + targets[i]];
        }
    }

    template <class T>
    static void reduce_cpu(T *out, T *in, int N, REDUCTION reduction = MEAN)
    {
        if (reduction == MAX)
        {
            *out = in[0];
            for (int i = 1; i < N; ++i)
            {
                if (in[i] > *out)
                {
                    *out = in[i];
                }
            }
            return;
        }
        T sum = 0;
        for (int i = 0; i < N; ++i)
        {
            sum += in[i];
        }
        *out = sum;
        if (reduction == MEAN)
        {
            *out /= N;
        }
    }

    /*###################################################################################
    #								BACK PROPAGATION
    #
    #####################################################################################*/

    /**
     * @brief
     *
     * @tparam T
     * @param down_grads: output tensor of shape (N, C)
     * @param probs: input tensor of shape (N, C) where N is the batch size (number of rows) and C (number of columns) is the number of classes
     * @param targets : target tensor of shape (N) contains number from 0 to C-1
     * @param N : number of rows
     * @param C : number of columns
     * @return __global__
     */
    template <class T>
    static void crossentropy_softmax_backward_cpu(T *down_grads, T *log_softmax, uint *targets, int N, int C, REDUCTION reduction = MEAN)
    {
        if (reduction == MEAN)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < C; j++)
                {
                    down_grads[j + i * C] = (exp(log_softmax[j + i * C]) - ((j == targets[i]) ? 1.0f : 0.0f)) / N;
                }
            }
        }
        else
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < C; j++)
                {
                    down_grads[j + i * C] = exp(log_softmax[j + i * C]) - ((j == targets[i]) ? 1.0f : 0.0f);
                }
            }
        }
    }

    static void relu_backward_cpu(float *input, float *grad_output, float *grad_input, int B, int N)
    {
        for (int i = 0; i < B; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int idx = i * N + j;
                grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
            }
        }
    }

    static void mat_mul_cpu(const float *A, const float *B, float *C, uint N, uint L, uint M)
    {
        for (uint i = 0; i < N; i++)
        {
            for (uint j = 0; j < M; j++)
            {
                float sum = 0;
                for (uint k = 0; k < L; k++)
                {
                    sum += A[i * L + k] * B[k * M + j];
                }
                C[i * M + j] = sum;
            }
        }
    }

    static void reduce_on_axis_cpu(const float *A, float *out, uint N, uint M)
    {
        for (uint j = 0; j < M; j++)
        {
            float sum = 0;
            for (uint k = 0; k < N; k++)
            {
                sum += A[k * M + j];
            }
            out[j] = sum;
        }
    }

    static void SGD_cpu(float *params_memory, const float *grads_memory, long num_parameters, float learning_rate = 1e-3, float weight_decay = 0.0)
    {
        for (int i = 0; i < num_parameters; i++)
        {
            params_memory[i] -= learning_rate * (grads_memory[i] + weight_decay * params_memory[i]);
        }
    }

    static float *transpose(const float *A, uint N, uint M)
    {
        float *out = (float *)malloc(N * M * sizeof(float));
        for (uint i = 0; i < N; i++)
        {
            for (uint j = 0; j < M; j++)
            {
                out[j * N + i] = A[i * M + j];
            }
        }
        return out;
    }

    static void run_linear_backward_cpu(const float *inp, const float *weight,
                                        const float *up_grad, float *dLdw, float *dLdb,
                                        float *dLdx, uint B, uint N, uint M)
    {
        float *up_grad_T = transpose(up_grad, B, M);
        float *inp_T = transpose(inp, B, N);
        float *weight_T = transpose(weight, M, N);
        mat_mul_cpu(up_grad_T, inp, dLdw, M, B, N);
        reduce_on_axis_cpu(up_grad, dLdb, B, M);
        mat_mul_cpu(up_grad, weight, dLdx, B, M, N);
        free(up_grad_T);
        free(inp_T);
        free(weight_T);
    }
};