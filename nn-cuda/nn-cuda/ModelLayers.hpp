#pragma once

enum REDUCTION
{
    SUM,
    MEAN,
};

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
    static void softmax_cpu(const T *in, T *out, int N, int C)
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
                out[i * C + j] = exp(in[i * C + j] - max_val);
                sum += out[i * C + j];
            }
            // output softmaxed values
            for (int j = 0; j < C; j++)
            {
                out[i * C + j] /= sum;
            }
        }
    }

    template <class T>
    static void cross_entropy_cpu(T *losses, const T *input, const int *targets, int N, int C)
    {
        // output: losses is (N) of the individual losses for each batch
        // input: input are (N,C) of the probabilities from softmax
        // input: targets is (N) of integers giving the correct index in logits
        for (int i = 0; i < N; i++)
        {
            losses[i] = -log(input[i * C + targets[i]]);
        }
    }

    template <class T>
    static void array_sum_cpu(T *out, const T *in, int N, REDUCTION reduction = SUM)
    {
        switch (reduction)
        {
        case SUM:
        {
            T sum = 0;
            for (int i = 0; i < N; ++i)
            {
                sum += in[i];
            }
            *out = sum;
            break;
        }
        case MEAN:
        {
            T sum = 0;
            for (int i = 0; i < N; ++i)
            {
                sum += in[i];
            }
            *out = sum / N;
            break;
        }
        }
    }
};