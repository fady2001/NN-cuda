// softmax kernel
#include <iostream>
template <class T>
void softmax(T* in, T* out, int N, int C)
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

int main()
{
    // define the input and output arrays
    float in[] = {1, 2, 3, 4};
    float out[4];
    // call the softmax function
    softmax(in, out, 1, 4);
    // print the output
    for (int i = 0; i < 4; i++)
    {
        printf("%f ", out[i]);
    }
    printf("\n");
    return 0;
}