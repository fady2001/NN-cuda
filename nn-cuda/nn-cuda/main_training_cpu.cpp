#include "ModelMemoryHandler.cuh"
#include "common.hpp"
#include "ModelLayers.hpp"
#include <cmath>
#include <vector>

#define TEST_PYTORTH true

auto reduction_type = REDUCTION::MEAN;

// Function to simulate loading a batch of data
void load_batch(trainData &data, float *curr_input, uint *current_target,
              uint B, uint batch_index)
{

  // Load the batch from the dataset
  for (int i = 0; i < B; i++)
  {
      //    load input vector
      for (int j = 0; j < data.input_shape[1]; j++)
      {
          curr_input[i * data.input_shape[1] + j] =
              data.inp[(batch_index * B + i) * data.input_shape[1] + j];
      }
      current_target[i] = data.target[batch_index * B + i];
  }
}

// Function to perform a single training step
void train_step(ModelMemoryHandler &model, float *inp, uint *target, uint B, uint input_dim, uint H1, uint C)
{

	// run CPU
	ModelLayers::linear_layer_forward_cpu(inp, model.GetParams().ln1w, model.GetParams().ln1b, model.GetActivations().ln1, B, input_dim, H1);
	ModelLayers::relu_forward_cpu(model.GetActivations().ln1, model.GetActivations().a1, B, H1);
	ModelLayers::linear_layer_forward_cpu(model.GetActivations().a1, model.GetParams().ln2w, model.GetParams().ln2b, model.GetActivations().ln2, B, H1, C);
	ModelLayers::log_softmax_cpu(model.GetActivations().ln2, model.GetActivations().sm, B, C);
	ModelLayers::cross_entropy_cpu(model.GetActivations().loss, model.GetActivations().sm, target, B, C);
	ModelLayers::reduce_cpu(model.GetActivations().reduced_loss, model.GetActivations().loss, B, reduction_type);

	// print results
	printf("Reduced Loss: %f\n", *model.GetActivations().reduced_loss);

	// backpropagation
	ModelLayers::crossentropy_softmax_backward_cpu(model.GetDownstreamGradients().dsm, model.GetActivations().sm, target, B, C, reduction_type);
	ModelLayers::run_linear_backward_cpu(model.GetActivations().a1, model.GetParams().ln2w,
		model.GetDownstreamGradients().dsm, model.GetGradients().ln2w_grad,
		model.GetGradients().ln2b_grad, model.GetDownstreamGradients().dln2,
		B, H1, C);

	ModelLayers::relu_backward_cpu(model.GetActivations().ln1, model.GetDownstreamGradients().dln2,
		model.GetDownstreamGradients().da1, B, H1);

	ModelLayers::run_linear_backward_cpu(
		inp, model.GetParams().ln1w, model.GetDownstreamGradients().da1,
		model.GetGradients().ln1w_grad, model.GetGradients().ln1b_grad,
		model.GetDownstreamGradients().dln1, B, input_dim, H1);

	ModelLayers::SGD_cpu(
		model.GetParamsMemory(), model.GetGradientsMemory(),
		model.get_num_parameters(), 0.01, 0.0);
}

int main()
{
  uint B = 32;
  trainData td{};
  read_training_data("../dataset/x_train.npy", "../dataset/y_train.npy", td,
                     true);
  uint input_dim = td.input_shape[1];
  uint H1 = 256;
  uint C = 16;
  const int EPOCHS = 10; // Number of epochs to train
  uint NUM_BATCHES = (td.input_shape[0] + B - 1) / B;
  ModelMemoryHandler h_model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

#if TEST_PYTORTH
  write_npy("trained-model-cpu\\ln1w.npy", h_model.GetParams().ln1w, 2,
            new unsigned long[2]{H1, input_dim});
  write_npy("trained-model-cpu\\ln1b.npy", h_model.GetParams().ln1b, 1,
            new unsigned long[1]{H1});
  write_npy("trained-model-cpu\\ln2w.npy", h_model.GetParams().ln2w, 2,
            new unsigned long[2]{C, H1});
  write_npy("trained-model-cpu\\ln2b.npy", h_model.GetParams().ln2b, 1,
            new unsigned long[1]{C});
#endif

  for (int epoch = 0; epoch < EPOCHS; epoch++)
  {
      printf("Epoch %d\n", epoch + 1);
      for (int batch = 0; batch < NUM_BATCHES; batch++)
      {
          // Load batch
          float *inp = (float *)malloc(B * input_dim * sizeof(float));
          uint *target = (uint *)malloc(B * sizeof(uint));
          uint batch_to_load = std::min(B, td.input_shape[0] - batch * B);
          load_batch(td, inp, target, batch_to_load, batch);

          // Train on batch
		  train_step(h_model, inp, target, batch_to_load, input_dim, H1, C);

          // Free allocated host memory for the batch
          free(inp);
          free(target);
      }
  }

  return 0;
}
