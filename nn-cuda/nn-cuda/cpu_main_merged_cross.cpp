//  #include <iostream>
//  #include <cmath>
//  #include "common.hpp"
//  #include "ModelLayers.hpp"
//  #include "ModelMemoryHandler.cuh"
//  #define TEST_PYTORTH true

//  auto reduction_type = REDUCTION::MEAN;

//  int main()
//  {
//   unsigned long input_dim = 32;
//   unsigned long B = 1024;
//   unsigned long H1 = 64;
//   unsigned long C = 8;
//   ModelMemoryHandler model(input_dim, B, H1, C, RANDOM_V, RANDOM_V);

//   // create host memory of random numbers
//   float *inp = make_random_float(B * input_dim);
//   uint *target = make_random_int(B, int(C));

//  #if TEST_PYTORTH
//   write_npy("all-model-cpu\\X_c.npy", inp, 2, new unsigned long[2]{B, input_dim});
//   write_npy("all-model-cpu\\target.npy", target, 1, new unsigned long[1]{B});
//   write_npy("all-model-cpu\\ln1w.npy", model.GetParams().ln1w, 2, new unsigned long[2]{H1, input_dim});
//   write_npy("all-model-cpu\\ln1b.npy", model.GetParams().ln1b, 1, new unsigned long[1]{H1});
//   write_npy("all-model-cpu\\ln2w.npy", model.GetParams().ln2w, 2, new unsigned long[2]{C, H1});
//   write_npy("all-model-cpu\\ln2b.npy", model.GetParams().ln2b, 1, new unsigned long[1]{C});
//  #endif

//   // run CPU
//   ModelLayers::linear_layer_forward_cpu(inp, model.GetParams().ln1w, model.GetParams().ln1b, model.GetActivations().ln1, B, input_dim, H1);
// 	write_npy("all-model-cpu\\ln1.npy", model.GetActivations().ln1, 2, new unsigned long[2]{ B, H1 });
//   ModelLayers::relu_forward_cpu(model.GetActivations().ln1, model.GetActivations().a1, B, H1);
// 	write_npy("all-model-cpu\\a1.npy", model.GetActivations().a1, 2, new unsigned long[2]{ B, H1 });
//   ModelLayers::linear_layer_forward_cpu(model.GetActivations().a1, model.GetParams().ln2w, model.GetParams().ln2b, model.GetActivations().ln2, B, H1, C);
// 	write_npy("all-model-cpu\\ln2.npy", model.GetActivations().ln2, 2, new unsigned long[2]{ B, C });
// 	ModelLayers::cross_entropy_cpu(model.GetActivations().ln2,target,model.GetActivations().sm,model.GetActivations().loss,B,C);
// 	write_npy("all-model-cpu\\loss.npy", model.GetActivations().loss, 1, new unsigned long[1]{ B});
//   ModelLayers::reduce_cpu(model.GetActivations().reduced_loss, model.GetActivations().loss, B, reduction_type);

//   // print results
//   printf("Reduced Loss: %f\n", *model.GetActivations().reduced_loss);

//   // backpropagation
//   ModelLayers::crossentropy_softmax_backward_cpu(model.GetDownstreamGradients().dsm, model.GetActivations().sm, target, B, C, reduction_type);
// 	write_npy("all-model-cpu\\dsm.npy", model.GetDownstreamGradients().dsm, 2, new unsigned long[2]{ B, C });
//   ModelLayers::run_linear_backward_cpu(model.GetActivations().a1, model.GetParams().ln2w,
//                                        model.GetDownstreamGradients().dsm, model.GetGradients().ln2w_grad,
//                                        model.GetGradients().ln2b_grad, model.GetDownstreamGradients().dln2,
//                                        B, H1, C);
// 	write_npy("all-model-cpu\\dln2.npy", model.GetDownstreamGradients().dln2, 2, new unsigned long[2]{ B,H1 });
// 	write_npy("all-model-cpu\\ln2w_grad.npy", model.GetGradients().ln2w_grad, 2, new unsigned long[2]{ C,H1 });
// 	write_npy("all-model-cpu\\ln2b_grad.npy", model.GetGradients().ln2b_grad, 1, new unsigned long[1]{ C});

// 	ModelLayers::relu_backward_cpu(model.GetActivations().ln1, model.GetDownstreamGradients().dln2,
// 		 model.GetDownstreamGradients().da1, B, H1);
// 	write_npy("all-model-cpu\\da1.npy", model.GetDownstreamGradients().da1, 2, new unsigned long[2]{ B,H1 });

//   ModelLayers::run_linear_backward_cpu(
//       inp, model.GetParams().ln1w, model.GetDownstreamGradients().da1,
//       model.GetGradients().ln1w_grad, model.GetGradients().ln1b_grad,
//       model.GetDownstreamGradients().dln1, B, input_dim, H1);
// 	write_npy("all-model-cpu\\dln1.npy", model.GetDownstreamGradients().dln1, 2, new unsigned long[2]{ B,input_dim });
// 	write_npy("all-model-cpu\\ln1w_grad.npy", model.GetGradients().ln1w_grad, 2, new unsigned long[2]{ H1,input_dim });
// 	write_npy("all-model-cpu\\ln1b_grad.npy", model.GetGradients().ln1b_grad, 1, new unsigned long[1]{ H1 });

//   ModelLayers::SGD_cpu(
//       model.GetParamsMemory(), model.GetGradientsMemory(),
//       model.get_num_parameters(), 0.01, 0.0);

//  #if TEST_PYTORTH
// 	write_npy("all-model-cpu\\updated_ln1w.npy", model.GetParams().ln1w, 2, new unsigned long[2]{ H1,input_dim });
// 	write_npy("all-model-cpu\\updated_ln1b.npy", model.GetParams().ln1b, 1, new unsigned long[1]{ H1});
// 	write_npy("all-model-cpu\\updated_ln2w.npy", model.GetParams().ln2w, 2, new unsigned long[2]{ C,H1 });
// 	write_npy("all-model-cpu\\updated_ln2b.npy", model.GetParams().ln2b, 1, new unsigned long[1]{ C });
//  #endif
//   return 0;
//  }
