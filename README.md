<div align="center">

# <a name="_60ra4m4w095l"></a>**Cuda Neural Network**

**Note that we followed pytorch kernels names.**

</div>

# <a name="_qhlfkzdtotwf"></a>**Our architecture**
We implemented a simple neural Network in Cuda and Trained it for a classification problem. We also implemented a similar pytorch model to compare it with our model.

<div align="center">

![structure](https://github.com/fady2001/NN-cuda/assets/75928317/7849c361-2280-4f07-87e3-6b622e6ea407)

</div>

1. **Linear Layer**
1. **Activation function (RelU)**
1. **Numerical stable softmax** 
1. **NLL Loss**
1. **SGD optimizer**

Later we merge the softmax with NLLLoss to be in one kernel **(Cross entropy kernel)**
# <a name="_c97u9wgjfs8c"></a>**Math notes behind the scene**
- We implemented logSoftmax from the beginning to ensure numerical stability and subtract the maximum from the input to avoid overflow logSoftmax(xi)=xi-log(j=0nexj-max(x0->n))
# <a name="_puvi6d8wylbn"></a>**Implemented kernels**
1. **For forward propagation**
   1. Linear forward kernel
   1. RelU forward kernel
   1. Log softmax kernel
   1. NLL loss kernel
   1. Cross Entropy kernel
1. **For backward propagation**
   1. Cross Entropy backward kernel
   1. Linear Backward kernel
   1. ReLU backward kernel
   1. SGD optimizer kernel
1. **Common helper kernels**
   1. Three kernels for reduction (Sum, Mean)
   1. Five kernels for Matrix multiplication
   1. Reduce On Axis

<div align="center">
  
![loss](https://github.com/fady2001/NN-cuda/assets/75928317/ed3abdf4-1eb9-43b6-93f1-b55fab67753f)

</div>

# <a name="_6y58qyjqwd7x"></a>**Performance analysis**
**Floating point operations for each implementation device**
## <a name="_6d17uy57u9uo"></a>**GPU**
- GPU 3050 RTX ti 
- 5.299 TFLOPS
- Bandwidth: 192.0 GB/s
## <a name="_1zl5d0ul3ist"></a>**CPU** 
- CPU: i7-11800 
- 4.6 Ghz
- 73.6 GFlops per core
## <a name="_6u50tp412r2n"></a>**RAM**
- Ram bandwidth: 51.1 GB/s.



**For B = 10000,C=10000,M,N,L all = 10000**

|**layer**|**Compute Complexity**|**Memory Complexity Reads/Writes**|**CPU - time**|**GPU - time**|**Theoretical speedup**|
| :-: | :-: | :-: | :- | :- | :- |
|**softmax**|5BC|N\*C/N\*C  4B|22\.47 ms |0\.0943ms +4.167ms =4.26ms|x5.274|
|**NLLLoss**|B|N\*C+N/N  4B|7\.84ms|0+0.0212=0.0212ms|x369.8|
|**Cross Entropy**|5BC+B|(N\*C\*+N)/(N\*C+N) 4B|22\.417 ms|0\.094ms+4.1ms=4.194ms|x5.34|
|**Matrix multiplication**|MN(2L) |ML+LN/MN 4B|27201 ms|377\.42ms+6.25ms=383.67ms|<p>**X70.8**</p><p>**This is due to the fact that the operation is very arithmetic focused**</p>|
|**Array reduction**|B|N/1 4B|0\.0009 ms|0+0=0.0002ms|~no speedup|
|**RelU**|BN|BN/BN|17\.04ms|0\.0188ms+4.1667ms=4.185ms|x4|
##



## <a name="_a07vc9f56qpc"></a><a name="_w4uzx37ehzjn"></a>**Practical comparison Forward propagation on 10000\*10000**

|**layer**|**CPU**|**GPU**|**speedup**|
| :-: | :-: | :-: | :-: |
|**softmax**|1327\.024048 ms|84\.1993 ms|~x16|
|**NLLLoss**|0\.225000 ms|0\.0155 ms|~x15|
|**Cross Entropy**|1430\.562988 ms|31\.2810 ms|~x45|
|**RelU**|790\.794983 ms|0\.0074 ms |~x106864|
|**Linear Layer (2048^3)**|28367\.611328 ms|49\.7811 ms|~x570|
|**Array Reduction**|0\.027000 ms|0\.017562 ms|~x2|
|**SGD**|0\.022000 ms|0\.0070 ms|~x3|



## <a name="_kacoxb5r5k9i"></a>**GPU compared with pytorch**

|**layer**|**Cuda Forward**|**Pytorch Forward**|
| :-: | :-: | :-: |
|**softmax**|84\.1993 ms|8\.226 ms|
|**NLLLoss**|0\.0155 ms|0\.291530 ms|
|**Cross Entropy**|31\.2810 ms|8\.053ms|
|**Linear Layer**|49\.7811 ms|` `5.419ms|
|**RelU**|0\.0074 ms|4\.551ms|










## <a name="_yel6u6a4m68e"></a>**Practical comparison Backward propagation on 10000\*10000**

|**layer**|**CPU**|**GPU**|**speedup**|
| :-: | :-: | :-: | :-: |
|**Cross Entropy**|0|0\.0050 ms |overhead|
|**Linear Backward ((2048^3))**|No end|98\.2392 ms|inf|
|**RelU**|860\.95 ms|0\.0103 ms|83587|


## <a name="_8cbiksmnx9tl"></a>**Theoretical VS practical speed**
Several factors can contribute to the observed speedup being lower than the theoretical one:

- **Memory Transfer Overhead:** Moving data between the CPU and GPU can incur significant latency. Optimizing data transfer by minimizing the frequency and size of transfers can help.
- **Kernel Launch Overhead:** The time taken to launch GPU kernels can affect performance. Overlapping data transfer with computation (using techniques such as CUDA streams) can help.
- **Suboptimal GPU Utilization:** Not fully utilizing the GPU's computational units can reduce performance. Ensuring that the workload is large enough and properly distributed across the GPU can improve utilization.
- **Algorithm Optimization:** The specific algorithm and its implementation can have a significant impact. Optimizing the algorithm for parallel execution and leveraging GPU-specific libraries can enhance performance.

To achieve a better speedup:

- **Optimize Data Transfer:** Use pinned memory and asynchronous data transfers.
- **Kernel Fusion:** Combine multiple small kernels into a larger one to reduce launch overhead.
- **Algorithm Tuning:** Refine the algorithm to better exploit GPU architecture.
# <a name="_g847ao60h5xs"></a>**Files description**
- **File for each layer contains the following:**
  - Kernel and kernel launcher
  - equivalent cpu layer
  - Main to test the layer, benchmark the difference in speed and check the both GPU and CPU output same results
  - Writing inputs and outputs to .npy file to be tested in python.
- **ModelLayers.hpp**
  - Class contains Layers implemented in CPU ,taken from each separate file, as a static member void functions.
- **Kernels.cuh**
  - Cuda header file contains all layers kernels taken from each separate file
- **KernelLauncher.cuh**
  - Cuda header class contain static void member function to run each kernel exists in kernels.cuh
- **ModelMemoryHandler.hpp**
  - A class handles anything related to memory
    - allocating / deallocating layers memory
    - A function mimics to\_cuda() in pytorch to move our model from cpu to gpu.
  - Contain to essential structs
    - Parameter: to store linear layers parameters
    - Activation: to store the output of the layer
- **Main\_training.cu**: training the gpu model
- **Main\_training\_merged\_cross.cu**
  - Training the gpu model but replacing the softmax followed by NLLLoss with cross entropy
- **Main\_training\_cpu.cpp**: training the cpu model
- **Main\_training\_cpu\_merged\_cross.cu**
  - Training the cpu model but replacing the softmax followed by NLLLoss with cross entropy
- **Mat\_muls\_kernels.cu:** Contains all multiplication kernels
## <a name="_ucywkavxrjx7"></a>**Kernel Fusion**
We apply kernel fusion in three places

1. Merging softmax with NLLLoss to perform Cross entropy in the forward propagation that reduce time from 84.2 ms to 31 ms
1. Merging softmax with NLLLoss to perform Cross entropy in the backward propagation, which also helps in algorithm optimization. Instead of calculating jacobian matrix, we simply subtract the input from the output of the softmax in the forward pass

## <a name="_jk6jkjwokdvi"></a>**Streaming**
We only had a change of streaming that when one batch is doing its forward and backward, the following batch would then load its data into memory then wait till weights are updated from the previous mini-batch.

This results in the following Timeline:

<div align="center">

![streaming](https://github.com/fady2001/NN-cuda/assets/75928317/d8e1790b-68a1-4c40-883a-19e43a247c17)

</div>

Where the green parts are the loading of the data from another stream.

One way to improve this approach was to use more than 2 streams which will all load their data while waiting for the update of the previous batches but we would be then limited by the VRAM of the GPU and also the batch size will need to be smaller So it would be somekind of a tradeoff.
##
## <a name="_l2aig8ae39fv"></a><a name="_tbyk8f2tgzbe"></a>**Time of Training:** 
**Pytorch**:

Total time: 6662.16ms

**Without Streaming:**

Total time: 982.333435ms

**With Streaming** 

Total time: 922.212280ms

**6% speedup.**

