# light-tensor framework (l-ten)

### Low overhead framework for fast neural network training
### Fast and efficient inference in production
### Faster than libtorch (the C++ backend for PyTorch)
### Quick turn-around on network architecture and hyper-parameter experiments


#### Features
- Auto differentiation
- Popular network layer types
- Popular optimizers
- Similar API to libtorch
- int8 quantization (linear layer only for now)
- Python frontend coming soon

#### Unit tests build instructions (linux only)
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Build the light-tensor library and unit tests
  * change directory to the folder into which you downloaded the source code
  * from the command line type ***make unit-tests***
- Download the MNIST data set
  *  http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  
  [Alternate MNIST download location: https://data.deepai.org/mnist.zip]
- Uncompress the downloaded MNIST archive files
- Run the unit tests
   *  from the command line type ***./test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -cpu*** to run the CPU unit tests (replace ***path_to_xxx*** with the actual paths to the ***xxx*** MNIST files)
   *  *  from the command line type ***./test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -gpu*** to run the GPU unit tests


#### Benchmarks (l-tensor vs libtorch) build instructions (linux only)
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Build the light-tensor library and unit tests
  * change directory to the folder into which you downloaded the source code
  * from the command line type ***make benchmarks***
- Download the MNIST data set
  *  http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  
  [Alternate MNIST download location: https://data.deepai.org/mnist.zip]
- Download the Speech Commands data set (pre-processed and converted to spectrograms)
  *  https://speechcommands.s3-us-west-2.amazonaws.com/speech_commands.zip
- Download the libtorch library (C++ backend for Pytorch)
  *  https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.8.0.zip
- Uncompress the downloaded MNIST archive files
- Uncompress the pre-processed Speech Commands data set
 - Run the unit tests
   *  from the command line type ***./test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -cpu*** to run the CPU unit tests (replace ***path_to_xxx*** with the actual paths to the ***xxx*** MNIST files)
   *  *  from the command line type ***./test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -gpu*** to run the GPU unit tests


#### Demos
- Build the l-tensor library and run tests
- UAV dynamics simulator

#### Benchmarks
- l-ten vs libtorch training and inference (MNIST)
- l-ten vs libtorch training and inference (Speech Commands)
