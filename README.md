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

#### Colab notebooks
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IaCnNqV8m58uGBqW9KSij61eAydVZC4C) unit tests
- benchmarks
  - l-ten vs libtorch training and inference (MNIST)
  - l-ten vs libtorch training and inference (Speech Commands)

#### Unit tests build instructions (linux only)
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Build the light-tensor library and unit tests
  * change directory to the folder into which you downloaded the source code
  * from the command line type ***make unit-tests***
- Download the MNIST data set
  https://data.deepai.org/mnist.zip
- Uncompress the downloaded MNIST archive files
- Run the unit tests
   *  from the command line type ***test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -cpu*** to run the CPU unit tests (replace ***path_to_xxx*** with the actual paths to the ***xxx*** MNIST files)
   *  from the command line type ***test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -gpu*** to run the GPU unit tests


#### Benchmarks build instructions (l-tensor vs libtorch, linux only)
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Build the light-tensor library and unit tests
  * change directory to the folder into which you downloaded the source code
  * from the command line type ***make benchmarks***
- Download the MNIST data set
  https://data.deepai.org/mnist.zip
- Download the Speech Commands data set (pre-processed and converted to spectrograms)
  *  https://speechcommands.s3-us-west-2.amazonaws.com/speech_commands.zip
- Download the libtorch library (C++ backend for Pytorch)
  *  https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.8.0.zip
- Uncompress the downloaded MNIST archive files
- Uncompress the pre-processed Speech Commands archive file
- Uncompress the libtorch archive file
- Run the benchmarks
  *  add the libtorch library to the library search path (env LD_LIBRARY_PATH=/usr/local/libtorch/lib)
  *  from the command line type ***benchmark-mnist path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte 20*** to run the MNIST benchmarks (replace ***path_to_xxx*** with the actual paths to the ***xxx*** MNIST files)
  *  from the command line type ***benchmark-speech-cmds path_to_training_list.txt path_to_testing_list.txt speech_commands/ speech_commands/ 30*** to run the Speech Commands benchmark tests (replace ***path_to_xxx*** with the actual paths to the ***xxx*** files and replace ***speech_commands*** with the actual folder into which Speech Commands archive file was uncompressed into)

