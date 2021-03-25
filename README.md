# light-tensor framework (l-ten)

### Low overhead framework for fast neural network training
### Fast and efficient inference in production
### Faster GPU training and inference than libtorch (the C++ backend for PyTorch)
### Quick turn-around on network architecture and hyper-parameter experiments


#### Features
- Auto differentiation
- Popular network layer types
- Popular optimizers
- int8 quantization (linear layer only for now)
- Similar API to libtorch



#### Notebooks
- Unit tests
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adeobootpin/light-tensor/blob/main/l_ten_unit_tests.ipynb)
- Benchmarks (l-ten vs libtorch)
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adeobootpin/light-tensor/blob/main/l_ten_benchmarks.ipynb)


#### Unit tests build instructions (linux)
- [Optional] Install CUDA 10.1 and CUDNN 7.6.3 on a computer with an NVIDIA GPU (if you plan to use a GPU)
  - follow the installation instructions on the NVIDIA web pages
  - newer versions of CUDA and CUDNN should work but have not been tested
- [Optional] Install OpenBLAS 
  - on Ubuntu: sudo apt-get install libopenblas-dev
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Build the light-tensor library and unit tests
  * change directory to the folder into which you downloaded the source code
  * edit the makefile in the folder so that the source, include, library etc. folders are correct
  * from the command line type: *make unit-tests*
- Download the MNIST data set
  https://data.deepai.org/mnist.zip
- Uncompress the downloaded MNIST archive files
- Run the unit tests
   *  from the command line type: *test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -cpu* to run the CPU unit tests (replace *path_to_xxx* with the actual paths to the *xxx* MNIST files)
   *  from the command line type: *test path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte -gpu* to run the GPU unit tests


#### Benchmarks build instructions (linux)
- [Optional] Install CUDA 10.1 and CUDNN 7.6.3 on a computer with an NVIDIA GPU (if you plan to use a GPU)
  - follow the installation instructions on the NVIDIA web pages
  - newer versions of CUDA and CUDNN should work but have not been tested
- [Optional] Install OpenBLAS 
  - on Ubuntu: sudo apt-get install libopenblas-dev
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Download the libtorch library (C++ backend for Pytorch)
  *  https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.8.0.zip
- Uncompress the libtorch archive file
- Build the light-tensor library and unit tests
  * change directory to the folder into which you downloaded the source code
  * edit the *makefile* in this folder so that the source, include, and library folders are correct
  * from the command line type: *make benchmarks*
- Download the MNIST data set
  https://data.deepai.org/mnist.zip
- Download the Speech Commands data set (pre-processed and converted to spectrograms)
  *  https://speechcommands.s3-us-west-2.amazonaws.com/speech_commands.zip
- Uncompress the downloaded MNIST archive files
- Uncompress the pre-processed Speech Commands archive file
- Run the benchmarks
  *  add the libtorch library to the library search path (e.g. *export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/libtorch/lib*)
  *  from the command line type: *benchmark-mnist path_to_train-images-idx3-ubyte path_to_train-labels-idx1-ubyte path_to_t10k-images-idx3-ubyte path_to_t10k-labels-idx1-ubyte 20* to run the MNIST benchmarks 
  *  replace *path_to_xxx* in the command above with the actual paths to the *xxx* MNIST files
  *  from the command line type: *benchmark-speech-cmds path_to_training_list.txt path_to_testing_list.txt speech_commands/ speech_commands/ 30* to run the Speech Commands benchmark tests 
  *  replace *path_to_xxx* in the command above with the actual paths to the *xxx* files and replace *speech_commands/* with the actual folder into which the Speech Commands archive file was uncompressed into (terminate the folder path with *'/'*)

