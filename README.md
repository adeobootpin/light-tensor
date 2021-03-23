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

#### Unit test build instructions (linux only)
- Download the source code
  *  use the code download button at the top of this page to get the source code
- Build the light-tensor library and unit tests
  * change directory to the root of the downloaded repository (
- Download the MNIST data set
  *  http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  *  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  
  Alternate MNIST download location: https://data.deepai.org/mnist.zip
  
- Download dependencies


#### Demos
- Build the l-tensor library and run tests
- UAV dynamics simulator

#### Benchmarks
- l-ten vs libtorch training and inference (MNIST)
- l-ten vs libtorch training and inference (Speech Commands)
