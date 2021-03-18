USE_CUDA=1
USE_AVX_256=1
USE_OPENBLAS=1
USE_THREADPOOL=1
USE_MEMORYPOOL=1

CUDA_INCLUDE_DIR="/usr/local/cuda-10.1/include/"
CUDNN_INCLUDE_DIR="/content/test/cuda/include/"
OPEN_BLAS_INCLUDE_DIR="/usr/include/x86_64-linux-gnu/"
LIBTORCH_INCLUDE_DIR1="/content/libtorch/include/"
LIBTORCH_INCLUDE_DIR2="/content/libtorch/include/torch/csrc/api/include"

CUDA_LIBS_DIR="/usr/local/cuda-10.1/lib64/"
CUDNN_LIBS_DIR="/content/test/cuda/lib64/"
OPEN_BLAS_LIBS_DIR="/usr/lib/x86_64-linux-gnu/openblas/"
LIBTORCH_LIBS_DIR="/content/libtorch/lib/"

CUDA_LIBS=-lcudart -lcublas -lcurand
CUDNN_LIBS=-lcudnn
OPEN_BLAS_LIBS=-lblas
LINK_LIBS=-ltensor
LIBTORCH_LIBS=-lc10 -lc10_cuda -ltorch_cuda -ltorch_cpu

CC=g++
LIB_SRC_DIR = ./l-tensor
TEST_SRC_DIR = ./tests
BENCHMARK_MNIST_SRC_DIR = ./benchmark_MNIST
BENCHMARK_SPEECH_CMDS_SRC_DIR = ./benchmark_speech_commands


LIB_CPP_FILES= \
	$(LIB_SRC_DIR)/backprop.cpp \
	$(LIB_SRC_DIR)/batchnorm.cpp \
	$(LIB_SRC_DIR)/conv2d.cpp \
	$(LIB_SRC_DIR)/dropout.cpp \
	$(LIB_SRC_DIR)/fully_connected.cpp \
	$(LIB_SRC_DIR)/gru.cpp \
	$(LIB_SRC_DIR)/layers.cpp \
	$(LIB_SRC_DIR)/math_fns.cpp \
	$(LIB_SRC_DIR)/optimizer.cpp \
	$(LIB_SRC_DIR)/tensor.cpp \
	$(LIB_SRC_DIR)/tensorimpl.cpp \
	$(LIB_SRC_DIR)/threadpool2.cpp \
	$(LIB_SRC_DIR)/utils.cpp

LIB_CU_FILES= \
	$(LIB_SRC_DIR)/im_col.cu \
	$(LIB_SRC_DIR)/math_fns.cu \
	$(LIB_SRC_DIR)/utils.cu

TEST_CPP_FILES= \
	$(TEST_SRC_DIR)/main.cpp \
	$(TEST_SRC_DIR)/unit_tests.cpp \
	$(TEST_SRC_DIR)/nn_tests.cpp

BENCHMARK_MNIST_CPP_FILES=\
	$(BENCHMARK_MNIST_SRC_DIR)/main.cpp \
	$(BENCHMARK_MNIST_SRC_DIR)/lten.cpp \
	$(BENCHMARK_MNIST_SRC_DIR)/libtorch.cpp

BENCHMARK_SPEECH_CMDS_CPP_FILES=\
	$(BENCHMARK_SPEECH_CMDS_SRC_DIR)/main.cpp \
	$(BENCHMARK_SPEECH_CMDS_SRC_DIR)/lten.cpp \
	$(BENCHMARK_SPEECH_CMDS_SRC_DIR)/libtorch.cpp

LIB_OBJ_FILES = $(LIB_CPP_FILES:.cpp=.o)
TEST_OBJ_FILES = $(TEST_CPP_FILES:.cpp=.o)
BENCHMARK_OBJ_FILES = $(BENCHMARK_MNIST_CPP_FILES:.cpp=.o)

CXXFLAGS=-I$(LIB_SRC_DIR) -m64 -MMD -O2 -Wno-delete-incomplete
NVCC_FLAGS=-c -O2
BENCHMARK_FLAGS=-O2

DEPENDS := $(patsubst %.cpp,%.d,$(LIB_CPP_FILES))

ifeq ($(USE_CUDA), 1)
LIB_OBJ_FILES+= $(LIB_CU_FILES:.cu=.cu.o)
CXXFLAGS+=-I$(CUDA_INCLUDE_DIR) -I$(CUDNN_INCLUDE_DIR) -D"USE_CUDA"
LINK_LIBS+=$(CUDA_LIBS) $(CUDNN_LIBS)
LIBS_DIR+=-L$(CUDNN_LIBS_DIR) -L$(CUDA_LIBS_DIR) -L"/usr/local/cuda/lib64/"
NVCC_TARGETS=$(LIB_CU_FILES:.cu=.cu.o)
NVCC_FLAGS+=-D"USE_CUDA"
BENCHMARK_FLAGS+=-D"USE_CUDA"
else
NVCC_TARGETS=
endif

ifeq ($(USE_AVX_256),1)
CXXFLAGS+=-D"USE_AVX_256" -mavx
NVCC_FLAGS+=-D"USE_AVX_256"
BENCHMARK_FLAGS+=-D"USE_AVX_256 -mavx"
endif


ifeq ($(USE_THREADPOOL), 1)
CXXFLAGS+=-D"USE_THREADPOOL"
NVCC_FLAGS+=-D"USE_THREADPOOL"
BENCHMARK_FLAGS+=-D"USE_THREADPOOL"
endif

ifeq ($(USE_MEMORYPOOL), 1)
CXXFLAGS+=-D"USE_MEMORYPOOL"
NVCC_FLAGS+=-D"USE_MEMORYPOOL"
BENCHMARK_FLAGS+=-D"USE_MEMORYPOOL"
endif

ifeq ($(USE_OPENBLAS), 1)
CXXFLAGS+=-D"USE_OPENBLAS" -I$(OPEN_BLAS_INCLUDE_DIR)
LINK_LIBS+=$(OPEN_BLAS_LIBS)
LIBS_DIR+=-L$(OPEN_BLAS_LIBS_DIR)
NVCC_FLAGS+=-D"USE_OPENBLAS"
BENCHMARK_FLAGS+=-D"USE_OPENBLAS"
endif



all: $(NVCC_TARGETS) libtensor.a test $(TEST_OBJ_FILES) benchmark-mnist benchmark-speech-cmds

unit-tests: $(NVCC_TARGETS) libtensor.a test
benchmarks: $(NVCC_TARGETS) libtensor.a benchmark-mnist benchmark-speech-cmds

test: $(TEST_OBJ_FILES)
	$(CC) $(CXXFLAGS) -o test $^ -Wl,--no-as-needed -ldl -lpthread -lm -lrt -L. $(LIBS_DIR) $(LINK_LIBS)


benchmark-mnist: CXXFLAGS=$(BENCHMARK_FLAGS)
benchmark-mnist: $(BENCHMARK_MNIST_CPP_FILES) 
	$(CC) $(CXXFLAGS) -o benchmark-mnist $^ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -DUSE_CUDA -std=c++17 -I$(LIB_SRC_DIR) -I$(CUDA_INCLUDE_DIR) -I$(LIBTORCH_INCLUDE_DIR1) -I$(LIBTORCH_INCLUDE_DIR2) -L. -L$(LIBTORCH_LIBS_DIR) -L$(CUDA_LIBS_DIR) -ldl $(LIBTORCH_LIBS) -lpthread -ltensor $(CUDA_LIBS) $(CUDNN_LIBS)


benchmark-speech-cmds: CXXFLAGS=$(BENCHMARK_FLAGS)
benchmark-speech-cmds: $(BENCHMARK_SPEECH_CMDS_CPP_FILES) 
	$(CC) $(CXXFLAGS) -o benchmark-speech-cmds $^ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -DUSE_CUDA -std=c++17 -I$(LIB_SRC_DIR) -I$(CUDA_INCLUDE_DIR) -I$(LIBTORCH_INCLUDE_DIR1) -I$(LIBTORCH_INCLUDE_DIR2) -L. -L$(LIBTORCH_LIBS_DIR) -L$(CUDA_LIBS_DIR) -ldl $(LIBTORCH_LIBS) -lpthread -ltensor $(CUDA_LIBS) $(CUDNN_LIBS)


libtensor.a: $(LIB_OBJ_FILES)
	ar r $@ $^
	ranlib $@


-include $(DEPENDS)

l-tensor/im_col.cu.o:
	nvcc l-tensor/im_col.cu $(NVCC_FLAGS) -I$(CUDA_INCLUDE_DIR) -o l-tensor/im_col.cu.o

l-tensor/utils.cu.o:
	nvcc l-tensor/utils.cu $(NVCC_FLAGS) -I$(CUDA_INCLUDE_DIR) -o l-tensor/utils.cu.o

l-tensor/math_fns.cu.o:
	nvcc l-tensor/math_fns.cu $(NVCC_FLAGS) -I$(CUDA_INCLUDE_DIR) -o l-tensor/math_fns.cu.o

clean:
	rm -f test benchmark $(TEST_OBJ_FILES) $(LIB_OBJ_FILES) libtensor.a $(DEPENDS)




