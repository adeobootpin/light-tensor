#include "tensor.h"

namespace lten {

	MISC_globals MISC_globals::misc_globals_;
	Tensor MISC_globals::null_tensor_;
#ifdef USE_THREADPOOL
	ThreadPool* MISC_globals::threadpool_ = 0;
#endif
#ifdef USE_MEMORYPOOL
	MemoryPool* MISC_globals::cpu_memorypool_ = 0;
#endif

#ifdef USE_CUDA
	CUDA_globlas CUDA_globlas::cuda_globals_;
	cublasHandle_t CUDA_globlas::cublas_handles_[MAX_DEVICES];
	cudnnHandle_t CUDA_globlas::cudnn_handles_[MAX_DEVICES];
	uint8_t CUDA_globlas::valid_cublas_handles_ = 0;
	uint8_t CUDA_globlas::valid_cudnn_handles_ = 0;
#ifdef USE_MEMORYPOOL
	MemoryPool* MISC_globals::gpu_memorypool_ = 0;
#endif
#endif
}