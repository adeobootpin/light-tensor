#include <assert.h>
#include <cstdint>
#include <stdio.h>
#include "utils.h"
#include "globals.h"
#include "error.h"

void* gpu_alloc(uint64_t size)
{
	void* memory = nullptr;
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaMalloc(&memory, size));
#else
	LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
	return memory;
}

void gpu_free(void* memory)
{
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaFree(memory));
#else
LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif

}


int AllocateMemoryOnGPU(void** memory_ptr_addr, uint64_t size, bool zero_memory)
{
#ifdef USE_CUDA
#ifdef USE_MEMORYPOOL
	*memory_ptr_addr = lten::MISC_globals::singleton()->get_gpu_memorypool()->AllocateMemory(size);
#else
	*memory_ptr_addr = gpu_alloc(size);
#endif

	if (zero_memory)
	{
		ZeroMemoryOnGPU(*memory_ptr_addr, size);
	}
#else
	LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
	return 0;
}


void ZeroMemoryOnGPU(void* memory, size_t size)
{
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaMemset(memory, 0, size));
#else
LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
}

void FreeMemoryOnGPU(void* memory)
{
	if (!memory)
	{
		return;
	}

#ifdef USE_CUDA
#ifdef USE_MEMORYPOOL
	lten::MISC_globals::singleton()->get_gpu_memorypool()->FreeMemory(memory);
#else
	gpu_free(memory);
#endif
#else
	LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
}


int CopyDataToGPU(void* gpu, void* host, size_t size)
{
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaMemcpy(gpu, host, size, cudaMemcpyHostToDevice));
#else
LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
	return 0;
}



int CopyDataFromGPU(void* host, void* gpu, size_t size)
{
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaMemcpy(host, gpu, size, cudaMemcpyDeviceToHost));
#else
	LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif

	return 0;
}


int GPUToGPUCopy(void* dst, void* src, size_t size)
{
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
#else
LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
	return 0;
}


int GetDevice(int* device)
{
#ifdef USE_CUDA
	LTEN_CUDA_CHECK(cudaGetDevice(device));
#else
	LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
	return 0;
}

#ifdef USE_CUDA
void cudaErrCheck_(cudaError_t stat, const char *file, int line) 
{
	if (stat != cudaSuccess) 
	{
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
	}
}

void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) 
{
	if (stat != CUDNN_STATUS_SUCCESS) 
	{
		fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
	}
}

void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
	if (stat != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf(stderr, "cuBlas Error: %d %s %d\n", cublasGetError(), file, line);
	}
}
#endif


