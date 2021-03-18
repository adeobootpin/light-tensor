#ifndef GLOBALS_H
#define GLOBALS_H


#include "threadpool2.h"
#include "memorypool.h"
#include "error.h"
#include "utils.h"



namespace lten {

	const int MAX_DEVICES = 8;

	struct MISC_globals
	{
		static inline MISC_globals* singleton()
		{
			return &misc_globals_;
		}
#ifdef USE_THREADPOOL
		static ThreadPool* get_threadpool()
		{
			if (!threadpool_)
			{
				threadpool_ = new ThreadPool;
				if (threadpool_)
				{
					threadpool_->Init(0);
				}
			}
			return threadpool_;
		}
#endif
#ifdef USE_MEMORYPOOL
		static MemoryPool* get_cpu_memorypool()
		{
			if (!cpu_memorypool_)
			{
				cpu_memorypool_ = new MemoryPool(64, 100); // 64 pools, each bit in a uint64_t represents one pool
				if (cpu_memorypool_)
				{
					LTEN_BOOL_ERR_CHECK(cpu_memorypool_->Init());
					cpu_memorypool_->SetMemoryAllocator(cpu_alloc, cpu_free);
				}
			}
			return cpu_memorypool_;
		}
#ifdef USE_CUDA
		static MemoryPool* get_gpu_memorypool()
		{
			if (!gpu_memorypool_)
			{
				gpu_memorypool_ = new MemoryPool(64, 100); // 64 pools, each bit in a uint64_t represents one pool
				if (gpu_memorypool_)
				{
					LTEN_BOOL_ERR_CHECK(gpu_memorypool_->Init());
					gpu_memorypool_->SetMemoryAllocator(gpu_alloc, gpu_free);
				}
			}
			return gpu_memorypool_;
		}
#endif
#endif//USE_MEMORYPOOL
	private:
		MISC_globals() {}
		static MISC_globals misc_globals_;
#ifdef USE_THREADPOOL
		static ThreadPool* threadpool_;
#endif
#ifdef USE_MEMORYPOOL
		static MemoryPool* cpu_memorypool_;
#ifdef USE_CUDA
		static MemoryPool* gpu_memorypool_;
#endif
#endif //USE_MEMORYPOOL
	};

#ifdef USE_CUDA
	struct CUDA_globlas
	{
		static inline CUDA_globlas* singleton()
		{
			return &cuda_globals_;
		}

		static cublasHandle_t get_cublas_handle(int device_index)
		{
			if (!(valid_cublas_handles_ & (0x1 << device_index)))
			{
				LTEN_CUBLAS_CHECK(cublasCreate(&cublas_handles_[device_index]));
				valid_cublas_handles_ |= (0x1 << device_index);
			}
			return cublas_handles_[device_index];
		}

		static cudnnHandle_t get_cudnn_handle(int device_index)
		{
			if (!(valid_cudnn_handles_ & (0x1 << device_index)))
			{
				LTEN_CUDNN_CHECK(cudnnCreate(&cudnn_handles_[device_index]));
				valid_cudnn_handles_ |= (0x1 << device_index);
			}
			return cudnn_handles_[device_index];
		}

	private:
		CUDA_globlas() {}
		static CUDA_globlas cuda_globals_;
		static cublasHandle_t cublas_handles_[MAX_DEVICES];
		static cudnnHandle_t cudnn_handles_[MAX_DEVICES];
		static uint8_t valid_cublas_handles_;
		static uint8_t valid_cudnn_handles_;
	};
#endif
}

#endif // GLOBALS_H