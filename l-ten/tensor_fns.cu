#include <assert.h>
#include <cstdint>
#include <stdio.h>
#include "error.h"

//const int DEFA_BLOCK_X = 32;
//const int DEFA_BLOCK_Y = 32;
//const int DEFA_THREADS = 1024;

//const int DEFA_REDUCTION_THREADS = 256;
//const int MAX_REDUCTION_BLOCKS = 64;
const int LTEN_MAX_WARPS_PER_BLOCK = 16;
const int CUDA_WARP_SIZE = 32;
#define FULL_MASK 0xffffffff

extern __shared__ float shared_mem_block[];

template<typename Dtype>
__global__ void gpu_mean_kernel(Dtype* dst, const Dtype* src, const uint64_t numels, const Dtype scale)
{
	Dtype* shared_memory;
	shared_memory = (Dtype*)shared_mem_block;
	int threadId;
	int gridSize;
	Dtype val;
	uint64_t i;
	int warpId = threadIdx.x / warpSize;
	int laneId = threadIdx.x % warpSize;
	int warps = blockDim.x / warpSize;

	gridSize = gridDim.x * blockDim.x;
	threadId = blockIdx.x * blockDim.x + threadIdx.x;

	val = 0;
	for (i = threadId; i < numels; i += gridSize)
	{
		val += src[i];
	}

	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		shared_memory[warpId] = val;
	}
	__syncthreads();

	if (warpId == 0)
	{
		val = 0;
		if (threadIdx.x < warps)
		{
			val = shared_memory[threadIdx.x];
		}

		val += __shfl_down_sync(FULL_MASK, val, 16);
		val += __shfl_down_sync(FULL_MASK, val, 8);
		val += __shfl_down_sync(FULL_MASK, val, 4);
		val += __shfl_down_sync(FULL_MASK, val, 2);
		val += __shfl_down_sync(FULL_MASK, val, 1);


		if (laneId == 0)
		{
			atomicAdd((float*)dst, val * scale);
		}
	}

}

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	int num_blocks;
	int threads_per_block;
	int shared_mem_size;
	float magic_factor = 10.5f; // empirical value from benchmarking

	cudaMemsetAsync(dst, 0, sizeof(Dtype) * numels); // get this going now...

	float work_per_thread = log2(numels) * magic_factor;
	float threads = numels / work_per_thread;
	float warps = max(1.0f, threads / CUDA_WARP_SIZE);
	float warps_per_block = min((float)LTEN_MAX_WARPS_PER_BLOCK, warps);

	num_blocks = max(1, (int)(warps / warps_per_block));
	threads_per_block = CUDA_WARP_SIZE * warps_per_block;
	shared_mem_size = sizeof(Dtype) * (threads_per_block / CUDA_WARP_SIZE); // one shared memory location per warp


	//Note: This kernel assumes warps per block <= warpSize so that a single warp can perform the 'final' reduction (ok for now since current CUDA requires <= warpSize warps per block)
	gpu_mean_kernel << <num_blocks, threads_per_block, shared_mem_size >> > (dst, src, numels, static_cast<Dtype>(1.0f / numels));

}


template void gpu_mean<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_mean<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_mean<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);