#include <assert.h>
#include <cstdint>
#include <tuple>
#include <stdio.h>
#include "error.h"
#include "utils.h"

//const int DEFA_BLOCK_X = 32;
//const int DEFA_BLOCK_Y = 32;
//const int DEFA_THREADS = 1024;

//const int DEFA_REDUCTION_THREADS = 256;
//const int MAX_REDUCTION_BLOCKS = 64;
const int LTEN_MAX_WARPS_PER_BLOCK = 16;
const int CUDA_WARP_SIZE = 32;
#define FULL_MASK 0xffffffff
const int vec_size = 4;
extern __shared__ float shared_mem_block[];

//-----------------------------------------------------------------------------------------------------
//
// global mean functions (i.e. mean accross all axes)
//
//-----------------------------------------------------------------------------------------------------
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
	float magic_factor = 10.5f; // empirical value derived from benchmarking

	cudaMemsetAsync(dst, 0, sizeof(Dtype)); // get this going now...

	int work_per_thread = static_cast<int>(max(1.0f, log2(numels) * magic_factor));
	int threads_required = static_cast<int>((numels + work_per_thread - 1) / work_per_thread);

	int warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	int warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;


	threads_per_block = CUDA_WARP_SIZE * warps_per_block;
	shared_mem_size = sizeof(Dtype) * (threads_per_block / CUDA_WARP_SIZE); // one shared memory location per warp


	//Note: This kernel assumes warps per block <= warpSize so that a single warp can perform the 'final' reduction (ok for now since current CUDA requires <= warpSize warps per block)
	gpu_mean_kernel << <num_blocks, threads_per_block, shared_mem_size >> > (dst, src, numels, static_cast<Dtype>(1.0f / numels));
}


template<typename Dtype>
__global__ void gpu_mean_backward_kernel(Dtype* bottom_gradient, const Dtype* top_gradient, const uint64_t numels, const Dtype scale)
{
	uint64_t i;
	Dtype top_grad;
	int gridSize;
	int threadId;

	gridSize = gridDim.x * blockDim.x;
	threadId = blockIdx.x * blockDim.x + threadIdx.x;

	top_grad = *top_gradient;

	for (i = threadId; i < numels; i += gridSize)
	{
		bottom_gradient[i] = top_grad * scale;
	}


}

template<typename Dtype>
void gpu_mean_backward(Dtype* bottom_gradient, const Dtype* top_gradient, const uint64_t numels)
{
	int num_blocks;
	int threads_per_block;


	threads_per_block = 512;

	num_blocks = (static_cast<int>(numels) + threads_per_block - 1) / threads_per_block;

	gpu_mean_backward_kernel << <num_blocks, threads_per_block >> > (bottom_gradient, top_gradient, numels, static_cast<Dtype>(1.0f / numels));

}
//-----------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------
//
// general mean functions (i.e. mean with axes)
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__device__ void warp_reduce(volatile Dtype* data, int thread_id)
{
	data[thread_id] += data[thread_id + 32];
	data[thread_id] += data[thread_id + 16];
	data[thread_id] += data[thread_id + 8];
	data[thread_id] += data[thread_id + 4];
	data[thread_id] += data[thread_id + 2];
	data[thread_id] += data[thread_id + 1];
}

template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel(Dtype* dst, const Dtype* src, const uint64_t numels, const uint32_t real_work_unit_size, const uint32_t reduction_work_unit_size, OffsetCalc_mean_std_simple offs_calc, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	Dtype* shared_memory;
	Dtype val;
	int i;

	shared_memory = (Dtype*)shared_mem_block;

	offset_dst = blockIdx.x;

	offset_src = offs_calc.GetOffsets(offset_dst);

	val = 0;

	if (offset_dst == 0 && threadIdx.x == 0)
	{
		val = 0;
	}

	if (threadIdx.x < real_work_unit_size)
	{
		offset_src += offs_calc.GetWorkspaceOffsets(threadIdx.x);
		val = src[offset_src + threadIdx.x];
	}
	shared_memory[threadIdx.x] = val;
	__syncthreads();

	for (i = reduction_work_unit_size / 2; i > 32; i >>= 1)
	{
		if (threadIdx.x < i)
		{
			shared_memory[threadIdx.x] += shared_memory[threadIdx.x + i];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32)
	{
		warp_reduce(shared_memory, threadIdx.x);
	}

	if (threadIdx.x == 0)
	{
		dst[offset_dst] = shared_memory[0] * scale;
	}
}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel11(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst_0;
	uint32_t offset_dst_1;
	uint32_t offset_dst_2;
	uint32_t offset_dst_3;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	float i;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;

	//offset_dst_0 = blockIdx.x * CUDA_WARP_SIZE * 4 + threadIdx.x;
	//offset_dst_1 = blockIdx.x * CUDA_WARP_SIZE * 4 + CUDA_WARP_SIZE + threadIdx.x;
	//offset_dst_2 = blockIdx.x * CUDA_WARP_SIZE * 4 + CUDA_WARP_SIZE * 2 + threadIdx.x;
	//offset_dst_3 = blockIdx.x * CUDA_WARP_SIZE * 4 + CUDA_WARP_SIZE * 3 + threadIdx.x;

	offset_dst_0 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * (4 * threadIdx.y + 0) + threadIdx.x;
	offset_dst_1 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * (4 * threadIdx.y + 1) + threadIdx.x;
	offset_dst_2 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * (4 * threadIdx.y + 2) + threadIdx.x;
	offset_dst_3 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * (4 * threadIdx.y + 3) + threadIdx.x;



	partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst_0);
	partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst_1);
	partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst_2);
	partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst_3);

	val_0 = 0;
	val_1 = 0;
	val_2 = 0;
	val_3 = 0;

	for (i = 0; i < len; i++)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);

		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
	}

	dst[offset_dst_0] = val_0 * scale;
	dst[offset_dst_1] = val_1 * scale;
	dst[offset_dst_2] = val_2 * scale;
	dst[offset_dst_3] = val_3 * scale;
}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel10(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst_0;
	uint32_t offset_dst_1;
	uint32_t offset_dst_2;
	uint32_t offset_dst_3;
	uint32_t offset_dst_4;
	uint32_t offset_dst_5;
	uint32_t offset_dst_6;
	uint32_t offset_dst_7;
	uint32_t offset_dst_8;
	uint32_t offset_dst_9;
	uint32_t offset_dst_10;
	uint32_t offset_dst_11;
	uint32_t offset_dst_12;
	uint32_t offset_dst_13;
	uint32_t offset_dst_14;
	uint32_t offset_dst_15;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	float val_4;
	float val_5;
	float val_6;
	float val_7;
	float val_8;
	float val_9;
	float val_10;
	float val_11;
	float val_12;
	float val_13;
	float val_14;
	float val_15;
	float i;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;
	uint32_t partial_offset_dst_4;
	uint32_t partial_offset_dst_5;
	uint32_t partial_offset_dst_6;
	uint32_t partial_offset_dst_7;
	uint32_t partial_offset_dst_8;
	uint32_t partial_offset_dst_9;
	uint32_t partial_offset_dst_10;
	uint32_t partial_offset_dst_11;
	uint32_t partial_offset_dst_12;
	uint32_t partial_offset_dst_13;
	uint32_t partial_offset_dst_14;
	uint32_t partial_offset_dst_15;

	offset_dst_0 = blockIdx.x * CUDA_WARP_SIZE * 16 + threadIdx.x;
	offset_dst_1 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE + threadIdx.x;
	offset_dst_2 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 2 + threadIdx.x;
	offset_dst_3 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 3 + threadIdx.x;
	offset_dst_4 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 4 + threadIdx.x;
	offset_dst_5 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 5 + threadIdx.x;
	offset_dst_6 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 6 + threadIdx.x;
	offset_dst_7 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 7 + threadIdx.x;
	offset_dst_8 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 8 + threadIdx.x;
	offset_dst_9 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 9 + threadIdx.x;
	offset_dst_10 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 10 + threadIdx.x;
	offset_dst_11 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 11 + threadIdx.x;
	offset_dst_12 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 12 + threadIdx.x;
	offset_dst_13 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 13 + threadIdx.x;
	offset_dst_14 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 14 + threadIdx.x;
	offset_dst_15 = blockIdx.x * CUDA_WARP_SIZE * 16 + CUDA_WARP_SIZE * 15 + threadIdx.x;


	partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst_0);
	partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst_1);
	partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst_2);
	partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst_3);
	partial_offset_dst_4 = offs_calc.GetPartialSrcOffset_d(offset_dst_4);
	partial_offset_dst_5 = offs_calc.GetPartialSrcOffset_d(offset_dst_5);
	partial_offset_dst_6 = offs_calc.GetPartialSrcOffset_d(offset_dst_6);
	partial_offset_dst_7 = offs_calc.GetPartialSrcOffset_d(offset_dst_7);
	partial_offset_dst_8 = offs_calc.GetPartialSrcOffset_d(offset_dst_8);
	partial_offset_dst_9 = offs_calc.GetPartialSrcOffset_d(offset_dst_9);
	partial_offset_dst_10 = offs_calc.GetPartialSrcOffset_d(offset_dst_10);
	partial_offset_dst_11 = offs_calc.GetPartialSrcOffset_d(offset_dst_11);
	partial_offset_dst_12 = offs_calc.GetPartialSrcOffset_d(offset_dst_12);
	partial_offset_dst_13 = offs_calc.GetPartialSrcOffset_d(offset_dst_13);
	partial_offset_dst_14 = offs_calc.GetPartialSrcOffset_d(offset_dst_14);
	partial_offset_dst_15 = offs_calc.GetPartialSrcOffset_d(offset_dst_15);

	val_0 = 0;
	val_1 = 0;
	val_2 = 0;
	val_3 = 0;
	val_4 = 0;
	val_5 = 0;
	val_6 = 0;
	val_7 = 0;
	val_8 = 0;
	val_9 = 0;
	val_10 = 0;
	val_11 = 0;
	val_12 = 0;
	val_13 = 0;
	val_14 = 0;
	val_15 = 0;

	for (i = 0; i < len; i += 4)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);
		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
		val_4 += src[offset_src + partial_offset_dst_4];
		val_5 += src[offset_src + partial_offset_dst_5];
		val_6 += src[offset_src + partial_offset_dst_6];
		val_7 += src[offset_src + partial_offset_dst_7];
		val_8 += src[offset_src + partial_offset_dst_8];
		val_9 += src[offset_src + partial_offset_dst_9];
		val_10 += src[offset_src + partial_offset_dst_10];
		val_11 += src[offset_src + partial_offset_dst_11];
		val_12 += src[offset_src + partial_offset_dst_12];
		val_13 += src[offset_src + partial_offset_dst_13];
		val_14 += src[offset_src + partial_offset_dst_14];
		val_15 += src[offset_src + partial_offset_dst_15];

		offset_src = offs_calc.GetPartialSrcOffset_s(i + 1);
		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
		val_4 += src[offset_src + partial_offset_dst_4];
		val_5 += src[offset_src + partial_offset_dst_5];
		val_6 += src[offset_src + partial_offset_dst_6];
		val_7 += src[offset_src + partial_offset_dst_7];
		val_8 += src[offset_src + partial_offset_dst_8];
		val_9 += src[offset_src + partial_offset_dst_9];
		val_10 += src[offset_src + partial_offset_dst_10];
		val_11 += src[offset_src + partial_offset_dst_11];
		val_12 += src[offset_src + partial_offset_dst_12];
		val_13 += src[offset_src + partial_offset_dst_13];
		val_14 += src[offset_src + partial_offset_dst_14];
		val_15 += src[offset_src + partial_offset_dst_15];

		offset_src = offs_calc.GetPartialSrcOffset_s(i + 2);
		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
		val_4 += src[offset_src + partial_offset_dst_4];
		val_5 += src[offset_src + partial_offset_dst_5];
		val_6 += src[offset_src + partial_offset_dst_6];
		val_7 += src[offset_src + partial_offset_dst_7];
		val_8 += src[offset_src + partial_offset_dst_8];
		val_9 += src[offset_src + partial_offset_dst_9];
		val_10 += src[offset_src + partial_offset_dst_10];
		val_11 += src[offset_src + partial_offset_dst_11];
		val_12 += src[offset_src + partial_offset_dst_12];
		val_13 += src[offset_src + partial_offset_dst_13];
		val_14 += src[offset_src + partial_offset_dst_14];
		val_15 += src[offset_src + partial_offset_dst_15];

		offset_src = offs_calc.GetPartialSrcOffset_s(i + 3);
		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
		val_4 += src[offset_src + partial_offset_dst_4];
		val_5 += src[offset_src + partial_offset_dst_5];
		val_6 += src[offset_src + partial_offset_dst_6];
		val_7 += src[offset_src + partial_offset_dst_7];
		val_8 += src[offset_src + partial_offset_dst_8];
		val_9 += src[offset_src + partial_offset_dst_9];
		val_10 += src[offset_src + partial_offset_dst_10];
		val_11 += src[offset_src + partial_offset_dst_11];
		val_12 += src[offset_src + partial_offset_dst_12];
		val_13 += src[offset_src + partial_offset_dst_13];
		val_14 += src[offset_src + partial_offset_dst_14];
		val_15 += src[offset_src + partial_offset_dst_15];
	}

	dst[offset_dst_0] = val_0 * scale;
	dst[offset_dst_1] = val_1 * scale;
	dst[offset_dst_2] = val_2 * scale;
	dst[offset_dst_3] = val_3 * scale;
	dst[offset_dst_4] = val_4 * scale;
	dst[offset_dst_5] = val_5 * scale;
	dst[offset_dst_6] = val_6 * scale;
	dst[offset_dst_7] = val_7 * scale;
	dst[offset_dst_8] = val_8 * scale;
	dst[offset_dst_9] = val_9 * scale;
	dst[offset_dst_10] = val_10 * scale;
	dst[offset_dst_11] = val_11 * scale;
	dst[offset_dst_12] = val_12 * scale;
	dst[offset_dst_13] = val_13 * scale;
	dst[offset_dst_14] = val_14 * scale;
	dst[offset_dst_15] = val_15 * scale;

}

template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel9(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst_0;
	uint32_t offset_dst_1;
	uint32_t offset_dst_2;
	uint32_t offset_dst_3;
	uint32_t offset_dst_4;
	uint32_t offset_dst_5;
	uint32_t offset_dst_6;
	uint32_t offset_dst_7;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	float val_4;
	float val_5;
	float val_6;
	float val_7;
	float i;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;
	uint32_t partial_offset_dst_4;
	uint32_t partial_offset_dst_5;
	uint32_t partial_offset_dst_6;
	uint32_t partial_offset_dst_7;

	offset_dst_0 = blockIdx.x * CUDA_WARP_SIZE * 8 + threadIdx.x;
	offset_dst_1 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE + threadIdx.x;
	offset_dst_2 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE * 2 + threadIdx.x;
	offset_dst_3 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE * 3 + threadIdx.x;
	offset_dst_4 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE * 4 + threadIdx.x;
	offset_dst_5 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE * 5 + threadIdx.x;
	offset_dst_6 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE * 6 + threadIdx.x;
	offset_dst_7 = blockIdx.x * CUDA_WARP_SIZE * 8 + CUDA_WARP_SIZE * 7 + threadIdx.x;


	partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst_0);
	partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst_1);
	partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst_2);
	partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst_3);
	partial_offset_dst_4 = offs_calc.GetPartialSrcOffset_d(offset_dst_4);
	partial_offset_dst_5 = offs_calc.GetPartialSrcOffset_d(offset_dst_5);
	partial_offset_dst_6 = offs_calc.GetPartialSrcOffset_d(offset_dst_6);
	partial_offset_dst_7 = offs_calc.GetPartialSrcOffset_d(offset_dst_7);

	val_0 = 0;
	val_1 = 0;
	val_2 = 0;
	val_3 = 0;
	val_4 = 0;
	val_5 = 0;
	val_6 = 0;
	val_7 = 0;

	for (i = 0; i < len; i+=2)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);

		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
		val_4 += src[offset_src + partial_offset_dst_4];
		val_5 += src[offset_src + partial_offset_dst_5];
		val_6 += src[offset_src + partial_offset_dst_6];
		val_7 += src[offset_src + partial_offset_dst_7];


		offset_src = offs_calc.GetPartialSrcOffset_s(i + 1);

		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
		val_4 += src[offset_src + partial_offset_dst_4];
		val_5 += src[offset_src + partial_offset_dst_5];
		val_6 += src[offset_src + partial_offset_dst_6];
		val_7 += src[offset_src + partial_offset_dst_7];
	}

	dst[offset_dst_0] = val_0 * scale;
	dst[offset_dst_1] = val_1 * scale;
	dst[offset_dst_2] = val_2 * scale;
	dst[offset_dst_3] = val_3 * scale;
	dst[offset_dst_4] = val_4 * scale;
	dst[offset_dst_5] = val_5 * scale;
	dst[offset_dst_6] = val_6 * scale;
	dst[offset_dst_7] = val_7 * scale;
}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel8(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst_0;
	uint32_t offset_dst_1;
	uint32_t offset_dst_2;
	uint32_t offset_dst_3;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	float i;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;

	offset_dst_0 = blockIdx.x * CUDA_WARP_SIZE * 4 + threadIdx.x;
	offset_dst_1 = blockIdx.x * CUDA_WARP_SIZE * 4 + CUDA_WARP_SIZE + threadIdx.x;
	offset_dst_2 = blockIdx.x * CUDA_WARP_SIZE * 4 + CUDA_WARP_SIZE * 2 + threadIdx.x;
	offset_dst_3 = blockIdx.x * CUDA_WARP_SIZE * 4 + CUDA_WARP_SIZE * 3 + threadIdx.x;


	partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst_0);
	partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst_1);
	partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst_2);
	partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst_3);

	val_0 = 0;
	val_1 = 0;
	val_2 = 0;
	val_3 = 0;

	for (i = 0; i < len; i++)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);

		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
	}

	dst[offset_dst_0] = val_0 * scale;
	dst[offset_dst_1] = val_1 * scale;
	dst[offset_dst_2] = val_2 * scale;
	dst[offset_dst_3] = val_3 * scale;
}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel7(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	float val;
	float i;
	uint32_t partial_offset_dst;

	offset_dst = blockIdx.x * CUDA_WARP_SIZE + threadIdx.x;
	
	partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

	val = 0;

	for (i = 0; i < len; i ++)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i) + partial_offset_dst;
		val += src[offset_src];
	}

	dst[offset_dst] = val * scale;
}

template<typename Dtype>
__global__ void gpu_mean_one_axis_kernel(Dtype* dst, const Dtype* __restrict__ src, const uint64_t numels, const uint32_t real_work_unit_size, const uint32_t reduction_work_unit_size, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t partial_offset_dst;
	float val;
	int i;
	int j;
	__shared__ float s[16];
	float4 sum;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	offset_dst = blockIdx.x * 4;


#ifdef __NVCC__
#pragma unroll
#endif
	for (i = 0; i < 4; i++)
	{
		val = 0;
		partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

		for (j = tid; j < len; j += blockDim.x)
		{
			offset_src = offs_calc.GetPartialSrcOffset_s(j);
			val += src[offset_src + partial_offset_dst];
		}

		val += __shfl_down_sync(FULL_MASK, val, 16);
		val += __shfl_down_sync(FULL_MASK, val, 8);
		val += __shfl_down_sync(FULL_MASK, val, 4);
		val += __shfl_down_sync(FULL_MASK, val, 2);
		val += __shfl_down_sync(FULL_MASK, val, 1);

		if (laneIdx == 0)
		{
			s[warpIdx] = val;
		}
		__syncthreads();

		val = (threadIdx.x < blockDim.x / warpSize) ? s[laneIdx] : 0;

		if (warpIdx == 0)
		{
			val += __shfl_down_sync(FULL_MASK, val, 16);
			val += __shfl_down_sync(FULL_MASK, val, 8);
			val += __shfl_down_sync(FULL_MASK, val, 4);
			val += __shfl_down_sync(FULL_MASK, val, 2);
			val += __shfl_down_sync(FULL_MASK, val, 1);

			if (tid == 0)
			{
				((float*)&sum)[i] = val * scale;
			}
		}

		offset_dst++;
	}

	if (warpIdx == 0 && tid == 0)
	{
		*(reinterpret_cast<float4*>(&dst[offset_dst - 4])) = sum;
	}

	/*
	remaining = numels_dst - (blockIdx.x * 4 + 4);

	if (remaining < 4)
	{
		offset_dst = (blockIdx.x + 1) * 4;

		for (i = 0; i < remaining; i++)
		{
			offset_src = offset_dst * stride;

			val = 0;

			for (j = tid; j < stride; j += blockDim.x)
			{
				val += src[offset_src + j];
			}

			val += __shfl_down_sync(FULL_MASK, val, 16);
			val += __shfl_down_sync(FULL_MASK, val, 8);
			val += __shfl_down_sync(FULL_MASK, val, 4);
			val += __shfl_down_sync(FULL_MASK, val, 2);
			val += __shfl_down_sync(FULL_MASK, val, 1);

			if (laneIdx == 0)
			{
				s[warpIdx] = val;
			}
			__syncthreads();

			val = (threadIdx.x < blockDim.x / warpSize) ? s[laneIdx] : 0;

			if (warpIdx == 0)
			{
				val += __shfl_down_sync(FULL_MASK, val, 16);
				val += __shfl_down_sync(FULL_MASK, val, 8);
				val += __shfl_down_sync(FULL_MASK, val, 4);
				val += __shfl_down_sync(FULL_MASK, val, 2);
				val += __shfl_down_sync(FULL_MASK, val, 1);

				if (tid == 0)
				{
					dst[offset_dst] = val * scale;
				}
			}
			offset_dst++;
		}

	}
	*/

}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	int i;
	int j;
	__shared__ float s[4][512];

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = 32 * (blockIdx.x * total_warps + warpIdx);


#ifdef __NVCC__
#pragma unroll
#endif
	for (i = 0; i < 8; i++)
	{
		partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst);
		partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst + 1);
		partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst + 2);
		partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst + 3);


		val_0 = 0;
		val_1 = 0;
		val_2 = 0;
		val_3 = 0;

		for (j = laneIdx; j < len; j += warpSize)
		{
			//offset_src = offs_calc.GetPartialSrcOffset_s(j);
			if (i == 0)
			{
				offset_src = offs_calc.GetPartialSrcOffset_s(j);
				s[warpIdx][j] = offset_src;
			}
			else
			{
				offset_src = s[warpIdx][j];
			}


			val_0 += src[offset_src + partial_offset_dst_0];
			val_1 += src[offset_src + partial_offset_dst_1];
			val_2 += src[offset_src + partial_offset_dst_2];
			val_3 += src[offset_src + partial_offset_dst_3];
		}

		val_0 += __shfl_down_sync(FULL_MASK, val_0, 16);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 8);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 4);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 2);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 1);

		val_1 += __shfl_down_sync(FULL_MASK, val_1, 16);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 8);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 4);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 2);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 1);

		val_2 += __shfl_down_sync(FULL_MASK, val_2, 16);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 8);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 4);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 2);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 1);

		val_3 += __shfl_down_sync(FULL_MASK, val_3, 16);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 8);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 4);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 2);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 1);


		if (laneIdx == 0)
		{
			dst[offset_dst] = val_0 * scale;
			dst[offset_dst + 1] = val_1 * scale;
			dst[offset_dst + 2] = val_2 * scale;
			dst[offset_dst + 3] = val_3 * scale;
		}

		offset_dst += 4;
	}

}

template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel_v(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	int i;
	int j;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = 32 * (blockIdx.x * total_warps + warpIdx);


#ifdef __NVCC__
#pragma unroll
#endif
	for (i = 0; i < 8; i++)
	{
		partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst);
		partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst + 1);
		partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst + 2);
		partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst + 3);


		val_0 = 0;
		val_1 = 0;
		val_2 = 0;
		val_3 = 0;

		for (j = laneIdx; j < len; j += warpSize)
		{
			offset_src = offs_calc.GetPartialSrcOffset_s(j);
			val_0 += src[offset_src + partial_offset_dst_0];
			val_1 += src[offset_src + partial_offset_dst_1];
			val_2 += src[offset_src + partial_offset_dst_2];
			val_3 += src[offset_src + partial_offset_dst_3];
		}

		val_0 += __shfl_down_sync(FULL_MASK, val_0, 16);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 8);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 4);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 2);
		val_0 += __shfl_down_sync(FULL_MASK, val_0, 1);

		val_1 += __shfl_down_sync(FULL_MASK, val_1, 16);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 8);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 4);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 2);
		val_1 += __shfl_down_sync(FULL_MASK, val_1, 1);

		val_2 += __shfl_down_sync(FULL_MASK, val_2, 16);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 8);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 4);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 2);
		val_2 += __shfl_down_sync(FULL_MASK, val_2, 1);

		val_3 += __shfl_down_sync(FULL_MASK, val_3, 16);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 8);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 4);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 2);
		val_3 += __shfl_down_sync(FULL_MASK, val_3, 1);


		if (laneIdx == 0)
		{
			dst[offset_dst] = val_0 * scale;
			dst[offset_dst + 1] = val_1 * scale;
			dst[offset_dst + 2] = val_2 * scale;
			dst[offset_dst + 3] = val_3 * scale;
		}

		offset_dst += 4;
	}

}

template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel_v2(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	uint32_t partial_offset_dst_0;
	uint32_t partial_offset_dst_1;
	uint32_t partial_offset_dst_2;
	uint32_t partial_offset_dst_3;
	float val_0;
	float val_1;
	float val_2;
	float val_3;
	int i;
	int j;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = 4 * (blockIdx.x * total_warps + warpIdx);

	partial_offset_dst_0 = offs_calc.GetPartialSrcOffset_d(offset_dst);
	partial_offset_dst_1 = offs_calc.GetPartialSrcOffset_d(offset_dst + 1);
	partial_offset_dst_2 = offs_calc.GetPartialSrcOffset_d(offset_dst + 2);
	partial_offset_dst_3 = offs_calc.GetPartialSrcOffset_d(offset_dst + 3);


	val_0 = 0;
	val_1 = 0;
	val_2 = 0;
	val_3 = 0;

	for (j = laneIdx; j < len; j += warpSize)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(j);
		val_0 += src[offset_src + partial_offset_dst_0];
		val_1 += src[offset_src + partial_offset_dst_1];
		val_2 += src[offset_src + partial_offset_dst_2];
		val_3 += src[offset_src + partial_offset_dst_3];
	}

	val_0 += __shfl_down_sync(FULL_MASK, val_0, 16);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 8);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 4);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 2);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 1);

	val_1 += __shfl_down_sync(FULL_MASK, val_1, 16);
	val_1 += __shfl_down_sync(FULL_MASK, val_1, 8);
	val_1 += __shfl_down_sync(FULL_MASK, val_1, 4);
	val_1 += __shfl_down_sync(FULL_MASK, val_1, 2);
	val_1 += __shfl_down_sync(FULL_MASK, val_1, 1);

	val_2 += __shfl_down_sync(FULL_MASK, val_2, 16);
	val_2 += __shfl_down_sync(FULL_MASK, val_2, 8);
	val_2 += __shfl_down_sync(FULL_MASK, val_2, 4);
	val_2 += __shfl_down_sync(FULL_MASK, val_2, 2);
	val_2 += __shfl_down_sync(FULL_MASK, val_2, 1);

	val_3 += __shfl_down_sync(FULL_MASK, val_3, 16);
	val_3 += __shfl_down_sync(FULL_MASK, val_3, 8);
	val_3 += __shfl_down_sync(FULL_MASK, val_3, 4);
	val_3 += __shfl_down_sync(FULL_MASK, val_3, 2);
	val_3 += __shfl_down_sync(FULL_MASK, val_3, 1);


	if (laneIdx == 0)
	{
		dst[offset_dst] = val_0 * scale;
		dst[offset_dst + 1] = val_1 * scale;
		dst[offset_dst + 2] = val_2 * scale;
		dst[offset_dst + 3] = val_3 * scale;
	}

}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel_v1_5(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	uint32_t partial_offset_dst;
	float val;
	int i;
	int j;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = 4 * (blockIdx.x * total_warps + warpIdx);

#ifdef __NVCC__
#pragma unroll
#endif
	for (i = 0; i < 4; i++)
	{
		partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

		val = 0;

		for (j = laneIdx; j < len; j += warpSize)
		{
			offset_src = offs_calc.GetPartialSrcOffset_s(j);
			val += src[offset_src + partial_offset_dst];
		}

		val += __shfl_down_sync(FULL_MASK, val, 16);
		val += __shfl_down_sync(FULL_MASK, val, 8);
		val += __shfl_down_sync(FULL_MASK, val, 4);
		val += __shfl_down_sync(FULL_MASK, val, 2);
		val += __shfl_down_sync(FULL_MASK, val, 1);

		if (laneIdx == 0)
		{
			dst[offset_dst] = val * scale;
		}

		offset_dst++;
	}

}



template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel_v1_3(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t partial_offset_dst;
	float val;
	int i;
	int j;


	offset_dst = (blockIdx.x * blockDim.x + threadIdx.x);
	partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

	for (i = 0; i < len; i++)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);
		val += src[offset_src + partial_offset_dst];
	}

	dst[offset_dst] = val * scale;
}

template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel_v1_2(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_src_2;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	uint32_t partial_offset_dst;
	float val_0;
	float val_1;
	int i;
	int j;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = blockIdx.x * total_warps + warpIdx;

	partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

	val_0 = 0;
	val_1 = 0;

	for (i = laneIdx; i < len; i += warpSize * 2)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);
		val_0 += src[offset_src + partial_offset_dst];

		offset_src_2 = offs_calc.GetPartialSrcOffset_s(i + warpSize);
		val_1 += src[offset_src_2 + partial_offset_dst];
	}

	val_0 += val_1;

	val_0 += __shfl_down_sync(FULL_MASK, val_0, 16);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 8);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 4);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 2);
	val_0 += __shfl_down_sync(FULL_MASK, val_0, 1);

	if (laneIdx == 0)
	{
		dst[offset_dst] = val_0 * scale;
	}

}


template<typename Dtype>
__global__ void gpu_mean_any_axes_kernel_v1(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	uint32_t partial_offset_dst;
	float val;
	int i;
	int j;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = blockIdx.x * total_warps + warpIdx;

	partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

	val = 0;

	for (i = laneIdx; i < len; i += warpSize)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);
		val += src[offset_src + partial_offset_dst];
	}

	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneIdx == 0)
	{
		dst[offset_dst] = val * scale;
	}
}

template<typename Dtype>
__global__ void gpu_mean_last_axis_only_kernel(Dtype* dst, const Dtype* src, const uint64_t numels_dst, const uint32_t stride, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	uint32_t total_warps;
	float val;
	int i;
	int j;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = blockIdx.x * total_warps + warpIdx;
	offset_src = offset_dst * stride;

	val = 0;
	for (i = laneIdx; i < stride; i += warpSize)
	{
		val += src[offset_src + i];
	}

	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneIdx == 0)
	{
		dst[offset_dst] = val * scale;
	}

}

template<typename Dtype>
__global__ void gpu_mean_last_axis_only_kernel_old_not_fastest(Dtype* dst, const Dtype* src, const uint64_t numels_dst, const uint32_t stride, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t remaining;
	float val;
	int i;
	int j;
	__shared__ float s[16];
	float4 sum;

	unsigned int tid = threadIdx.x;
	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	offset_dst = blockIdx.x * 4;

#ifdef __NVCC__
#pragma unroll
#endif
	for (i = 0; i < 4; i++)
	{
		offset_src = offset_dst * stride;

		val = 0;

		for (j = tid; j < stride; j += blockDim.x)
		{
			val += src[offset_src + j];
		}

		val += __shfl_down_sync(FULL_MASK, val, 16);
		val += __shfl_down_sync(FULL_MASK, val, 8);
		val += __shfl_down_sync(FULL_MASK, val, 4);
		val += __shfl_down_sync(FULL_MASK, val, 2);
		val += __shfl_down_sync(FULL_MASK, val, 1);

		if (laneIdx == 0)
		{
			s[warpIdx] = val;
		}
		__syncthreads();

		val = (threadIdx.x < blockDim.x / warpSize) ? s[laneIdx] : 0;

		if (warpIdx == 0)
		{
			val += __shfl_down_sync(FULL_MASK, val, 16);
			val += __shfl_down_sync(FULL_MASK, val, 8);
			val += __shfl_down_sync(FULL_MASK, val, 4);
			val += __shfl_down_sync(FULL_MASK, val, 2);
			val += __shfl_down_sync(FULL_MASK, val, 1);

			if (tid == 0)
			{
				((float*)&sum)[i] = val * scale;
			}
		}

		offset_dst++;
	}

	if (warpIdx == 0 && tid == 0)
	{
		*(reinterpret_cast<float4*>(&dst[offset_dst - 4])) = sum;
	}

	
	remaining = numels_dst - (blockIdx.x * 4 + 4);

	if(remaining < 4)
	{
		offset_dst = (blockIdx.x + 1 )* 4;

		for (i = 0; i < remaining; i++)
		{
			offset_src = offset_dst * stride;

			val = 0;

			for (j = tid; j < stride; j += blockDim.x)
			{
				val += src[offset_src + j];
			}

			val += __shfl_down_sync(FULL_MASK, val, 16);
			val += __shfl_down_sync(FULL_MASK, val, 8);
			val += __shfl_down_sync(FULL_MASK, val, 4);
			val += __shfl_down_sync(FULL_MASK, val, 2);
			val += __shfl_down_sync(FULL_MASK, val, 1);

			if (laneIdx == 0)
			{
				s[warpIdx] = val;
			}
			__syncthreads();

			val = (threadIdx.x < blockDim.x / warpSize) ? s[laneIdx] : 0;

			if (warpIdx == 0)
			{
				val += __shfl_down_sync(FULL_MASK, val, 16);
				val += __shfl_down_sync(FULL_MASK, val, 8);
				val += __shfl_down_sync(FULL_MASK, val, 4);
				val += __shfl_down_sync(FULL_MASK, val, 2);
				val += __shfl_down_sync(FULL_MASK, val, 1);

				if (tid == 0)
				{
					dst[offset_dst] = val * scale;
				}
			}
			offset_dst++;
		}

	}
}


// stride must be a multiple of 4 so that vectorization works
template<typename Dtype>
__global__ void gpu_mean_one_axis_only_vectorized_kernel(Dtype* dst, const Dtype* __restrict__ src, uint64_t stride, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	int warpId;
	int laneIdx;
	float4 val;
	float4 sum;
	int i;
	uint32_t start;
	uint32_t stop;
	const int interleave = 4;

	Dtype* shared_memory; // [4][blockDim.x]

	shared_memory = (Dtype*)shared_mem_block;

	warpId = threadIdx.x / warpSize;
	laneIdx = threadIdx.x % warpSize;
	offset_src = blockIdx.y * (len * stride) + blockIdx.x * blockDim.x + warpId * stride;
	offset_dst = blockIdx.y * stride + blockIdx.x * blockDim.x;

	sum.x = sum.y = sum.z = sum.w = 0;

	if (blockIdx.x * blockDim.x + laneIdx * vec_size >= stride)
	{
		return; // handle cases that are not multiples of 128 (i.e. warpSize * vec_size)
	}

	start = blockIdx.y * (len * stride) + blockIdx.x * blockDim.x + warpId * stride + laneIdx * vec_size;
	stop = blockIdx.y * (len * stride) + blockIdx.x * blockDim.x + 0 * stride + laneIdx * vec_size + (stride * len); // start offset of warp 0 + (stride * len)

	for (offset_src = start; offset_src < stop; offset_src += stride * 4)
	{
		val = *(reinterpret_cast<const float4*>(&src[offset_src]));

#pragma unroll
		for (i = 0; i < vec_size; i++)
		{
			((float*)&sum)[i] += ((float*)&val)[i];
		}
	}

#pragma unroll
	for (i = 0; i < vec_size; i++)
	{
		shared_memory[warpId * blockDim.x + laneIdx * 4 + i] = ((float*)&sum)[i] * scale;
	}

	__syncthreads();

	val.x = val.y = val.z = val.w = 0;

	if (warpId == 0)
	{
#pragma unroll
		for (i = 0; i < vec_size; i++)
		{
#pragma unroll
			for (int j = 0; j < interleave; j++)
			{
				((float*)&val)[i] += shared_memory[j * blockDim.x + threadIdx.x * vec_size + i];
			}			
		}

		*(reinterpret_cast<float4*>(&dst[offset_dst + threadIdx.x * 4])) = val;
	}

	/*
	float temp = shared_memory[0 * blockDim.x + threadIdx.x];

#pragma unroll
	for (i = 1; i < interleave; i++)
	{
		temp += shared_memory[i * blockDim.x + threadIdx.x];
	}

	dst[offset_dst + threadIdx.x] = temp;
	*/

}


template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels_dst, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes)
{
	int naxes;
	int i;
	int num_blocks;
	uint32_t threads_per_block;
	float scale;
	int num_warps;

	naxes = ndims_src - ndims_dst;
	if (naxes > 1 || (axes[0] != ndims_src - 1))
	{
		int len;
		int stride;

		len = 1;
		for (i = 0; i < naxes; i++)
		{
			len *= dims_src[axes[i]];
		}

		scale = 1.0f / len;

		if (naxes == 1 && !(strides_src[axes[0]] % vec_size))
		{
			stride = strides_src[axes[0]];

			int batches = numels_dst / stride;
			int interleave = 4;

			num_warps = interleave; // warps process every other interleave row

			threads_per_block = CUDA_WARP_SIZE * num_warps;

			threads_per_block = min(threads_per_block, 128);
			threads_per_block = max(threads_per_block, CUDA_WARP_SIZE);

			dim3 dimGrid;
			dimGrid.x = (stride + (CUDA_WARP_SIZE * vec_size - 1)) / (CUDA_WARP_SIZE * vec_size); // oversubscribe in case stride is not a multiple of CUDA_WARP_SIZE * vec_size (only some lanes of extra block will participate) 
			dimGrid.y = batches;
			gpu_mean_one_axis_only_vectorized_kernel<Dtype> << <dimGrid, threads_per_block, threads_per_block *  vec_size * sizeof(float) >> > (dst, src, stride, len, scale);
		}
		else
		{
			OffsetCalc_mean_var offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);

			num_blocks = static_cast<int>(numels_dst);

			threads_per_block = GetNextPowerOf2(len);
			threads_per_block = min(threads_per_block, 128);
			threads_per_block = max(threads_per_block, CUDA_WARP_SIZE);

			num_warps = threads_per_block / CUDA_WARP_SIZE;
			num_blocks = static_cast<int>(numels_dst) / (num_warps * 4 * 8);
			num_blocks = max(num_blocks, 1);

			dim3 dimBlock;
			dimBlock.x = CUDA_WARP_SIZE;
			dimBlock.y = 1;

			//gpu_mean_one_axis_kernel<Dtype> << <num_blocks/4, threads_per_block >> > (dst, src, numels_dst, len, threads_per_block, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel3<Dtype> << <num_blocks / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, numels_dst, len, threads_per_block, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel5<Dtype> << <(num_blocks / 16) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, numels_dst, len, threads_per_block, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel6<Dtype> << <(num_blocks / 16) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, offs_calc, len, scale);

			//gpu_mean_any_axes_kernel5<Dtype> << <1, CUDA_WARP_SIZE >> > (dst, src, numels_dst, len, threads_per_block, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel7<Dtype> << <(num_blocks) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel7<Dtype> << <(num_blocks) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel9<Dtype> << <(num_blocks / 8) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, offs_calc, len, scale);
			//gpu_mean_any_axes_kernel10<Dtype> << <(num_blocks / 16) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, offs_calc, len, scale);
			//dimBlock.y = 4;
			//gpu_mean_any_axes_kernel11<Dtype> << <(num_blocks / 16) / (CUDA_WARP_SIZE), dimBlock >> > (dst, src, offs_calc, len, scale);


			//num_blocks = static_cast<int>(numels_dst) / (num_warps * 4);
			//gpu_mean_any_axes_kernel_v1_5<Dtype> << <num_blocks, threads_per_block >> > (dst, src, offs_calc, len, scale);

			//num_blocks = static_cast<int>(numels_dst) / (num_warps * 4);
			//gpu_mean_any_axes_kernel_v2<Dtype> << <num_blocks, threads_per_block >> > (dst, src, offs_calc, len, scale);

			//num_blocks = static_cast<int>(numels_dst) / (num_warps);
			//gpu_mean_any_axes_kernel_v1_2<Dtype> << <num_blocks, threads_per_block >> > (dst, src, offs_calc, len, scale);


			//num_blocks = static_cast<int>(numels_dst) / (threads_per_block);
			//gpu_mean_any_axes_kernel_v1_3<Dtype> << <num_blocks, threads_per_block >> > (dst, src, offs_calc, len, scale);

		}
	}
	else
	{
		int stride;

		stride = dims_src[ndims_src - 1];

		threads_per_block = GetNextPowerOf2(static_cast<int>(stride));
		threads_per_block = min(threads_per_block, 512);
		threads_per_block = max(threads_per_block, CUDA_WARP_SIZE);

		num_warps = threads_per_block / CUDA_WARP_SIZE;
		num_blocks = static_cast<int>(numels_dst) / num_warps;
		num_blocks = max(num_blocks, 1);

		scale = 1.0f / stride;

		//gpu_mean_last_axis_only_kernel_old_not_fastest<Dtype> << <num_blocks / 4, threads_per_block >> > (dst, src, numels_dst, stride, scale);
		gpu_mean_last_axis_only_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, numels_dst, stride, scale);

	}
}
//-----------------------------------------------------------------------------------------------------




//-----------------------------------------------------------------------------------------------------
//
// transpose functions
//
//-----------------------------------------------------------------------------------------------------
const int defa_threads = 64;
constexpr int num_threads = 64;
constexpr int thread_work_size = 4;
constexpr int block_work_size = 256;


__global__ void gpu_transpose_kernel(const float* __restrict__ A, float* At, int N, OffsetCalc offs_calc)
{
	int index = threadIdx.x + block_work_size * blockIdx.x;
	float args[thread_work_size];

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (index < N)
		{
			args[i] = A[offs_calc.GetOffset(index)];
			index += num_threads;
		}
	}

	index = threadIdx.x + block_work_size * blockIdx.x;;

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (index < N)
		{
			At[index] = args[i];
			index += num_threads;
		}
	}
}




template<typename Dtype>
void gpu_transpose(const Dtype* A, Dtype* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;
	uint64_t N;


	OffsetCalc off_calc(ndims, a_strides, at_strides);


	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + 256 - 1) / 256;


	gpu_transpose_kernel << < num_blocks, defa_threads >> > ((float*)A, (float*)At, N, off_calc);

}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// general var functions (i.e. var with axes)
//
//-----------------------------------------------------------------------------------------------------
int constexpr warp_iterationz = 23;
template<typename Dtype>
__global__ void gpu_var_kernel_shfl(Dtype* dst, const Dtype* src, const uint64_t numels, const uint32_t dim_len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	float count;
	float count_b;
	float mean;
	float mean_b;
	float Ms;
	float Ms_b;
	float delta;
	float n_ab;
	float temp;
	int i;

	int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	int laneIdx = threadIdx.x % warpSize;

	offset_dst = warpIdx;

	offset_src = offset_dst * dim_len;

	count = 1.0f;
	mean = src[offset_src + laneIdx];
	Ms = 0;


#pragma unroll
	for (i = 0; i < warp_iterationz; i++)
	{
		count_b = 1;
		mean_b = src[offset_src + laneIdx + warpSize];
		Ms_b = 0;

		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;

		offset_src += warpSize;
	}


	//
	// reduce
	//
	//------------------------------------------------------
	count_b = __shfl_down_sync(FULL_MASK, count, 16);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 16);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 16);
	//if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}

	count_b = __shfl_down_sync(FULL_MASK, count, 8);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 8);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 8);
	//if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}


	count_b = __shfl_down_sync(FULL_MASK, count, 4);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 4);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 4);
	//if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}


	count_b = __shfl_down_sync(FULL_MASK, count, 2);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 2);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 2);
	//if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}


	count_b = __shfl_down_sync(FULL_MASK, count, 1);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 1);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 1);
	//if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}
	//------------------------------------------------------


	if (laneIdx == 0)
	{
		dst[offset_dst] = Ms * scale;
	}

}

template<typename Dtype>
void gpu_var(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes)
{
	uint64_t len;
	int naxes;
	int i;
	int num_blocks;
	uint32_t threads_per_block;
	float scale;

	naxes = ndims_src - ndims_dst;

	if (naxes > 1 || (axes[0] != ndims_src - 1))
	{
		LTEN_ERR("Not yet implemented: gpu_mean for naxes > 1 or axis != last axis"); // need to copy and clean up implementation from merge3
	}
	//OffsetCalc_mean_std_simple offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);

	len = 1;
	for (i = 0; i < naxes; i++)
	{
		len *= dims_src[axes[i]];
	}

	threads_per_block = GetNextPowerOf2(len);

	num_blocks = static_cast<int>(numels);

	scale = 1.0f / (len - 1);


	num_blocks = 392 * 2;
	threads_per_block = 32 * 16;
	gpu_var_kernel_shfl<Dtype> << <num_blocks, threads_per_block>> > (dst, src, numels, len, scale);
	//gpu_var_kernel_shfl<Dtype> << <num_blocks / 4, threads_per_block / 2 >> > (dst, src, numels, len, scale);
	//gpu_var_kernel_shfl<Dtype> << <392 * 2, 32 * 16 >> > (dst, src, numels, len, scale);
}


template void gpu_mean<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_mean<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_mean<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);

template void gpu_mean_backward<float>(float* bottom_gradient, const float* top_gradient, const uint64_t numels);
template void gpu_mean_backward<int>(int* bottom_gradient, const int* top_gradient, const uint64_t numels);
template void gpu_mean_backward<uint8_t>(uint8_t* bottom_gradient, const uint8_t* top_gradient, const uint64_t numels);

template void gpu_mean<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void gpu_mean<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void gpu_mean<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template void gpu_var<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void gpu_var<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void gpu_var<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template void gpu_transpose<float>(const float* A, float* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
template void gpu_transpose<int>(const int* A, int* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
template void gpu_transpose<uint8_t>(const uint8_t* A, uint8_t* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
