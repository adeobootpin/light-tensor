#include <assert.h>
#include <cstdint>
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
__global__ void gpu_mean_kernel_shfl(Dtype* dst, const Dtype* src, const uint64_t numels, const uint32_t real_work_unit_size, const uint32_t reduction_work_unit_size, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	float val;
	int i;
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
		//offset_dst = blockIdx.x * 4 + i;
		//offset_dst = blockIdx.x + i * gridDim.x;
		//offset_dst = blockIdx.x;

		offset_src = offset_dst * 768;

		val = src[offset_src + tid];

		if (tid >= 256)
		{
			val += src[offset_src + tid + 256];
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
				//dst[offset_dst] = val * scale;
				((float*)&sum)[i] = val * scale;
			}
		}

		offset_dst++;
	}

	if (warpIdx == 0 && tid == 0)
	{
		*(reinterpret_cast<float4*>(&dst[offset_dst - 4])) = sum;
	}

}

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes)
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

	scale = 1.0f / len;

	gpu_mean_kernel_shfl<Dtype> << <num_blocks / 4, threads_per_block / 2 >> > (dst, src, numels, len, threads_per_block, scale);

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
