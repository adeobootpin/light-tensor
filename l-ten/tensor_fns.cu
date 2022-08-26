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
__global__ void gpu_mean_vectorized_kernel(Dtype* dst, const Dtype* src, const uint64_t numels, const Dtype scale)
{
	Dtype* shared_memory;
	shared_memory = (Dtype*)shared_mem_block;
	int threadId;
	int gridSize;
	float4 val4;
	Dtype val;
	uint64_t i;
	int j;

	int warpId = threadIdx.x / warpSize;
	int laneId = threadIdx.x % warpSize;
	int warps = blockDim.x / warpSize;

	gridSize = gridDim.x * blockDim.x;
	threadId = blockIdx.x * blockDim.x + threadIdx.x;


	val = val4.x = val4.y = val4.z = val4.w = 0;
	for (i = threadId * vec_size; i < numels; i += gridSize * vec_size)
	{
		val4 = *(reinterpret_cast<const float4*>(&src[i]));
#pragma unroll
		for (j = 0; j < vec_size; j++)
		{
			val += ((float*)&val4)[j];
		}
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


	if (numels % 4)
	{
		//Note: This kernel assumes warps per block <= warpSize so that a single warp can perform the 'final' reduction (ok for now since current CUDA requires <= warpSize warps per block)
		gpu_mean_kernel << <num_blocks, threads_per_block, shared_mem_size >> > (dst, src, numels, static_cast<Dtype>(1.0f / numels));
	}
	else
	{
		gpu_mean_vectorized_kernel << <num_blocks, threads_per_block, shared_mem_size >> > (dst, src, numels, static_cast<Dtype>(1.0f / numels));
	}

	
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
__global__ void gpu_mean_any_axes_kernel(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t partial_offset_dst;
	float val;
	int i;
	int j;


	offset_dst = (blockIdx.x * blockDim.x + threadIdx.x);
	partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);

	val = 0;

	for (i = 0; i < len; i++)
	{
		offset_src = offs_calc.GetPartialSrcOffset_s(i);
		val += src[offset_src + partial_offset_dst];
	}

	dst[offset_dst] = val * scale;
}


template<typename Dtype>
__global__ void gpu_mean_last_axis_only_kernel(Dtype* dst, const Dtype* src, const uint64_t numels_dst, const uint32_t stride, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
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

// Note: stride must be a multiple of 4 so that vectorization works
// warps read sequential bytes then sum 'verically'
// row 0 -> read by warp 0 
// row 1 -> read by warp 1 
// row 2 -> read by warp 2 
// row 3 -> read by warp 3 
// then waps 0 - 3 are reduced (sum) into one row
// i.e. interleave = 4 (unrelated to vec_size)
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
		shared_memory[warpId * blockDim.x + laneIdx * 4 + i] = ((float*)&sum)[i] * scale; // this line causes bank conflicts however they disappear if blockDim.x is replaced with actual value of 128 (compiler optimization?)
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

			threads_per_block = GetNextPowerOf2(len);
			threads_per_block = min(threads_per_block, 128);
			threads_per_block = max(threads_per_block, CUDA_WARP_SIZE);

			num_warps = threads_per_block / CUDA_WARP_SIZE;
			num_blocks = static_cast<int>(numels_dst) / (num_warps * 4 * 8);
			num_blocks = max(num_blocks, 1);

			num_blocks = static_cast<int>(numels_dst) / (threads_per_block);
			gpu_mean_any_axes_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, offs_calc, len, scale);
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

		gpu_mean_last_axis_only_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, numels_dst, stride, scale);

	}
}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// general var functions (i.e. var with axes)
//
//-----------------------------------------------------------------------------------------------------
// uses Welford's online algorithm (see:https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm)
template<typename Dtype>
__global__ void gpu_var_last_axis_only_kernel(Dtype* dst, const Dtype* src, const uint64_t numels, const uint32_t dim_len, float scale)
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
	int warp_iterations;

	int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	int laneIdx = threadIdx.x % warpSize;

	offset_dst = warpIdx;

	offset_src = warpIdx * dim_len;

	Ms = 0;
	if (laneIdx < dim_len)
	{
		count = 1.0f;
		mean = src[offset_src + laneIdx];
	}
	else
	{
		count = 0;
		mean = 0;
	}

	offset_src += laneIdx + warpSize; // advance to next element

	uint32_t end = warpIdx * dim_len + dim_len;

	for (i = offset_src; i < end; i += warpSize)
	{
		count_b = 1;
		mean_b = src[i];
		Ms_b = 0;

		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		
		delta = mean_b - mean;
		//mean += delta * (count_b * temp);
		mean = ((count * mean) + (count_b * mean_b)) / n_ab;


		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}
	
	//
	// reduce
	//
	//------------------------------------------------------
	count_b = __shfl_down_sync(FULL_MASK, count, 16);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 16);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 16);
	if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		//mean += delta * (count_b * temp);
		mean = ((count * mean) + (count_b * mean_b)) / n_ab;
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}

	count_b = __shfl_down_sync(FULL_MASK, count, 8);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 8);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 8);
	if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		//mean += delta * (count_b * temp);
		mean = ((count * mean) + (count_b * mean_b)) / n_ab;
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}


	count_b = __shfl_down_sync(FULL_MASK, count, 4);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 4);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 4);
	if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		//mean += delta * (count_b * temp);
		mean = ((count * mean) + (count_b * mean_b)) / n_ab;
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}


	count_b = __shfl_down_sync(FULL_MASK, count, 2);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 2);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 2);
	if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		//mean += delta * (count_b * temp);
		mean = ((count * mean) + (count_b * mean_b)) / n_ab;
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}


	count_b = __shfl_down_sync(FULL_MASK, count, 1);
	mean_b = __shfl_down_sync(FULL_MASK, mean, 1);
	Ms_b = __shfl_down_sync(FULL_MASK, Ms, 1);
	if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		//mean += delta * (count_b * temp);
		mean = ((count * mean) + (count_b * mean_b)) / n_ab;
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
	int warps_required;
	int warps_per_block;

	naxes = ndims_src - ndims_dst;


	if (naxes > 1 || (axes[0] != ndims_src - 1))
	{
		//OffsetCalc_mean_std_simple offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);
		LTEN_ERR("Not yet implemented: gpu_var for naxes > 1 or axis != last axis"); // need to copy and clean up implementation from merge3
	}
	else
	{
		len = 1;
		for (i = 0; i < naxes; i++)
		{
			len *= dims_src[axes[i]];
		}


		warps_required = numels; // one warp for each output (note: numels is dst numels)
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		threads_per_block = warps_per_block * CUDA_WARP_SIZE;
		num_blocks = (warps_required + warps_per_block - 1)/ warps_per_block;

		scale = 1.0f / (len - 1);

		gpu_var_last_axis_only_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, numels, len, scale);

	}

}


//-----------------------------------------------------------------------------------------------------
//
// layer norm functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_layer_norm_last_axis_only_kernel(Dtype* dst, const Dtype* src, const uint64_t numels, const uint32_t dim_len, float scale, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd)
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
	float std;
	int i;
	float epsilon = 0;

	int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	int laneIdx = threadIdx.x % warpSize;

	offset_src = warpIdx * dim_len;

	Ms = 0;
	if (laneIdx < dim_len)
	{
		count = 1.0f;
		mean = src[offset_src + laneIdx];
	}
	else
	{
		count = 0;
		mean = 0;
	}

	offset_src += laneIdx + warpSize; // advance to next element

	uint32_t end = warpIdx * dim_len + dim_len;

	for (i = offset_src; i < end; i += warpSize)
	{
		count_b = 1;
		mean_b = src[i];
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
	if (count_b)
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
	if (count_b)
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
	if (count_b)
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
	if (count_b)
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
	if (count_b)
	{
		n_ab = count + count_b;
		temp = 1.0f / n_ab;
		delta = mean_b - mean;
		mean += delta * (count_b * temp);
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}
	//------------------------------------------------------


	mean = __shfl_sync(FULL_MASK, mean, 0);
	Ms = __shfl_sync(FULL_MASK, Ms, 0);

	std = sqrt(Ms * scale + epsilon);

	if (laneIdx == 0)
	{
		sd[warpIdx] = std;
	}


	offset_src = warpIdx * dim_len;

	std = 1.0f / std; // invert so that multiplication can be used 

	for (i = laneIdx; i < dim_len; i += warpSize)
	{
		temp = src[offset_src + i];
		temp = (temp - mean) * std;

		if (weight)
		{
			ln[offset_src + i] = temp;

			temp = temp * weight[i] + bias[i];
		}

		dst[offset_src + i] = temp; // offset_src == offset_dst
	}

}

template<typename Dtype>
void gpu_layer_norm(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd)
{
	uint64_t len;
	int naxes;
	int i;
	int num_blocks;
	uint32_t threads_per_block;
	float scale;
	int warps_required;
	int warps_per_block;

	naxes = ndims_src - ndims_dst;

	if (naxes > 1 || (axes[0] != ndims_src - 1))
	{
		//OffsetCalc_mean_std_simple offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);
		LTEN_ERR("Not yet implemented: gpu_layer_norm for naxes > 1 or axis != last axis");
	}
	else
	{
		len = 1;
		for (i = 0; i < naxes; i++)
		{
			len *= dims_src[axes[i]];
		}

		warps_required = numels / len; // one warp for each traversal along the axis
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		threads_per_block = warps_per_block * CUDA_WARP_SIZE;
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		scale = 1.0f / len; // use biased estimator


		gpu_layer_norm_last_axis_only_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, numels, len, scale, weight, bias, ln, sd);
	}
}


//-----------------------------------------------------------------------------------------------------
//
// transpose functions
//
//-----------------------------------------------------------------------------------------------------
const int defa_threads = 64;
constexpr int num_threads = 64;
constexpr int thread_work_size = vec_size;
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

	index = threadIdx.x + block_work_size * blockIdx.x;

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
// repeat functions
//
//-----------------------------------------------------------------------------------------------------
__global__ void gpu_repeat_vectorized_kernel(float* dst, float* src, int N, OffsetCalc_repeat offs_calc)
{
	float4 data;
	unsigned int dst_offset;
	unsigned int src_offset;
	int i;

	dst_offset = (blockIdx.x * blockDim.x + threadIdx.x) * thread_work_size;


	if (dst_offset + vec_size <= N)
	{
#pragma unroll
		for (i = 0; i < vec_size; i++)
		{
			src_offset = offs_calc.GetOffsets(dst_offset + i);
			((float*)&data)[i] = src[src_offset];
		}
		*(reinterpret_cast<float4*>(&dst[dst_offset])) = data;
	}
	else
	{
#pragma unroll		
		for (i = 0; i < vec_size - 1; i++)
		{
			if (dst_offset < N)
			{
				src_offset = offs_calc.GetOffsets(dst_offset);
				dst[dst_offset] = src[src_offset];
			}
			dst_offset++;
		}
	}
}


template<typename Dtype>
void gpu_repeat(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims)
{
	uint64_t N;
	int num_blocks;
	uint32_t factor;

	OffsetCalc_repeat offs_calc(strides_dst, strides_src, dims_src, ndims);

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	factor = defa_threads * vec_size;

	num_blocks = (static_cast<int>(N) + factor - 1) / factor; // allocate (1/vec_size) threads since vectorization kernel to be used

	gpu_repeat_vectorized_kernel << < num_blocks, defa_threads >> > ((float*)dst, (float*)src, N, offs_calc);

}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// repeat interleave functions
//
//-----------------------------------------------------------------------------------------------------
__global__ void gpu_repeat_interleave_kernel(float* dst, float* src, int N, OffsetCalc_repeat_interleave offs_calc)
{
	unsigned int dst_offset;
	unsigned int src_offset;

	dst_offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (dst_offset < N)
	{
		src_offset = offs_calc.GetOffsets(dst_offset);

		dst[dst_offset] = src[src_offset];
	}
}

template<typename Dtype>
void gpu_repeat_interleave(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim)
{
	uint64_t N;
	int num_blocks;

	OffsetCalc_repeat_interleave offs_calc(strides_dst, strides_array, cummulative_times, ndims, ndims_times, dim);

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_repeat_interleave_kernel << < num_blocks, defa_threads >> > ((float*)dst, (float*)src, N, offs_calc);

}
//-----------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------
//
// index functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_index_kernel(Dtype* dst, const Dtype* src, const int* indices, uint64_t copy_len, const uint64_t N)
{
	unsigned int offset;
	unsigned int index;

	offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset < N)
	{
		index = indices[offset / copy_len];

		dst[offset] = src[index * copy_len + offset % copy_len];
	}
}

template<typename Dtype>
void gpu_index(Dtype* dst, const Dtype* src, const int* indices, uint64_t copy_len, const uint64_t numels)
{
	uint64_t N;
	int num_blocks;

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_index_kernel << < num_blocks, defa_threads >> > (dst, src, indices, copy_len, N);
}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// permutation functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_permute_kernel(Dtype* dst, const Dtype* src, const uint64_t N, OffsetCalc_permutaion ofs)
{
	uint32_t thread_id;
	uint32_t i;
	uint32_t grid_stride;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	grid_stride = blockDim.x * gridDim.x;

	for (i = thread_id; i < N; i += grid_stride)
	{
		uint32_t offset;
		offset = ofs.GetOffset(i);
		dst[offset] = src[i];
	}
}

template<typename Dtype>
void gpu_permute(Dtype* dst, const Dtype* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutations)
{
	OffsetCalc_permutaion ofs(strides_dst, strides_src, permutations, ndims);


	gpu_permute_kernel << < 5, 128 >> > (dst, src, numels, ofs );
}


template<typename Dtype>
__global__ void set_addresses_kernel(Dtype* base_addr_a, Dtype* base_addr_b, Dtype* base_addr_c, Dtype** addresses_a, Dtype** addresses_b, Dtype** addresses_c, const uint32_t* offsets_a, const uint32_t* offsets_b, const uint32_t* offsets_c, const uint64_t num_addresses)
{
	uint32_t thread_id;
	uint32_t i;
	uint32_t grid_stride;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	grid_stride = blockDim.x * gridDim.x;

	for (i = thread_id; i < num_addresses; i += grid_stride)
	{
		addresses_a[i] = base_addr_a + offsets_a[i];
		addresses_b[i] = base_addr_b + offsets_b[i];
		addresses_c[i] = base_addr_c + offsets_c[i];
	}
}

template<typename Dtype>
void set_addresses(Dtype* A, Dtype* B, Dtype* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses)
{
	int num_blocks;
	int num_threads = 128;

	num_blocks = max((int)1, (int)(num_addresses / num_threads));

	set_addresses_kernel << < num_blocks, num_threads >> > (A, B, C, (Dtype**)addresses->a_array, (Dtype**)addresses->b_array, (Dtype**)addresses->c_array, offsets->a_array, offsets->b_array, offsets->c_array, num_addresses);
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

template void gpu_layer_norm<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, float* weights, float* bias, float* ln, float* sd);
template void gpu_layer_norm<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, int* weights, int* bias, int* ln, int* sd);
template void gpu_layer_norm<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, uint8_t* weights, uint8_t* bias, uint8_t* ln, uint8_t* sd);

template void gpu_transpose<float>(const float* A, float* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
template void gpu_transpose<int>(const int* A, int* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
template void gpu_transpose<uint8_t>(const uint8_t* A, uint8_t* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);

template void gpu_repeat<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);
template void gpu_repeat<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);
template void gpu_repeat<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);

template void gpu_repeat_interleave<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);
template void gpu_repeat_interleave<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);
template void gpu_repeat_interleave<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);

template void gpu_index<float>(float* dst, const float* src, const int* indices, uint64_t copy_len, const uint64_t numels);
template void gpu_index<int>(int* dst, const int* src, const int* indices, uint64_t copy_len, const uint64_t numels);
template void gpu_index<uint8_t>(uint8_t* dst, const uint8_t* src, const int* indices, uint64_t copy_len, const uint64_t numels);

template void gpu_permute<float>(float* dst, const float* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions);
template void gpu_permute<int>(int* dst, const int* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions);
template void gpu_permute<uint8_t>(uint8_t* dst, const uint8_t* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions);

template void set_addresses<float>(float* A, float* B, float* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);
template void set_addresses<int>(int* A, int* B, int* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);
template void set_addresses<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);
