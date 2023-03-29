#include <assert.h>
#include <cstdint>
#include <tuple>
#include <stdio.h>
#include <algorithm>
#include "tensorimpl.h"
#include "lten.h"
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

//***********************************************************************************************************************************************
const int defa_threads = 64;
constexpr int num_threads = 64;
constexpr int thread_work_size = 4;
constexpr int block_work_size = 256;


template<typename args_t>
__device__ inline void unrolled_load(args_t *args, float* A, float* B, int idx, int remaining, OffsetCalc_broadcast* input_offset_calculator)
{
	int thread_idx = threadIdx.x;
	uint32_t offset[2];

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (thread_idx >= remaining)
		{
			return;
		}
		int linear_idx = thread_idx + block_work_size * idx;

		//input_offset_calculator->GetOffsets(linear_idx, &offset[0], &offset[1]);
		input_offset_calculator->GetOffsets(linear_idx, offset);

		std::get<0>(args[i]) = A[offset[0]];
		std::get<1>(args[i]) = B[offset[1]];

		thread_idx += num_threads;
	}
}


__device__ inline void unrolled_save(float* C, float* results, int idx, int remaining)
{
	int thread_idx = threadIdx.x;

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (thread_idx >= remaining)
		{
			return;
		}
		int linear_idx = thread_idx + block_work_size * idx;

		C[linear_idx] = results[i];

		thread_idx += num_threads;
	}

}

__device__ inline bool bounds_check(int index, int remaining)
{
	return ((threadIdx.x + index * num_threads) < remaining);
}

//***********************************************************************************************************************************************

//-----------------------------------------------------------------------------------------------------
//
// mul functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_mul_kernel(uint32_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = A[i] * B[i];
	}
}

template<typename Dtype>
void gpu_mul(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_mul_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, A, B, C);
}

__global__ void gpu_mul_kernel(float* A, float* B, float* C, int N, int ndims, OffsetCalc_broadcast offs_calc)
{
	int remaining = N - block_work_size * blockIdx.x;
	int idx = blockIdx.x;

	std::tuple<float, float> args[thread_work_size];
	float results[thread_work_size];

	unrolled_load(args, A, B, idx, remaining, &offs_calc);

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (bounds_check(i, remaining))
		{
			results[i] = std::get<0>(args[i]) * std::get<1>(args[i]);
		}
	}

	unrolled_save(C, results, idx, remaining);
}

template<typename Dtype>
void gpu_mul(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims)
{
	OffsetCalc_broadcast off_calc(ndims, a_dims, b_dims, c_dims, a_strides, b_strides, c_strides);

	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;
	uint64_t N;

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + 256 - 1) / 256;

	gpu_mul_kernel << < num_blocks, defa_threads >> > ((float*)A, (float*)B, (float*)C, (int)N, ndims, off_calc);
}


template<typename Dtype>
__global__ void gpu_mul_backward_kernel(uint32_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = A[i] * B[i];
	}
}


template<typename Dtype>
void gpu_mul_backward(uint64_t N, Dtype* operand, Dtype* top_gradient, Dtype* bottom_gradient)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_mul_backward_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, operand, top_gradient, bottom_gradient);
}



template<typename Dtype>
__global__ void gpu_mul_backward_kernel(uint32_t numoutputs, Dtype* top_gradient, Dtype* bottom_gradient, Dtype* other_operand, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count)
{
	// one warp for each output
	uint32_t warpId;
	uint32_t laneId;
	uint32_t i;
	uint32_t other_operand_offset;
	uint32_t tg_offset;
	Dtype val;

	warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpId >= numoutputs)
	{
		return;
	}

	laneId = threadIdx.x % warpSize;

	val = 0;
	for (i = laneId; i < loop_count; i+= warpSize)
	{
		offs_rev.GetOffsets(warpId, i, &other_operand_offset, &tg_offset);
		val += other_operand[other_operand_offset] * top_gradient[tg_offset];
	}

	//
	// reduce
	//
	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		bottom_gradient[warpId] = val;
	}
	
}

template<typename Dtype>
void gpu_mul_backward(Dtype* top_gradient, Dtype* bottom_gradient, Dtype* other_operand, const int ndims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides)
{
	// gpu_mul_backward w.r.t. operand 1
	uint64_t numels_tg;
	uint64_t numoutputs;
	int i;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	uint32_t loop_count;
	OffsetCalc_reverse_broadcast offs_rev(ndims, op1_dims, op2_dims, tg_dims, op1_strides, op2_strides, tg_strides);

	numels_tg = numoutputs = 1;
	for (i = 0; i < ndims; i++)
	{
		numels_tg *= tg_dims[i];
		numoutputs *= op1_dims[i];
	}
	
	loop_count = numels_tg / numoutputs;
	warps_required = static_cast<uint32_t>(numoutputs); // one warp for each output
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	gpu_mul_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (numoutputs, top_gradient, bottom_gradient, other_operand, offs_rev, loop_count);
}
//-----------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------
//
// div functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_div_kernel(uint32_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = A[i] / B[i];
	}
}

template<typename Dtype>
void gpu_div(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_div_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, A, B, C);
}

__global__ void gpu_div_kernel(float* A, float* B, float* C, int N, int ndims, OffsetCalc_broadcast offs_calc)
{
	int remaining = N - block_work_size * blockIdx.x;
	int idx = blockIdx.x;

	std::tuple<float, float> args[thread_work_size];
	float results[thread_work_size];

	unrolled_load(args, A, B, idx, remaining, &offs_calc);

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (bounds_check(i, remaining))
		{
			results[i] = std::get<0>(args[i]) / std::get<1>(args[i]);
		}
	}

	unrolled_save(C, results, idx, remaining);
}

template<typename Dtype>
void gpu_div(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims)
{
	OffsetCalc_broadcast off_calc(ndims, a_dims, b_dims, c_dims, a_strides, b_strides, c_strides);

	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;
	uint64_t N;

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + 256 - 1) / 256;

	gpu_div_kernel << < num_blocks, defa_threads >> > ((float*)A, (float*)B, (float*)C, (int)N, ndims, off_calc);
}

template<typename Dtype>
__global__ void gpu_div_backward_kernel(uint32_t N, Dtype* operand2, Dtype* top_gradient, Dtype* bottom_gradient)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		bottom_gradient[i] = top_gradient[i] / operand2[i];
	}
}

template<typename Dtype>
__global__ void gpu_div_backward_kernel(uint32_t N, Dtype* operand1, Dtype* operand2, Dtype* top_gradient, Dtype* bottom_gradient)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		bottom_gradient[i] = top_gradient[i] * (-operand1[i]) / (operand2[i] * operand2[i]);
	}
}


template<typename Dtype>
void gpu_div_backward(uint64_t N, Dtype* operand1, Dtype* operand2, Dtype* top_gradient, Dtype* bottom_gradient, bool divisor)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	if (divisor)
	{
		gpu_div_backward_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, operand1, operand2, top_gradient, bottom_gradient);
	}
	else
	{
		gpu_div_backward_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, operand2, top_gradient, bottom_gradient);
	}	
}




template<typename Dtype>
__global__ void gpu_div_backward_kernel(uint32_t numoutputs, Dtype* top_gradient, Dtype* bottom_gradient, Dtype* other_operand, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count)
{
	// one warp for each output
	uint32_t warpId;
	uint32_t laneId;
	uint32_t i;
	uint32_t other_operand_offset;
	uint32_t tg_offset;
	Dtype val;

	warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpId >= numoutputs)
	{
		return;
	}
	laneId = threadIdx.x % warpSize;

	val = 0;
	for (i = laneId; i < loop_count; i += warpSize)
	{
		offs_rev.GetOffsets(warpId, i, &other_operand_offset, &tg_offset);
		val += top_gradient[tg_offset] / other_operand[other_operand_offset];
	}

	//
	// reduce
	//
	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		bottom_gradient[warpId] = val;
	}

}

template<typename Dtype>
__global__ void gpu_div_backward_kernel(uint32_t numoutputs, Dtype* top_gradient, Dtype* bottom_gradient, Dtype* operand1, Dtype* operand2, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count)
{
	// one warp for each output
	uint32_t warpId;
	uint32_t laneId;
	uint32_t i;
	uint32_t operand2_offset;
	uint32_t tg_offset;
	Dtype val;

	warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpId >= numoutputs)
	{
		return;
	}
	laneId = threadIdx.x % warpSize;

	val = 0;
	for (i = laneId; i < loop_count; i += warpSize)
	{
		offs_rev.GetOffsets(warpId, i, &operand2_offset, &tg_offset);
		val += top_gradient[tg_offset] * (-operand2[operand2_offset]) / (operand1[warpId] * operand1[warpId]);
	}

	//
	// reduce
	//
	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		bottom_gradient[warpId] = val;
	}

}

template<typename Dtype>
void gpu_div_backward(Dtype* top_gradient, Dtype* bottom_gradient, Dtype* operand1, Dtype* operand2, const int ndims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, bool divisor)
{
	// gpu_div_backward w.r.t. operand 1
	uint64_t numels_tg;
	uint64_t numoutputs;
	int i;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	uint32_t loop_count;
	OffsetCalc_reverse_broadcast offs_rev(ndims, op1_dims, op2_dims, tg_dims, op1_strides, op2_strides, tg_strides);

	numels_tg = numoutputs = 1;
	for (i = 0; i < ndims; i++)
	{
		numels_tg *= tg_dims[i];
		numoutputs *= op1_dims[i];
	}

	loop_count = numels_tg / numoutputs;
	warps_required = static_cast<uint32_t>(numoutputs); // one warp for each output
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	if (divisor)
	{
		gpu_div_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (numoutputs, top_gradient, bottom_gradient, operand1, operand2, offs_rev, loop_count);
	}
	else
	{
		gpu_div_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (numoutputs, top_gradient, bottom_gradient, operand2, offs_rev, loop_count);
	}
	
}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// add functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_add_kernel(uint32_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = A[i] + B[i];
	}
}

template<typename Dtype>
void gpu_add(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_add_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, A, B, C);
}

template<typename Dtype>
__global__ void gpu_add_backward_kernel(uint32_t N, Dtype* top_gradient, Dtype* bottom_gradient)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		bottom_gradient[i] = top_gradient[i];
	}
}


template<typename Dtype>
void gpu_add_backward(uint64_t N, Dtype* top_gradient, Dtype* bottom_gradient)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_add_backward_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, top_gradient, bottom_gradient);
}


template<typename Dtype>
__global__ void gpu_add_backward_kernel(uint32_t num_outputs, Dtype* top_gradient, Dtype* bottom_gradient, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count)
{
	uint32_t warpIdx;
	uint32_t laneId;
	uint32_t i;
	uint32_t j;
	uint32_t tg_offset;
	Dtype val;
	uint32_t p_offset;

	warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpIdx >= num_outputs)
	{
		return;
	}

	laneId = threadIdx.x % warpSize;

	offs_rev.GetParialOffsets(warpIdx, nullptr, &p_offset);
	val = 0;
	i = laneId;

	while (i < loop_count)
	{
#pragma unroll
		for (j = 0; j < 4; j++)
		{
			if (i >= loop_count) break;

			tg_offset = p_offset + offs_rev.GetBroadcastOffset(i);
			val += top_gradient[tg_offset];
			i += warpSize;
		}
	}

	//
	// reduce
	//
	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		bottom_gradient[warpIdx] = val;
	}

}


// one thread per output but with 'clone' blocks to reduce the work load
// good for large to small reductions, where thread count (i.e. number of outputs) is low and therefore work per thread is high
template<typename Dtype>
__global__ void gpu_add_backward_block_expanded_kernel(uint32_t num_outputs, Dtype* top_gradient, Dtype* bottom_gradient, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count, uint32_t remaining)
{
	uint32_t threadId;
	uint32_t i;
	uint32_t j;
	uint32_t index;
	Dtype val[4];
	uint32_t tg_offset[4];
	uint32_t start;
	uint32_t stop;
	uint32_t p_offset;


	threadId = ((blockIdx.x / 64) * blockDim.x) + threadIdx.x;
	if (threadId >= num_outputs)
	{
		return;
	}

	start = loop_count * ((blockIdx.x) % 64);
	stop = start + loop_count;

	p_offset = offs_rev.GetParialOffsets(threadId, nullptr);

	val[0] = 0;
	val[1] = 0;
	val[2] = 0;
	val[3] = 0;

	if (loop_count % 4)
	{
		for (i = start; i < stop; i += 4)
		{
#pragma unroll
			for (j = 0; j < 4; j++)
			{
				if (i + j >= stop) break;
				tg_offset[j] = p_offset + offs_rev.GetBroadcastOffset(i + j);
			}

#pragma unroll
			for (j = 0; j < 4; j++)
			{
				if (i + j >= stop) break;
				val[j] += top_gradient[tg_offset[j]];
			}
		}

	}
	else
	{
		// no checks to break out of loops
		for (i = start; i < stop; i += 4)
		{
#pragma unroll
			for (j = 0; j < 4; j++)
			{
				tg_offset[j] = p_offset + offs_rev.GetBroadcastOffset(i + j);
			}

#pragma unroll
			for (j = 0; j < 4; j++)
			{
				val[j] += top_gradient[tg_offset[j]];
			}
		}

	}


	if (remaining)
	{
		start = loop_count * 64;
		stop = start + remaining;
		if ((blockIdx.x % 64) == 0)
		{
			i = start;
			while (i < stop)
			{
#pragma unroll
				for (j = 0; j < 4; j++)
				{
					if (i + j >= stop) break;
					index = p_offset + offs_rev.GetBroadcastOffset(i + j);
					val[0] += top_gradient[index];
				}
				i += 4;
			}

		}
	}

	Dtype valx = val[0] + val[1] + val[2] + val[3];

	atomicAdd((float*)&bottom_gradient[threadId], valx);

}

template<typename Dtype>
void gpu_add_backward(Dtype* top_gradient, Dtype* bottom_gradient, const int ndims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides)
{
	// gpu_add_backward w.r.t. operand 1
	uint64_t numels_tg;
	uint64_t num_outputs;
	int i;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	uint32_t loop_count;
	OffsetCalc_reverse_broadcast offs_rev(ndims, op1_dims, op2_dims, tg_dims, op1_strides, op2_strides, tg_strides);

	numels_tg = num_outputs = 1;
	for (i = 0; i < ndims; i++)
	{
		numels_tg *= tg_dims[i];
		num_outputs *= op1_dims[i];
	}

	loop_count = numels_tg / num_outputs;

	if (loop_count < 4096)
	{
		warps_required = static_cast<uint32_t>(num_outputs); // one warp for each output
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		gpu_add_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (num_outputs, top_gradient, bottom_gradient, offs_rev, loop_count);
	}
	else
	{
		warps_required = static_cast<uint32_t>(num_outputs + 32 - 1) / 32; // one thread for each output
		warps_per_block = min(4, warps_required);
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		cudaMemsetAsync(bottom_gradient, 0, sizeof(Dtype) * num_outputs);
		gpu_add_backward_block_expanded_kernel << <num_blocks * 64, warps_per_block * CUDA_WARP_SIZE >> > (num_outputs, top_gradient, bottom_gradient, offs_rev, loop_count / 64, loop_count % 64);
	}
}



__global__ void gpu_add_kernel(float* A, float* B, float* C, int N, int ndims, OffsetCalc_broadcast offs_calc)
{
	int remaining = N - block_work_size * blockIdx.x;
	int idx = blockIdx.x;

	std::tuple<float, float> args[thread_work_size];
	float results[thread_work_size];

	unrolled_load(args, A, B, idx, remaining, &offs_calc);

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (bounds_check(i, remaining))
		{
			results[i] = std::get<0>(args[i]) + std::get<1>(args[i]);
		}
	}

	unrolled_save(C, results, idx, remaining);
}

template<typename Dtype>
void gpu_add(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims)
{
	OffsetCalc_broadcast off_calc(ndims, a_dims, b_dims, c_dims, a_strides, b_strides, c_strides);

	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;
	uint64_t N;

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + 256 - 1) / 256;

	gpu_add_kernel << < num_blocks, defa_threads >> > ((float*)A, (float*)B, (float*)C, (int)N, ndims, off_calc);
}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// sub functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_sub_kernel(uint32_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = A[i] - B[i];
	}
}

template<typename Dtype>
void gpu_sub(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_sub_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, A, B, C);
}


template<typename Dtype>
__global__ void gpu_sub_backward_kernel(uint32_t N, Dtype* top_gradient, Dtype* bottom_gradient, Dtype scale)
{
	uint32_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		bottom_gradient[i] = scale * top_gradient[i];
	}
}


template<typename Dtype>
void gpu_sub_backward(uint64_t N, Dtype* top_gradient, Dtype* bottom_gradient, Dtype scale)
{
	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;

	int defa_threads = 64;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + defa_threads - 1) / defa_threads;

	gpu_sub_backward_kernel<Dtype> << <num_blocks, defa_threads >> > ((uint32_t)N, top_gradient, bottom_gradient, scale);
}


template<typename Dtype>
__global__ void gpu_sub_backward_kernel(uint32_t num_outputs, Dtype* top_gradient, Dtype* bottom_gradient, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count, Dtype scale)
{
	uint32_t warpIdx;
	uint32_t laneId;
	uint32_t i;
	uint32_t j;
	uint32_t tg_offset;
	Dtype val;
	uint32_t p_offset;

	warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpIdx >= num_outputs)
	{
		return;
	}

	laneId = threadIdx.x % warpSize;

	offs_rev.GetParialOffsets(warpIdx, nullptr, &p_offset);
	val = 0;
	i = laneId;

	while (i < loop_count)
	{
#pragma unroll
		for (j = 0; j < 4; j++)
		{
			if (i >= loop_count) break;

			tg_offset = p_offset + offs_rev.GetBroadcastOffset(i);
			val += top_gradient[tg_offset];
			i += warpSize;
		}
	}

	//
	// reduce
	//
	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		bottom_gradient[warpIdx] = val * scale;
	}

}


// one thread per output but with 'clone' blocks to reduce the work load
// good for large to small reductions, where thread count (i.e. number of outputs) is low and therefore work per thread is high
template<typename Dtype>
__global__ void gpu_sub_backward_block_expanded_kernel(uint32_t num_outputs, Dtype* top_gradient, Dtype* bottom_gradient, OffsetCalc_reverse_broadcast offs_rev, uint32_t loop_count, uint32_t remaining, Dtype scale)
{
	uint32_t threadId;
	uint32_t i;
	uint32_t j;
	uint32_t index;
	Dtype val[4];
	uint32_t tg_offset[4];
	uint32_t start;
	uint32_t stop;
	uint32_t p_offset;


	threadId = ((blockIdx.x / 64) * blockDim.x) + threadIdx.x;
	if (threadId >= num_outputs)
	{
		return;
	}

	start = loop_count * ((blockIdx.x) % 64);
	stop = start + loop_count;

	p_offset = offs_rev.GetParialOffsets(threadId, nullptr);

	val[0] = 0;
	val[1] = 0;
	val[2] = 0;
	val[3] = 0;

	if (loop_count % 4)
	{
		for (i = start; i < stop; i += 4)
		{
#pragma unroll
			for (j = 0; j < 4; j++)
			{
				if (i + j >= stop) break;
				tg_offset[j] = p_offset + offs_rev.GetBroadcastOffset(i + j);
			}

#pragma unroll
			for (j = 0; j < 4; j++)
			{
				if (i + j >= stop) break;
				val[j] += top_gradient[tg_offset[j]];
			}
		}

	}
	else
	{
		// no checks to break out of loops
		for (i = start; i < stop; i += 4)
		{
#pragma unroll
			for (j = 0; j < 4; j++)
			{
				tg_offset[j] = p_offset + offs_rev.GetBroadcastOffset(i + j);
			}

#pragma unroll
			for (j = 0; j < 4; j++)
			{
				val[j] += top_gradient[tg_offset[j]];
			}
		}

	}


	if (remaining)
	{
		start = loop_count * 64;
		stop = start + remaining;
		if ((blockIdx.x % 64) == 0)
		{
			i = start;
			while (i < stop)
			{
#pragma unroll
				for (j = 0; j < 4; j++)
				{
					if (i + j >= stop) break;
					index = p_offset + offs_rev.GetBroadcastOffset(i + j);
					val[0] += top_gradient[index];
				}
				i += 4;
			}
		}
	}

	Dtype valx = scale * (val[0] + val[1] + val[2] + val[3]);

	atomicAdd((float*)&bottom_gradient[threadId], valx);

}

template<typename Dtype>
void gpu_sub_backward(Dtype* top_gradient, Dtype* bottom_gradient, const int ndims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, Dtype scale)
{
	// gpu_sub_backward w.r.t. operand 1
	uint64_t numels_tg;
	uint64_t num_outputs;
	int i;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	uint32_t loop_count;
	OffsetCalc_reverse_broadcast offs_rev(ndims, op1_dims, op2_dims, tg_dims, op1_strides, op2_strides, tg_strides);

	numels_tg = num_outputs = 1;
	for (i = 0; i < ndims; i++)
	{
		numels_tg *= tg_dims[i];
		num_outputs *= op1_dims[i];
	}

	loop_count = numels_tg / num_outputs;

	if (loop_count < 4096)
	{
		warps_required = static_cast<uint32_t>(num_outputs); // one warp for each output
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		gpu_sub_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (num_outputs, top_gradient, bottom_gradient, offs_rev, loop_count, scale);
	}
	else
	{
		warps_required = static_cast<uint32_t>(num_outputs + 32 - 1) / 32; // one thread for each output
		warps_per_block = min(4, warps_required);
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		cudaMemsetAsync(bottom_gradient, 0, sizeof(Dtype) * num_outputs);
		gpu_sub_backward_block_expanded_kernel << <num_blocks * 64, warps_per_block * CUDA_WARP_SIZE >> > (num_outputs, top_gradient, bottom_gradient, offs_rev, loop_count / 64, loop_count % 64, scale);
	}
}



__global__ void gpu_sub_kernel(float* A, float* B, float* C, int N, int ndims, OffsetCalc_broadcast offs_calc)
{
	int remaining = N - block_work_size * blockIdx.x;
	int idx = blockIdx.x;

	std::tuple<float, float> args[thread_work_size];
	float results[thread_work_size];

	unrolled_load(args, A, B, idx, remaining, &offs_calc);

#pragma unroll
	for (int i = 0; i < thread_work_size; i++)
	{
		if (bounds_check(i, remaining))
		{
			results[i] = std::get<0>(args[i]) - std::get<1>(args[i]);
		}
	}

	unrolled_save(C, results, idx, remaining);
}

template<typename Dtype>
void gpu_sub(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims)
{
	OffsetCalc_broadcast off_calc(ndims, a_dims, b_dims, c_dims, a_strides, b_strides, c_strides);

	dim3 dimGrid;
	dim3 dimBlock;
	int num_blocks;
	uint64_t N;

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + 256 - 1) / 256;

	gpu_sub_kernel << < num_blocks, defa_threads >> > ((float*)A, (float*)B, (float*)C, (int)N, ndims, off_calc);
}
//-----------------------------------------------------------------------------------------------------


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


template<typename Dtype>
__global__ void gpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numoutputs, OffsetCalc_broadcast offs, Dtype scale)
{
	uint32_t threadId;
	uint32_t index;

	threadId = blockIdx.x * blockDim.x + threadIdx.x; // absolute thread id
	if (threadId  >= numoutputs)
	{
		return;
	}

	index = offs.GetOffset(threadId);
	dst[threadId] = scale * src[index];

}

template<typename Dtype>
void gpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, Dtype scale)
{
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	uint32_t threads_required;


	threads_required = static_cast<uint32_t>(numoutputs); // one thread for each output
	warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;

	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	gpu_mean_backward<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (dst, src, numoutputs, *offs, scale);
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
__global__ void gpu_mean_any_axes_kernel(Dtype* dst, const Dtype* __restrict__ src, OffsetCalc_mean_var offs_calc, uint32_t len, uint32_t numels_dst, float scale)
{
	uint32_t offset_src;
	uint32_t offset_dst;
	uint32_t partial_offset_dst;
	float val;
	int i;

	offset_dst = (blockIdx.x * blockDim.x + threadIdx.x);
	if (offset_dst >= numels_dst)
	{
		return;
	}
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

	int warpIdx = threadIdx.x / warpSize;
	int laneIdx = threadIdx.x % warpSize;

	total_warps = blockDim.x / warpSize;

	offset_dst = blockIdx.x * total_warps + warpIdx;
	if (offset_dst >= numels_dst)
	{
		return;
	}

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
		uint32_t len;
		uint32_t stride;

		len = 1;
		for (i = 0; i < naxes; i++)
		{
			len *= static_cast<uint32_t>(dims_src[axes[i]]);
		}

		scale = 1.0f / len;

		if (naxes == 1 && !(strides_src[axes[0]] % vec_size))
		{
			stride = static_cast<uint32_t>(strides_src[axes[0]]);

			uint32_t batches = static_cast<uint32_t>(numels_dst) / stride;
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

			num_blocks = static_cast<int>(numels_dst + threads_per_block - 1) / (threads_per_block);
			gpu_mean_any_axes_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, offs_calc, len, numels_dst, scale);
		}
	}
	else
	{
		uint32_t stride;

		stride = static_cast<uint32_t>(dims_src[ndims_src - 1]);

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
__global__ void gpu_var_kernel(Dtype* dst, const Dtype* src, const uint64_t num_outputs, const uint32_t loop_count, OffsetCalc_mean_var offs_calc, float scale)
{
	uint32_t partial_offset;
	uint32_t src_offset;
	Dtype mean;
	Dtype count;
	Dtype mean_b;
	Dtype count_b;
	Dtype Ms_b;
	Dtype n_ab;
	Dtype temp;
	Dtype delta;
	Dtype Ms;
	uint32_t i;
	int threadId;

	threadId = blockIdx.x * blockDim.x + threadIdx.x; // absolute thread id 

	if (threadId >= num_outputs)
	{
		return;
	}

	partial_offset = offs_calc.GetPartialSrcOffset_d(threadId);
	src_offset = offs_calc.GetPartialSrcOffset_s(0);

	mean = src[partial_offset + src_offset];
	Ms = 0;
	count = static_cast<Dtype>(1);

	for (i = 1; i < loop_count; i++)
	{
		src_offset = offs_calc.GetPartialSrcOffset_s(i);

		count_b = 1;
		mean_b = src[partial_offset + src_offset];
		Ms_b = 0;

		n_ab = count + count_b;
		temp = 1.0f / n_ab;

		delta = mean_b - mean;
		mean = ((count * mean) + (count_b * mean_b)) * temp;
		Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
		count = n_ab;
	}

	dst[threadId] = Ms * scale;
}

template<typename Dtype>
__global__ void gpu_var_last_axis_only_kernel(Dtype* dst, const Dtype* src, const uint64_t num_outputs, const uint32_t dim_len, float scale)
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
	if (warpIdx >= num_outputs)
	{
		return;
	}
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
		mean = ((count * mean) + (count_b * mean_b)) * temp;


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
		mean = ((count * mean) + (count_b * mean_b)) * temp;
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
		mean = ((count * mean) + (count_b * mean_b)) * temp;
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
		mean = ((count * mean) + (count_b * mean_b)) * temp;
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
		mean = ((count * mean) + (count_b * mean_b)) * temp;
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
		mean = ((count * mean) + (count_b * mean_b)) * temp;
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
void gpu_var(Dtype* dst, const Dtype* src, const uint64_t num_outputs, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased)
{
	uint32_t loop_count;
	int naxes;
	int i;
	int num_blocks;
	uint32_t threads_per_block;
	float scale;
	uint32_t warps_required;
	uint32_t warps_per_block;

	naxes = ndims_src - ndims_dst;

	loop_count = 1;
	for (i = 0; i < naxes; i++)
	{
		loop_count *= static_cast<uint32_t>(dims_src[axes[i]]);
	}

	if (unbiased)
	{
		scale = 1.0f / (loop_count - 1);
	}
	else
	{
		scale = 1.0f / loop_count;
	}

	if (naxes > 1 || (axes[0] != ndims_src - 1))
	{
		OffsetCalc_mean_var offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);

		warps_required = (num_outputs + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE; // one thread per output
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		threads_per_block = warps_per_block * CUDA_WARP_SIZE;
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		gpu_var_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, num_outputs, loop_count, offs_calc, scale);
	}
	else
	{
		warps_required = static_cast<uint32_t>(num_outputs); // one warp for each output
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		threads_per_block = warps_per_block * CUDA_WARP_SIZE;
		num_blocks = (warps_required + warps_per_block - 1)/ warps_per_block;

		gpu_var_last_axis_only_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, num_outputs, loop_count, scale);
	}

}


//-----------------------------------------------------------------------------------------------------
//
// layer norm functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_layer_norm_last_axis_only_kernel(Dtype* dst, const Dtype* src, const uint32_t warps_required, const uint32_t dim_len, float scale, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd)
{
	uint32_t offset_src;
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
	float epsilon = 1e-5;

	int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpIdx >= warps_required)
	{
		return;
	}
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
		if (sd)
		{
			sd[warpIdx] = std; // save this for backprop
		}	
	}


	offset_src = warpIdx * dim_len;

	std = 1.0f / std; // invert so that multiplication can be used 

	for (i = laneIdx; i < dim_len; i += warpSize)
	{
		temp = src[offset_src + i];
		temp = (temp - mean) * std;

		if (ln)
		{
			ln[offset_src + i] = temp; // save this for backprop
		}

		if (weight)
		{
			temp = temp * weight[i] + bias[i];
		}

		dst[offset_src + i] = temp; // offset_src == offset_dst
	}

}

template<typename Dtype>
void gpu_layer_norm(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd)
{
	uint32_t len;
	int naxes;
	int i;
	int num_blocks;
	uint32_t threads_per_block;
	float scale;
	uint32_t warps_required;
	uint32_t warps_per_block;

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
			len *= static_cast<uint32_t>(dims_src[axes[i]]);
		}

		warps_required = static_cast<uint32_t>(numels) / len; // one warp for each traversal along the axis
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		threads_per_block = warps_per_block * CUDA_WARP_SIZE;
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		scale = 1.0f / len; // use biased estimator


		gpu_layer_norm_last_axis_only_kernel<Dtype> << <num_blocks, threads_per_block >> > (dst, src, warps_required, len, scale, weight, bias, ln, sd);
	}
}

template<typename Dtype>
__global__ void gpu_layer_norm_wt_bias_backward_kernel(Dtype* wt_grad, Dtype* bias_grad, const Dtype* top_gradient, Dtype* feeder_gradient, Dtype* ln, Dtype* sd, Dtype* wt, const uint64_t num_outputs, const uint32_t loop_count )
{
	int warpId;
	int laneId;	
	Dtype w_grd;
	Dtype b_grd;
	Dtype top_grd;
	Dtype wts;
	uint32_t i;
	uint32_t offset_src;

	warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpId >= num_outputs)
	{
		return;
	}

	laneId = threadIdx.x % warpSize;

	i = laneId;
	wts = wt[warpId];
	w_grd = 0;
	b_grd = 0;
	
	while (i < loop_count)
	{
		offset_src = i * num_outputs + warpId;
		top_grd = top_gradient[offset_src];
		w_grd += top_grd * ln[offset_src];
		b_grd += top_grd;

		feeder_gradient[offset_src] = top_grd * wts;

		i += warpSize;
	}

	//
	// reduce
	//
	w_grd += __shfl_down_sync(FULL_MASK, w_grd, 16);
	w_grd += __shfl_down_sync(FULL_MASK, w_grd, 8);
	w_grd += __shfl_down_sync(FULL_MASK, w_grd, 4);
	w_grd += __shfl_down_sync(FULL_MASK, w_grd, 2);
	w_grd += __shfl_down_sync(FULL_MASK, w_grd, 1);

	b_grd += __shfl_down_sync(FULL_MASK, b_grd, 16);
	b_grd += __shfl_down_sync(FULL_MASK, b_grd, 8);
	b_grd += __shfl_down_sync(FULL_MASK, b_grd, 4);
	b_grd += __shfl_down_sync(FULL_MASK, b_grd, 2);
	b_grd += __shfl_down_sync(FULL_MASK, b_grd, 1);

	if (laneId == 0)
	{
		wt_grad[warpId] = w_grd;
		bias_grad[warpId] = b_grd;
	}
}

template<typename Dtype>
__global__ void gpu_layer_norm_wt_bias_backward_kernel_xx(Dtype* wt_grad, Dtype* bias_grad, const Dtype* top_gradient, Dtype* feeder_gradient, Dtype* lnorm, Dtype* sd, Dtype* wt, const uint64_t numels, const uint32_t dim_len, int warps_per_block)
{
	uint32_t offset_src;
	uint32_t len;
	uint32_t i;
	float w_grd;
	float b_grd;
	float top_grd;
	float wts;

	__shared__ float s1[16 * 32]; // TODO make these dynamic
	__shared__ float s2[16 * 32];

	int warpIdx = threadIdx.x / warpSize; // * local warp id = local thread id /  warpSize
	int laneIdx = threadIdx.x % warpSize;
	

	offset_src = blockIdx.x * warpSize + warpIdx * dim_len + laneIdx; // consecutive rows for each warp
	wts = wt[blockIdx.x * warpSize + laneIdx];
	w_grd = 0;
	b_grd = 0;

	len = numels / dim_len / warps_per_block; // OPTOPT: move this division out to host code

	for (i = 0; i < len; i++)
	{
		if (offset_src < numels)
		{
			top_grd = top_gradient[offset_src];
			w_grd += top_grd * lnorm[offset_src];
			b_grd += top_grd;

			feeder_gradient[offset_src] = top_grd * wts;
		}
		offset_src += dim_len * warps_per_block;
	}

	s1[warpIdx * warpSize + laneIdx] = w_grd;
	s2[warpIdx * warpSize + laneIdx] = b_grd;
	__syncthreads();


	if (warpIdx > 0)
	{
		return;
	}

	w_grd = 0;
	b_grd = 0;

	for (i = 0; i < warps_per_block; i++)
	{
		w_grd += s1[i * warpSize + laneIdx];
		b_grd += s2[i * warpSize + laneIdx];
	}

	wt_grad[blockIdx.x * warpSize + laneIdx] = w_grd;
	bias_grad[blockIdx.x * warpSize + laneIdx] = b_grd;

}


template<typename Dtype>
__global__ void gpu_layer_norm_backward_kernel(Dtype* x_grad, Dtype* wt, Dtype* lnorm, Dtype* sd, Dtype* feeder_gradient, const uint64_t numels, const uint32_t ln_dim, float inv_dim, uint32_t loop_count)
{
	uint32_t offset;
	uint32_t i;
	uint32_t index;
	float m_g;
	float f_g;
	float tmp;
	float invsd;
	float l_n;

	__shared__ float fg[768]; // TODO make these dynamic
	__shared__ float ln[768];


	int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	int laneIdx = threadIdx.x % warpSize;

	offset = ln_dim * warpIdx;
	m_g = 0;
	tmp = 0;
	invsd = 1.0f / sd[blockIdx.x];

	index = offset + laneIdx;


#pragma unroll
	for (i = 0; i < loop_count; i++)
	{
		if ((index - offset) < ln_dim)
		{
			f_g = feeder_gradient[index];
			l_n = lnorm[index];
			m_g -= f_g * invsd;
			tmp -= f_g * l_n;

			fg[laneIdx + i * warpSize] = f_g;
			ln[laneIdx + i * warpSize] = l_n;

			index += warpSize;
		}
	}
	
	//
	// reduce
	//
	m_g += __shfl_down_sync(FULL_MASK, m_g, 16);
	tmp += __shfl_down_sync(FULL_MASK, tmp, 16);

	m_g += __shfl_down_sync(FULL_MASK, m_g, 8);
	tmp += __shfl_down_sync(FULL_MASK, tmp, 8);

	m_g += __shfl_down_sync(FULL_MASK, m_g, 4);
	tmp += __shfl_down_sync(FULL_MASK, tmp, 4);

	m_g += __shfl_down_sync(FULL_MASK, m_g, 2);
	tmp += __shfl_down_sync(FULL_MASK, tmp, 2);

	m_g += __shfl_down_sync(FULL_MASK, m_g, 1);
	tmp += __shfl_down_sync(FULL_MASK, tmp, 1);


	m_g = __shfl_sync(FULL_MASK, m_g, 0);
	tmp = __shfl_sync(FULL_MASK, tmp, 0);


	index = offset + laneIdx;
#pragma unroll
	for (i = 0; i < loop_count; i++)
	{
		if ((index - offset) < ln_dim)
		{
			//x_grad[index] = (feeder_gradient[index] + inv_dim * lnorm[index] * tmp) *  invsd + inv_dim * m_g;
			x_grad[index] = (fg[laneIdx + i * warpSize] + inv_dim * ln[laneIdx + i * warpSize] * tmp) *  invsd + inv_dim * m_g;
			//x_grad[index] = (fg[index%768] + inv_dim * ln[index % 768] * tmp) *  invsd + inv_dim * m_g;
		}
		index += warpSize;
	}

}


template<typename Dtype>
void gpu_layer_norm_backwards(void* vlayer_norm, Dtype* x, Dtype* top_gradient, Dtype* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, Dtype* feeder_gradient)
{
	lten::LayerNorm* layer_norm;
	Dtype* ln;
	Dtype* sd;
	Dtype* wt;
	Dtype* wt_grad;
	Dtype* bias_grad;
	uint32_t i;
	uint64_t numels;
	uint32_t warps_per_block;
	uint32_t warps_required;
	int num_blocks;
	uint64_t loop_count;
	uint32_t num_outputs;

	if (naxes > 1 || (axes[0] != ndims - 1))
	{
		LTEN_ERR("Not yet implemented: gpu_layer_norm_backwards for naxes > 1 or axis != last axis");
	}
	else
	{
		//if(dst_dims[ndims - 1] % CUDA_WARP_SIZE)
		//{
		//	LTEN_ERR("Only muliples of CUDA_WARP_SIZE for now");
		//}
		

		layer_norm = (lten::LayerNorm*)vlayer_norm;

		numels = 1;
		for (i = 0; i < ndims; i++)
		{
			numels *= dst_dims[i];
		}

		ln = layer_norm->get_ln()->get_mdarray<Dtype>()->GetDataPtr();
		sd = layer_norm->get_sd()->get_mdarray<Dtype>()->GetDataPtr();

		if (layer_norm->is_affine())
		{
			wt = layer_norm->get_weights()->get_mdarray<Dtype>()->GetDataPtr();
			wt_grad = layer_norm->get_weights()->get_gradients_mdarray<Dtype>()->GetDataPtr();
			bias_grad = layer_norm->get_bias()->get_gradients_mdarray<Dtype>()->GetDataPtr();

			num_outputs = dst_dims[ndims - 1];
			warps_required = static_cast<uint32_t>(num_outputs); // one warp for each output
			warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
			num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

			loop_count = numels / num_outputs;

			gpu_layer_norm_wt_bias_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (wt_grad, bias_grad, top_gradient, feeder_gradient, ln, sd, wt, num_outputs, loop_count);


			//warps_per_block = 16;
			//num_blocks = (dst_dims[ndims - 1] + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
			//gpu_layer_norm_wt_bias_backward_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (wt_grad, bias_grad, top_gradient, feeder_gradient, ln, sd, wt, numels, dst_dims[ndims-1], warps_per_block);
		}
		else
		{
			feeder_gradient = top_gradient;
		}

		num_blocks = numels / dst_dims[ndims - 1];
		loop_count = (dst_dims[ndims - 1] + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
		gpu_layer_norm_backward_kernel<Dtype> << <num_blocks, CUDA_WARP_SIZE >> > (bottom_gradient, wt, ln, sd, feeder_gradient, numels, dst_dims[ndims - 1], 1.0f / dst_dims[ndims - 1], loop_count);
	}

}

//-----------------------------------------------------------------------------------------------------
//
// transpose functions
//
//-----------------------------------------------------------------------------------------------------
__global__ void gpu_transpose_kernel(const float* __restrict__ A, float* At, int N, OffsetCalc_transpose offs_calc)
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


	OffsetCalc_transpose off_calc(ndims, a_strides, at_strides);


	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	num_blocks = (static_cast<int>(N) + 256 - 1) / 256;


	gpu_transpose_kernel << < num_blocks, defa_threads >> > ((float*)A, (float*)At, static_cast<int>(N), off_calc);

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
void gpu_repeat(Dtype* dst, const Dtype* src, uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, int ndims)
{
	uint64_t N;
	int num_blocks;
	uint32_t factor;

	OffsetCalc_repeat offs_calc(strides_dst, strides_src, dims_src, ndims);

	N = numels;

	assert(N < UINT_MAX); // offsets are 32 bit

	factor = defa_threads * vec_size;

	num_blocks = (static_cast<int>(N) + factor - 1) / factor; // allocate (1/vec_size) threads since vectorization kernel to be used

	gpu_repeat_vectorized_kernel << < num_blocks, defa_threads >> > ((float*)dst, (float*)src, static_cast<int>(N), offs_calc);

}

template<typename Dtype>
__global__ void gpu_repeat_backward_vectorized_kernel(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t repeat_factor, OffsetCalc_repeat_backwards offs)
{
	uint32_t thread_id;
	uint32_t grid_stride;
	uint32_t i;
	uint32_t j;
	uint32_t offset;
	uint32_t fine_index;
	float4 src4;
	float4 dst4;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	
	grid_stride = blockDim.x * gridDim.x * vec_size;

	fine_index = thread_id * vec_size;

	for (i = fine_index; i < numels_dst; i += grid_stride)
	{
		dst4.x = dst4.y = dst4.z = dst4.w = 0;
		for (j = 0; j < repeat_factor; j++)
		{
			offset = offs.GetOffset(i, j);
			src4 = *(reinterpret_cast<const float4*>(&src[offset]));

			dst4.x += src4.x;
			dst4.y += src4.y;
			dst4.z += src4.z;
			dst4.w += src4.w;
		}
		*(reinterpret_cast<float4*>(&dst[i])) = dst4;
	}
}

template<typename Dtype>
__global__ void gpu_repeat_backward_kernel(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t repeat_factor, OffsetCalc_repeat_backwards offs)
{
	uint32_t thread_id;
	uint32_t grid_stride;
	uint32_t i;
	uint32_t j;
	uint32_t offset;
	Dtype val;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	grid_stride = blockDim.x * gridDim.x;


	for (i = thread_id; i < numels_dst; i += grid_stride)
	{
		val = 0;
		for (j = 0; j < repeat_factor; j++)
		{
			offset = offs.GetOffset(i, j);
			val += src[offset];
		}
		dst[i] = val;
	}
}

template<typename Dtype>
void gpu_repeat_backward(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t numels_src, const uint64_t* dims_src, int ndims_src, OffsetCalc_repeat_backwards* offs)
{
	uint32_t threads_required;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;

	threads_required = static_cast<uint32_t>(numels_dst);
	warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	
	//TODO OPTOPT: add vectorization support even when last dim not a multiple of 4 (also add support for when first non-1 dim is a multiple of 4, e.g if dims are 2, 2, 8, 1, 1, 1 this should work with vectorized kernel)
	if (dims_src[ndims_src - 1] % 4)
	{
		warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
		num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

		gpu_repeat_backward_kernel << < num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (dst, src, numels_dst, numels_src / numels_dst, *offs);
	}
	else
	{
		gpu_repeat_backward_vectorized_kernel << < num_blocks, warps_per_block * CUDA_WARP_SIZE / vec_size >> > (dst, src, numels_dst, numels_src / numels_dst, *offs);
	}
	
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
		src_offset = offs_calc.GetOffset(dst_offset);

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

	gpu_repeat_interleave_kernel << < num_blocks, defa_threads >> > ((float*)dst, (float*)src, static_cast<int>(N), offs_calc);

}



template<typename Dtype>
__global__ void gpu_repeat_interleave_backward_broadcast_kernel(Dtype* dst, const Dtype* src, uint64_t numels_src, uint32_t repeat_dim_dim, uint32_t repeat, uint32_t stride, uint32_t warps_per_src_bloc, OffsetCalc_repeat_interleave offs)
{
	uint32_t dst_offset;
	uint32_t src_offset;
	int i;
	int j;
	float val;

	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	uint32_t warpIdx = thread_id / warpSize;	
	uint32_t srcBlockIdx = warpIdx / warps_per_src_bloc;
	uint32_t laneId = threadIdx.x % warpSize;

	for (i = laneId; i < stride; i += warpSize)
	{
		src_offset = srcBlockIdx * repeat * stride + i;
		if (src_offset >= numels_src)
		{
			break;
		}
		dst_offset = offs.GetOffset(src_offset);
		val = 0;

		for (j = warpIdx % warps_per_src_bloc; j < repeat; j += warps_per_src_bloc)
		{
			val += src[src_offset + j * stride];
		}

		atomicAdd((float*)&dst[dst_offset], val);
	}

}



template<typename Dtype>
__global__ void gpu_repeat_interleave_backward_vectorized_kernel(Dtype* dst, const Dtype* src, uint64_t numels_src, OffsetCalc_repeat_interleave offs)
{
	uint32_t thread_id;
	uint32_t dst_offset;
	float4 src4;
	int i;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	thread_id *= vec_size;

	if (thread_id >= numels_src)
	{
		return;
	}

	
	src4 = *(reinterpret_cast<const float4*>(&src[thread_id]));
#pragma unroll
	for (i = 0; i < vec_size; i++)
	{
		dst_offset = offs.GetOffset(thread_id + i);
		atomicAdd((float*)&dst[dst_offset], ((float*)&src4)[i]);
	}
}


template<typename Dtype>
__global__ void gpu_repeat_interleave_backward_kernel(Dtype* dst, const Dtype* src, uint64_t numels_src, OffsetCalc_repeat_interleave offs)
{
	uint32_t thread_id;
	uint32_t dst_offset;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	if (thread_id >= numels_src)
	{
		return;
	}

	dst_offset = offs.GetOffset(thread_id);
	
	atomicAdd((float*)&dst[dst_offset], src[thread_id]);
}


template<typename Dtype>
void gpu_repeat_interleave_broadcast_backward(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t numels_src, uint32_t repeat_dim_dim, uint32_t repeat, uint32_t stride, OffsetCalc_repeat_interleave* offs) // special case for when all repeat values are the same (much faster)
{
	uint32_t num_src_blocks;
	uint32_t warps_per_src_bloc;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;

	cudaMemsetAsync(dst, 0, sizeof(Dtype) * numels_dst); // get this going now...

	num_src_blocks = static_cast<uint32_t>(numels_src) / repeat / stride;
	warps_per_src_bloc = min(128, repeat); // <----------------tweak this for perf but make less than repeat
	warps_required = num_src_blocks * warps_per_src_bloc;


	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	gpu_repeat_interleave_backward_broadcast_kernel << < num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (dst, src, numels_src, repeat_dim_dim, repeat, stride, warps_per_src_bloc, *offs);

}

template<typename Dtype>
void gpu_repeat_interleave_backward(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t numels_src, OffsetCalc_repeat_interleave* offs)
{
	uint32_t threads_required;
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;

	cudaMemsetAsync(dst, 0, sizeof(Dtype) * numels_dst); // get this going now...

	threads_required = static_cast<uint32_t>(numels_src);
	warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;


	if (numels_src % 4)
	{
		gpu_repeat_interleave_backward_kernel << < num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (dst, src, numels_src, *offs);
	}
	else
	{
		gpu_repeat_interleave_backward_vectorized_kernel << < num_blocks, warps_per_block * CUDA_WARP_SIZE / 4 >> > (dst, src, numels_src, *offs);		
	}
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


template<typename Dtype>
__global__ void gpu_index_backward_kernel(Dtype* dst, const Dtype* src, const int* indices, int num_indices, uint32_t copy_len)
{
	uint32_t thread_id;
	uint32_t index;
	uint32_t i;
	uint32_t j;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	if (thread_id >= num_indices * copy_len)
	{
		return;
	}

	i = thread_id / copy_len;
	j = thread_id % copy_len;

	index = indices[i];

	atomicAdd((float*)&dst[index * copy_len + j], src[i * copy_len + j]);

}

template<typename Dtype>
void gpu_index_backward(Dtype* dst, uint64_t numels_dst, const Dtype* src, const int* indices, int num_indices, uint64_t copy_len)
{
	int threads_required;
	int warps_required;
	int warps_per_block;
	int num_blocks;

	cudaMemsetAsync(dst, 0, sizeof(Dtype) * numels_dst); // get this going now...

	threads_required = num_indices * static_cast<int>(copy_len);
	warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	gpu_index_backward_kernel << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (dst, src, indices, num_indices, static_cast<uint32_t>(copy_len));

}
//-----------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------
//
// permutation functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_permute_vectorized_kernel(Dtype* dst, const Dtype* src, const uint64_t N, OffsetCalc_permutaion ofs)
{
	uint32_t thread_id;
	uint32_t i;
	uint32_t j;
	uint32_t grid_stride;
	float4 val4;
	uint32_t src_offset;
	uint32_t dst_offset;
	uint32_t remainder;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	grid_stride = blockDim.x * gridDim.x;

	for (i = thread_id; i < N / vec_size; i += grid_stride)
	{
		src_offset = i * vec_size;

		val4 = *(reinterpret_cast<const float4*>(&src[src_offset]));
#pragma unroll
		for (j = 0; j < vec_size; j++)
		{
			dst_offset = ofs.GetOffset(src_offset);
			dst[dst_offset] = ((float*)&val4)[j];
			src_offset++;
		}
	}

	remainder = N % vec_size;
	if ((thread_id == 0) && remainder)
	{
		src_offset = N - remainder;
		for (j = 0; j < remainder; j++)
		{
			dst_offset = ofs.GetOffset(src_offset);
			dst[dst_offset] = src[src_offset];
			src_offset++;
		}
	}
}

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
void gpu_permute(Dtype* dst, const Dtype* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutations, bool reverse )
{
	//-------------------------------
	// use reverse mode for back prop
	//-------------------------------

	OffsetCalc_permutaion ofs(strides_dst, strides_src, permutations, ndims, reverse);

	int threads_required = static_cast<int>((numels + vec_size - 1)/ vec_size);
	//int threads_required = static_cast<int>(numels);

	int warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	int warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	int num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;


	int threads_per_block = CUDA_WARP_SIZE * warps_per_block;

	//gpu_permute_kernel << < num_blocks, threads_per_block >> > (dst, src, numels, ofs);
	gpu_permute_vectorized_kernel << < num_blocks, threads_per_block >> > (dst, src, numels, ofs);
	
}

//-----------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------
//
// gelu functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_gelu_vectorized_kernel(Dtype* dst, Dtype* src, unsigned int len)
{
	int i;
	float4 src4;
	float4 dst4;
	int threadId;
	float val;
	int remainder;

	threadId = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	threadId *= vec_size;

	remainder = len - threadId;

	if (remainder >= vec_size)
	{
		src4 = *(reinterpret_cast<const float4*>(&src[threadId]));

		val = ((float*)&src4)[0];
		((float*)&dst4)[0] = val * (0.5f * (1.0f + std::erf(val / 1.4142135623730950488016887242097f)));

		val = ((float*)&src4)[1];
		((float*)&dst4)[1] = val * (0.5f * (1.0f + std::erf(val / 1.4142135623730950488016887242097f)));

		val = ((float*)&src4)[2];
		((float*)&dst4)[2] = val * (0.5f * (1.0f + std::erf(val / 1.4142135623730950488016887242097f)));

		val = ((float*)&src4)[3];
		((float*)&dst4)[3] = val * (0.5f * (1.0f + std::erf(val / 1.4142135623730950488016887242097f)));

		*(reinterpret_cast<float4*>(&dst[threadId])) = dst4;
	}
	else
	{
		for (i = 0; i < remainder; i++)
		{
			val = src[threadId + i];
			dst[threadId + i] = val * (0.5f * (1.0f + std::erf(val / 1.4142135623730950488016887242097f)));
		}
	}
}


template<typename Dtype>
__global__ void gpu_gelu_kernel(Dtype* dst, Dtype* src, unsigned int len)
{
	int i;
	float val;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		val = src[i];
		dst[i] = val * (0.5f * (1.0f + std::erf(val / 1.4142135623730950488016887242097f))); //dst[i] = src[i] * (0.5f * (1.0f + std::erf(src[i] / sqrt(2.0f))));

		// approximation
		//float tt = 0.79788456080286535587989211986876f * (val + 0.044715f * (val * val * val));
		//dst[i] = 0.5f * val * (1.0f + tanh(tt));
	}

}

template<typename Dtype>
void gpu_gelu(Dtype* dst, Dtype* src, uint64_t len)
{
	int threads_required = static_cast<int>((len + vec_size - 1) / vec_size);
	int warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	int warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	int num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;
	int threads_per_block = CUDA_WARP_SIZE * warps_per_block;

	//gpu_gelu_kernel << <num_blocks, 512 >> > (dst, src, (unsigned int)len);
	gpu_gelu_vectorized_kernel << <num_blocks, threads_per_block >> > (dst, src, (unsigned int)len);

}


template<typename Dtype>
__global__ void gpu_gelu_backward_vectorized_kernel(Dtype* bottom_gradient, const Dtype* top_gradient, const Dtype* src, unsigned int len)
{
	int threadId;
	float pdf;
	float src_val;
	float bot_val;
	float top_val;
	float4 src4;
	float4 bot4;
	float4 top4;
	int i;
	int remainder;

	threadId = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	threadId *= vec_size;

	remainder = len - threadId;

	if (remainder >= vec_size)
	{
		src4 = *(reinterpret_cast<const float4*>(&src[threadId]));
		top4 = *(reinterpret_cast<const float4*>(&top_gradient[threadId]));
#pragma unroll
		for (i = 0; i < vec_size; i++)
		{
			src_val = ((float*)&src4)[i];
			top_val = ((float*)&top4)[i];

			pdf = 0.39894228040143267793994605993438f;
			pdf *= expf(-0.5 * src_val * src_val);

			bot_val = (0.5f * (1.0f + std::erf(src_val * 0.70710678118654752440084436210485f))) + src_val * pdf;
			bot_val *= top_val;

			((float*)&bot4)[i] = bot_val;
		}

		*(reinterpret_cast<float4*>(&bottom_gradient[threadId])) = bot4;
	}
	else
	{
		for (i = 0; i < remainder; i++)
		{
			src_val = src[threadId + i];
			top_val = top_gradient[threadId + i];

			pdf = 0.39894228040143267793994605993438f;
			pdf *= expf(-0.5 * src_val * src_val);

			bot_val = (0.5f * (1.0f + std::erf(src_val * 0.70710678118654752440084436210485f))) + src_val * pdf;
			bot_val *= top_val;

			bottom_gradient[threadId + i] = bot_val;
		}
	}
/*
	if (threadId < len)
	{
		src4 = *(reinterpret_cast<const float4*>(&src[threadId]));
		top4 = *(reinterpret_cast<const float4*>(&top_gradient[threadId]));
#pragma unroll
		for (i = 0; i < vec_size; i++)
		{
			src_val = ((float*)&src4)[i];
			top_val = ((float*)&top4)[i];

			pdf = 0.39894228040143267793994605993438f;
			pdf *= expf(-0.5 * src_val * src_val);

			bot_val = (0.5f * (1.0f + std::erf(src_val * 0.70710678118654752440084436210485f))) + src_val * pdf;
			bot_val *= top_val;

			((float*)&bot4)[i] = bot_val;
		}

		*(reinterpret_cast<float4*>(&bottom_gradient[threadId])) = bot4;
	}
	*/
}


template<typename Dtype>
__global__ void gpu_gelu_backward_kernel(Dtype* bottom_gradient, const Dtype* top_gradient, const Dtype* src, unsigned int len)
{
	int i;
	float pdf;
	float src_val;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		src_val = src[i];

		pdf = 0.39894228040143267793994605993438f;
		pdf *= expf(-0.5 * src_val * src_val);

		bottom_gradient[i] = (0.5f * (1.0f + std::erf(src_val * 0.70710678118654752440084436210485f))) + src_val * pdf; //bottom_gradient[i] = (0.5f * (1.0f + std::erf(src[i] / sqrt(2.0f)))) + src[i] * pdf;
		bottom_gradient[i] *= top_gradient[i];
	}
}

template<typename Dtype>
void gpu_gelu_backward(Dtype* bottom_gradient, const Dtype* top_gradient, const Dtype* src, uint64_t len)
{
	int threads_required = static_cast<int>((len + vec_size - 1) / vec_size);
	int warps_required = (threads_required + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
	int warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	int num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;
	int threads_per_block = CUDA_WARP_SIZE * warps_per_block;

	//gpu_gelu_backward_kernel << <num_blocks, 512 >> > (bottom_gradient, top_gradient, src, (unsigned int)len);
	gpu_gelu_backward_vectorized_kernel << <num_blocks, threads_per_block >> > (bottom_gradient, top_gradient, src, (unsigned int)len);
}
//-----------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------
//
// nll functions
//
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
__global__ void gpu_nll_backward_kernel(Dtype* bottom_gradient, const Dtype* one_hot_indices, uint64_t one_hot_tensor_size, OffsetCalc_nll ofs, Dtype gradient)
{
	uint32_t i;
	uint32_t threadId;

	threadId = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	if (threadId >= one_hot_tensor_size)
	{
		return;
	}

	i = threadId;
	bottom_gradient[ofs.GetOffset(i, (int)one_hot_indices[i])] = gradient;

}


template<typename Dtype>
void gpu_nll_backward(Dtype* bottom_gradient, const Dtype* one_hot_indices, uint64_t one_hot_tensor_size, OffsetCalc_nll* ofs)
{
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	Dtype gradient;

	cudaMemsetAsync(bottom_gradient, 0, sizeof(Dtype));

	warps_required = (one_hot_tensor_size + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE; // one thread per work item (i.e. one_hot_tensor tensor element)
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	gradient = static_cast<Dtype>(1.0f / (-1.0f * one_hot_tensor_size));

	gpu_nll_backward_kernel << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (bottom_gradient, one_hot_indices, one_hot_tensor_size, *ofs, gradient);
}


template<typename Dtype>
__global__ void gpu_nll_kernel(Dtype* loss, const Dtype* probabilities, const Dtype* one_hot_indices, uint64_t one_hot_tensor_size, OffsetCalc_nll ofs, Dtype scale)
{
	uint32_t i;
	uint32_t laneId;	
	uint32_t  threadId;
	uint32_t stride;
	Dtype loss_val;

	threadId = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	laneId = threadIdx.x % warpSize;
	stride = gridDim.x * blockDim.x;

	loss_val = 0;
	for (i = threadId; i < one_hot_tensor_size; i += stride)
	{
		loss_val += probabilities[ofs.GetOffset(i, (int)one_hot_indices[i])];
	}

	loss_val += __shfl_down_sync(FULL_MASK, loss_val, 16);
	loss_val += __shfl_down_sync(FULL_MASK, loss_val, 8);
	loss_val += __shfl_down_sync(FULL_MASK, loss_val, 4);
	loss_val += __shfl_down_sync(FULL_MASK, loss_val, 2);
	loss_val += __shfl_down_sync(FULL_MASK, loss_val, 1);

	if (laneId == 0)
	{
		loss_val *= scale;
		atomicAdd((float*)loss, loss_val);
	}

}

template<typename Dtype>
void gpu_nll(Dtype* loss, const Dtype* probabilities, const Dtype* one_hot_indices, uint64_t one_hot_tensor_size, OffsetCalc_nll* ofs)
{
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	Dtype scale;

	cudaMemsetAsync(loss, 0, sizeof(Dtype));

	//
	// one thread per work item (i.e. one_hot_tensor tensor element), and no looping if using default num_blocks
	//
	warps_required = (one_hot_tensor_size + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE; 
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;
	
	// reduce atomicAdd pressure by reducing num_blocks (and thereby forcing kernel to loop)
	num_blocks = 1; // what is a good metric? Need to also prevent too may loop iterations

	scale = static_cast<Dtype>(1.0f/(-1.0f * one_hot_tensor_size));
	gpu_nll_kernel << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (loss, probabilities, one_hot_indices, one_hot_tensor_size, *ofs, scale);
}
//-----------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------
//
// helper functions for cublasSgemmBatched
//
//-----------------------------------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------
//
// misc reduction functions
//
//-----------------------------------------------------------------------------------------------------

template<typename Dtype>
__global__ void gpu_reduce_kernel(uint32_t numoutputs, OffsetCalc_reverse_broadcast offs, Dtype* dst, Dtype* src, uint32_t loop_count)
{
	// one warp for each output
	uint32_t warpId;
	uint32_t laneId;
	uint32_t i;
	uint32_t other_operand_offset;
	uint32_t tg_offset;
	Dtype val;

	warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // absolute thread id /  warpSize
	if (warpId >= numoutputs)
	{
		return;
	}

	laneId = threadIdx.x % warpSize;

	val = 0;
	for (i = laneId; i < loop_count; i += warpSize)
	{
		offs.GetOffsets(warpId, i, &other_operand_offset, &tg_offset);
		val += src[tg_offset];
	}

	//
	// reduce
	//
	val += __shfl_down_sync(FULL_MASK, val, 16);
	val += __shfl_down_sync(FULL_MASK, val, 8);
	val += __shfl_down_sync(FULL_MASK, val, 4);
	val += __shfl_down_sync(FULL_MASK, val, 2);
	val += __shfl_down_sync(FULL_MASK, val, 1);

	if (laneId == 0)
	{
		dst[warpId] = val;
	}

}

template<typename Dtype>
void gpu_reduce(uint32_t numels_dst, uint32_t numels_src, OffsetCalc_reverse_broadcast* offs, Dtype* dst, Dtype* src) // reducing src into dst
{
	uint32_t warps_required;
	uint32_t warps_per_block;
	uint32_t num_blocks;
	uint32_t loop_count;


	loop_count = numels_src / numels_dst;
	warps_required = static_cast<uint32_t>(numels_dst); // one warp for each output
	warps_per_block = min(LTEN_MAX_WARPS_PER_BLOCK, warps_required);
	num_blocks = (warps_required + warps_per_block - 1) / warps_per_block;

	gpu_reduce_kernel<Dtype> << <num_blocks, warps_per_block * CUDA_WARP_SIZE >> > (numels_dst, *offs, dst, src, loop_count);
	
}

//-----------------------------------------------------------------------------------------------------
//
// memory test functions for debugging perf
//
//-----------------------------------------------------------------------------------------------------
__global__ void memory_test_kernel3(float* dst, const float* src, int numels)
{
	uint32_t thread_id;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	atomicAdd((float*)&dst[thread_id], src[thread_id]);

}


__global__ void memory_test_kernel2(float* dst, const float* __restrict__ src, int numels)
{
	uint32_t thread_id;
	float4 val4;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;
	thread_id *= 4;

	val4 = *(reinterpret_cast<const float4*>(&src[thread_id]));

	*(reinterpret_cast<float4*>(&dst[thread_id])) = val4;
}

__global__ void memory_test_kernel(float* dst, const float* src, int numels)
{
	uint32_t thread_id;

	thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global index;

	dst[thread_id] = src[thread_id];
}

void memory_test(float* dst, float* src, int numels)
{
	int num_blocks;

	num_blocks = numels / (32 * 16);
	memory_test_kernel << <num_blocks, 32 * 16 >> > (dst, src, numels);
	//memory_test_kernel2 << <num_blocks/4, 32 * 16 >> > (dst, src, numels);
	//memory_test_kernel3 << <num_blocks, 32 * 16 >> > (dst, src, numels);
}

template void gpu_mul<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_mul<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_mul<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);

template void gpu_mul_backward<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_mul_backward<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_mul_backward<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);

template void gpu_mul_backward<float>(float* top_gradient, float* bottom_gradient, float* other_operand, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);
template void gpu_mul_backward<int>(int* top_gradient, int* bottom_gradient, int* other_operand, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);
template void gpu_mul_backward<uint8_t>(uint8_t* top_gradient, uint8_t* bottom_gradient, uint8_t* other_operand, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);

template void gpu_mul<float>(float* A, float* B, float* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_mul<int>(int* A, int* B, int* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_mul<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);

template void gpu_div<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_div<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_div<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);

template void gpu_div_backward<float>(uint64_t N, float* op1, float* op2, float* tg, float* bg, bool divisor);
template void gpu_div_backward<int>(uint64_t N, int* op1, int* op2, int* tg, int* bg, bool divisor);
template void gpu_div_backward<uint8_t>(uint64_t N, uint8_t* op1, uint8_t* op2, uint8_t* tg, uint8_t* bg, bool divisor);


template void gpu_div_backward<float>(float* top_gradient, float* bottom_gradient, float* operand1, float* operand2, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, bool divisor);
template void gpu_div_backward<int>(int* top_gradient, int* bottom_gradient, int* operand1, int* operand2, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, bool divisor);
template void gpu_div_backward<uint8_t>(uint8_t* top_gradient, uint8_t* bottom_gradient, uint8_t* operand1, uint8_t* operand2, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, bool divisor);

template void gpu_div<float>(float* A, float* B, float* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_div<int>(int* A, int* B, int* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_div<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);

template void gpu_add<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_add<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_add<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);

template void gpu_add_backward<float>(uint64_t N, float* top_gradient, float* bottom_gradient);
template void gpu_add_backward<int>(uint64_t N, int* top_gradient, int* bottom_gradient);
template void gpu_add_backward<uint8_t>(uint64_t N, uint8_t* top_gradient, uint8_t* bottom_gradient);

template void gpu_add_backward<float>(float* top_gradient, float* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);
template void gpu_add_backward<int>(int* top_gradient, int* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);
template void gpu_add_backward<uint8_t>(uint8_t* top_gradient, uint8_t* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);

template void gpu_add<float>(float* A, float* B, float* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_add<int>(int* A, int* B, int* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_add<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);

template void gpu_sub<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_sub<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_sub<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);

template void gpu_sub_backward<float>(uint64_t N, float* top_gradient, float* bottom_gradient, float scale);
template void gpu_sub_backward<int>(uint64_t N, int* top_gradient, int* bottom_gradient, int  scale);
template void gpu_sub_backward<uint8_t>(uint64_t N, uint8_t* top_gradient, uint8_t* bottom_gradient, uint8_t scale);

template void gpu_sub_backward<float>(float* top_gradient, float* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, float scale);
template void gpu_sub_backward<int>(int* top_gradient, int* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, int scale);
template void gpu_sub_backward<uint8_t>(uint8_t* top_gradient, uint8_t* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, uint8_t scale);

template void gpu_sub<float>(float* A, float* B, float* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_sub<int>(int* A, int* B, int* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);
template void gpu_sub<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);

template void gpu_mean<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_mean<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_mean<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);

template void gpu_mean_backward<float>(float* bottom_gradient, const float* top_gradient, const uint64_t numels);
template void gpu_mean_backward<int>(int* bottom_gradient, const int* top_gradient, const uint64_t numels);
template void gpu_mean_backward<uint8_t>(uint8_t* bottom_gradient, const uint8_t* top_gradient, const uint64_t numels);

template void gpu_mean_backward<float>(float* dst, const float* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, float scale);
template void gpu_mean_backward<int>(int* dst, const int* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, int scale);
template void gpu_mean_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, uint8_t scale);

template void gpu_mean<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void gpu_mean<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void gpu_mean<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template void gpu_var<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);
template void gpu_var<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);
template void gpu_var<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);

template void gpu_layer_norm<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, float* weights, float* bias, float* ln, float* sd);
template void gpu_layer_norm<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, int* weights, int* bias, int* ln, int* sd);
template void gpu_layer_norm<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, uint8_t* weights, uint8_t* bias, uint8_t* ln, uint8_t* sd);

template void gpu_layer_norm_backwards<float>(void* vlayer_norm, float* x, float* top_gradient, float* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, float* feeder_gradient);
template void gpu_layer_norm_backwards<int>(void* vlayer_norm, int* x, int* top_gradient, int* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, int* feeder_gradient);
template void gpu_layer_norm_backwards<uint8_t>(void* vlayer_norm, uint8_t* x, uint8_t* top_gradient, uint8_t* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, uint8_t* feeder_gradient);

template void gpu_transpose<float>(const float* A, float* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
template void gpu_transpose<int>(const int* A, int* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);
template void gpu_transpose<uint8_t>(const uint8_t* A, uint8_t* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);

template void gpu_repeat<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);
template void gpu_repeat<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);
template void gpu_repeat<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);

template void gpu_repeat_backward<float>(float* dst, const float* src, uint64_t numels_dst, uint64_t numels_src, const uint64_t* dims_src, int ndims_src, OffsetCalc_repeat_backwards* offs);
template void gpu_repeat_backward<int>(int* dst, const int* src, uint64_t numels_dst, uint64_t numels_src, const uint64_t* dims_src, int ndims_src, OffsetCalc_repeat_backwards* offs);
template void gpu_repeat_backward<uint8_t>(uint8_t* dst, const uint8_t* src, uint64_t numels_dst, uint64_t numels_src, const uint64_t* dims_src, int ndims_src, OffsetCalc_repeat_backwards* offs);

template void gpu_repeat_interleave<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);
template void gpu_repeat_interleave<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);
template void gpu_repeat_interleave<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);

template void gpu_repeat_interleave_backward<float>(float* dst, const float* src, uint64_t numels_dst, uint64_t numels_src, OffsetCalc_repeat_interleave* offs);
template void gpu_repeat_interleave_backward<int>(int* dst, const int* src, uint64_t numels_dst, uint64_t numels_src, OffsetCalc_repeat_interleave* offs);
template void gpu_repeat_interleave_backward<uint8_t>(uint8_t* dst, const uint8_t* src, uint64_t numels_dst, uint64_t numels_src, OffsetCalc_repeat_interleave* offs);

template void gpu_repeat_interleave_broadcast_backward<float>(float* dst, const float* src, uint64_t numels_dst, uint64_t numels_src, uint32_t repeat_dim_dim, uint32_t repeat, uint32_t stride, OffsetCalc_repeat_interleave* offs);
template void gpu_repeat_interleave_broadcast_backward<int>(int* dst, const int* src, uint64_t numels_dst, uint64_t numels_src, uint32_t repeat_dim_dim, uint32_t repeat, uint32_t stride, OffsetCalc_repeat_interleave* offs);
template void gpu_repeat_interleave_broadcast_backward<uint8_t>(uint8_t* dst, const uint8_t* src, uint64_t numels_dst, uint64_t numels_src, uint32_t repeat_dim_dim, uint32_t repeat, uint32_t stride, OffsetCalc_repeat_interleave* offs);

template void gpu_index<float>(float* dst, const float* src, const int* indices, uint64_t copy_len, const uint64_t numels);
template void gpu_index<int>(int* dst, const int* src, const int* indices, uint64_t copy_len, const uint64_t numels);
template void gpu_index<uint8_t>(uint8_t* dst, const uint8_t* src, const int* indices, uint64_t copy_len, const uint64_t numels);

template void gpu_index_backward<float>(float* dst, uint64_t numels_dst, const float* src, const int* indices, int num_indices, uint64_t copy_len);
template void gpu_index_backward<int>(int* dst, uint64_t numels_dst, const int* src, const int* indices, int num_indices, uint64_t copy_len);
template void gpu_index_backward<uint8_t>(uint8_t* dst, uint64_t numels_dst, const uint8_t* src, const int* indices, int num_indices, uint64_t copy_len);

template void gpu_permute<float>(float* dst, const float* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions, bool reverse);
template void gpu_permute<int>(int* dst, const int* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions, bool reverse);
template void gpu_permute<uint8_t>(uint8_t* dst, const uint8_t* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions, bool reverse);

template void set_addresses<float>(float* A, float* B, float* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);
template void set_addresses<int>(int* A, int* B, int* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);
template void set_addresses<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);


template void gpu_gelu<float>(float* dst, float* src, uint64_t len);
template void gpu_gelu<int>(int* dst, int* src, uint64_t len);
template void gpu_gelu<uint8_t>(uint8_t* dst, uint8_t* src, uint64_t len);


template void gpu_gelu_backward<float>(float* bottom_gradient, const float* top_gradient, const float* src, uint64_t len);
template void gpu_gelu_backward<int>(int* bottom_gradient, const int* top_gradient, const int* src, uint64_t len);
template void gpu_gelu_backward<uint8_t>(uint8_t* bottom_gradient, const uint8_t* top_gradient, const uint8_t* src, uint64_t len);

template void gpu_reduce<float>(uint32_t numels_dst, uint32_t numels_src, OffsetCalc_reverse_broadcast* offs, float* dst, float* src);
template void gpu_reduce<int>(uint32_t numels_dst, uint32_t numels_src, OffsetCalc_reverse_broadcast* offs, int* dst, int* src);
template void gpu_reduce<uint8_t>(uint32_t numels_dst, uint32_t numels_src, OffsetCalc_reverse_broadcast* offs, uint8_t* dst, uint8_t* src);

template void gpu_nll<float>(float* loss, const float* probabilities, const float* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);
template void gpu_nll<int>(int* loss, const int* probabilities, const int* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);
template void gpu_nll<uint8_t>(uint8_t* loss, const uint8_t* probabilities, const uint8_t* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);

template void gpu_nll_backward<float>(float* bottom_gradient, const float* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);
template void gpu_nll_backward<int>(int* bottom_gradient, const int* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);
template void gpu_nll_backward<uint8_t>(uint8_t* bottom_gradient, const uint8_t* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);

