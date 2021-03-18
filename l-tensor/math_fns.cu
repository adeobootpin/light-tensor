#include <assert.h>
#include <cstdint>
#include <stdio.h>
#include "error.h"

const int DEFA_BLOCK_X = 32;
const int DEFA_BLOCK_Y = 32;
const int DEFA_THREADS = 1024;

const int DEFA_REDUCTION_THREADS = 256;
const int MAX_REDUCTION_BLOCKS = 64;

extern __shared__ float shared_mem_block[];

__global__ void gpu_axpy_kernel(float* A, float* C, int height_A, int width_A, int height_C, int width_C, int max_height, int max_width)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y > (max_height - 1) || x > (max_width - 1))
	{
		return;
	}

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_c_y = min(y, (height_C - 1));
	unsigned int broadcast_c_x = min(x, (width_C - 1));


	atomicAdd(&C[broadcast_c_y * width_C + broadcast_c_x], A[broadcast_a_y * width_A + broadcast_a_x]);
}


template<typename Dtype>
__global__ void gpu_sub_kernel(Dtype* A, Dtype* B, Dtype* C, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int i = y * width_C + x;


	if ((y < height_C) && (x < width_C))
	{
		C[i] = A[broadcast_a_y * width_A + broadcast_a_x] - B[broadcast_b_y * width_B + broadcast_b_x];
	}
}


template<typename Dtype>
void gpu_sub(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B)
{
	int height;
	int width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	height = max((int)height_A, (int)height_B);
	width = max((int)width_A, (int)width_B);

	dimGrid.x = (int)ceil((float)width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)height / (float)dimBlock.y);

	gpu_sub_kernel<Dtype> << <dimGrid, dimBlock >> > (A, B, C, (int)height_A, (int)width_A, (int)height_B, (int)width_B, height, width);

}


template<typename Dtype>
__global__ void gpu_sum_kernel(Dtype* A, Dtype* B, Dtype* C, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int i = y * width_C + x;


	if ((y < height_C) && (x < width_C))
	{
		C[i] = A[broadcast_a_y * width_A + broadcast_a_x] + B[broadcast_b_y * width_B + broadcast_b_x];
	}

}


template<typename Dtype>
void gpu_sum(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B)
{
	int height;
	int width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	height = max((int)height_A, (int)height_B);
	width = max((int)width_A, (int)width_B);

	dimGrid.x = (int)ceil((float)width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)height / (float)dimBlock.y);

	gpu_sum_kernel<Dtype> << <dimGrid, dimBlock >> > (A, B, C, (int)height_A, (int)width_A, (int)height_B, (int)width_B, height, width);

}

template<typename Dtype>
__global__ void gpu_mul_kernel(Dtype* A, Dtype* B, Dtype* C, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C, Dtype beta)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int i = y * width_C + x;

	if ((y < height_C) && (x < width_C))
	{
		C[i] = beta * C[i] + A[broadcast_a_y * width_A + broadcast_a_x] * B[broadcast_b_y * width_B + broadcast_b_x];
	}
}

template<typename Dtype>
void gpu_mul(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, Dtype beta)
{
	int height;
	int width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	height = max((int)height_A, (int)height_B);
	width = max((int)width_A, (int)width_B);

	dimGrid.x = (int)ceil((float)width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)height / (float)dimBlock.y);

	gpu_mul_kernel<Dtype> << <dimGrid, dimBlock >> > (A, B, C, (int)height_A, (int)width_A, (int)height_B, (int)width_B, height, width, beta);
}


template<typename Dtype>
__global__ void gpu_mul_kernel(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint64_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = A[i] * B[i];
	}
}

template<typename Dtype>
void gpu_mul(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;


	gpu_mul_kernel<Dtype> << <num_blocks, DEFA_THREADS >> > (N, A, B, C);

}


__global__ void gpu_mul_kernel(float* A, float* B, float* C, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C, int max_height, int max_width)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y > (max_height - 1) || x > (max_width - 1))
	{
		return;
	}

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int broadcast_c_y = min(y, (height_C - 1));
	unsigned int broadcast_c_x = min(x, (width_C - 1));



	atomicAdd(&C[broadcast_c_y * width_C + broadcast_c_x], A[broadcast_a_y * width_A + broadcast_a_x] * B[broadcast_b_y * width_B + broadcast_b_x]);
}

void gpu_mul(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint64_t height_C, uint64_t width_C)
{
	int max_height;
	int max_width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	max_height = max((int)height_A, (int)height_B);
	max_width = max((int)width_A, (int)width_B);

	max_height = max((int)max_height, (int)height_C);
	max_width = max((int)max_width, (int)width_C);


	dimGrid.x = (int)ceil((float)max_width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)max_height / (float)dimBlock.y);

	gpu_mul_kernel << <dimGrid, dimBlock >> > (A, B, C, (int)height_A, (int)width_A, (int)height_B, (int)width_B, (int)height_C, (int)width_C, max_height, max_width);
}

template<typename Dtype>
__global__ void gpu_mul_kernel(unsigned int N, Dtype alpha, Dtype* A, Dtype* B, Dtype beta, Dtype* C)
{
	unsigned int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		if (beta == 0)
		{
			if (alpha == 1)
			{
				C[i] = A[i] * B[i];
			}
			else
			{
				C[i] = alpha * A[i] * B[i];
			}
		}
		else
		{
			if (alpha == 1)
			{
				C[i] = C[i] * beta + A[i] * B[i];
			}
			else
			{
				C[i] = C[i] * beta + alpha * A[i] * B[i];
			}
		}
	}
}

template<typename Dtype>
void gpu_mul(uint64_t N, Dtype alpha, Dtype* A, Dtype* B, Dtype beta, Dtype* C) // c = beta * c + alpha * a * b
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_mul_kernel << <num_blocks, DEFA_THREADS >> > (static_cast<unsigned int>(N), alpha, A, B, beta, C);
}


template<typename Dtype>
__global__ void gpu_mul_kernel(unsigned int N, Dtype alpha, Dtype* A, Dtype* B)
{
	unsigned int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		B[i] = alpha * A[i];
	}
}


template<typename Dtype>
void gpu_mul(uint64_t N, Dtype alpha, Dtype* A, Dtype* B) // b = alpha * a
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_mul_kernel << <num_blocks, DEFA_THREADS >> > (static_cast<unsigned int>(N), alpha, A, B);

}


__global__ void gpu_div_kernel(float* A, float* B, float* C, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C, int max_height, int max_width)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y > (max_height - 1) || x > (max_width - 1))
	{
		return;
	}

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int broadcast_c_y = min(y, (height_C - 1));
	unsigned int broadcast_c_x = min(x, (width_C - 1));



	atomicAdd(&C[broadcast_c_y * width_C + broadcast_c_x], A[broadcast_a_y * width_A + broadcast_a_x] / B[broadcast_b_y * width_B + broadcast_b_x]);
}

void gpu_div(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint64_t height_C, uint64_t width_C)
{
	int max_height;
	int max_width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	max_height = max((int)height_A, (int)height_B);
	max_width = max((int)width_A, (int)width_B);

	max_height = max((int)max_height, (int)height_C);
	max_width = max((int)max_width, (int)width_C);


	dimGrid.x = (int)ceil((float)max_width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)max_height / (float)dimBlock.y);

	gpu_div_kernel << <dimGrid, dimBlock >> > (A, B, C, (int)height_A, (int)width_A, (int)height_B, (int)width_B, (int)height_C, (int)width_C, max_height, max_width);
}



__global__ void gpu_div_back_kernel(float* A, float* B, float* C, float* D, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C, int height_D, int width_D, int max_height, int max_width)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	float c_val;

	if (y > (max_height - 1) || x > (max_width - 1))
	{
		return;
	}

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int broadcast_c_y = min(y, (height_C - 1));
	unsigned int broadcast_c_x = min(x, (width_C - 1));

	unsigned int broadcast_d_y = min(y, (height_D - 1));
	unsigned int broadcast_d_x = min(x, (width_D - 1));

	c_val = C[broadcast_c_y * width_C + broadcast_c_x];

	atomicAdd(&D[broadcast_d_y * width_D + broadcast_d_x], A[broadcast_a_y * width_A + broadcast_a_x] * (-B[broadcast_b_y * width_B + broadcast_b_x]) / (c_val * c_val));
}

void gpu_div_back(float* A, float* B, float* C, float* D, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint64_t height_C, uint64_t width_C, uint64_t height_D, uint64_t width_D)
{
	int max_height;
	int max_width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	max_height = max((int)height_A, (int)height_B);
	max_width = max((int)width_A, (int)width_B);

	max_height = max((int)max_height, (int)height_C);
	max_width = max((int)max_width, (int)width_C);

	max_height = max((int)max_height, (int)height_D);
	max_width = max((int)max_width, (int)width_D);

	dimGrid.x = (int)ceil((float)max_width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)max_height / (float)dimBlock.y);

	gpu_div_back_kernel << <dimGrid, dimBlock >> > (A, B, C, D, (int)height_A, (int)width_A, (int)height_B, (int)width_B, (int)height_C, (int)width_C, (int)height_D, (int)width_D, max_height, max_width);
}

template<typename Dtype>
__global__ void gpu_div_kernel(Dtype* A, Dtype* B, Dtype* C, int height_A, int width_A, int height_B, int width_B, int height_C, int width_C)
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int broadcast_a_y = min(y, (height_A - 1));
	unsigned int broadcast_a_x = min(x, (width_A - 1));

	unsigned int broadcast_b_y = min(y, (height_B - 1));
	unsigned int broadcast_b_x = min(x, (width_B - 1));

	unsigned int i = y * width_C + x;

	if (y > (height_C - 1) || x > (width_C - 1))
	{
		return;
	}

	C[i] = A[broadcast_a_y * width_A + broadcast_a_x] / B[broadcast_b_y * width_B + broadcast_b_x];
}

template<typename Dtype>
void gpu_div(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B)
{
	int height;
	int width;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	height = max((int)height_A, (int)height_B);
	width = max((int)width_A, (int)width_B);

	dimGrid.x = (int)ceil((float)width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)height / (float)dimBlock.y);

	gpu_div_kernel<Dtype> << <dimGrid, dimBlock >> > (A, B, C, (int)height_A, (int)width_A, (int)height_B, (int)width_B, height, width);
}


template<typename Dtype>
__global__ void gpu_div_kernel(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	uint64_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		C[i] = A[i] / B[i];
	}
}


template<typename Dtype>
void gpu_div(uint64_t N, Dtype* A, Dtype* B, Dtype* C)
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_div_kernel << <num_blocks, DEFA_THREADS >> > (N, A, B, C);

}



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
__global__ void gpu_sum_kernel(Dtype* data, Dtype* partial_sums, uint64_t len)
{
	uint64_t i;
	int thread_per_block;
	int thread_id;
	int grid_size;
	Dtype thread_sum;
	Dtype* shared_memory;
	
	i = blockIdx.x * blockDim.x + threadIdx.x;

	thread_id = threadIdx.x;

	thread_per_block = blockDim.x;
	shared_memory = (Dtype*)shared_mem_block;

	grid_size = thread_per_block * gridDim.x;

	thread_sum = 0;
	while (i < len)
	{
		thread_sum += data[i];
		i += grid_size;
	}

	shared_memory[thread_id] = thread_sum;
	__syncthreads();

	for (i = thread_per_block / 2; i > 32; i >>= 1)
	{
		if (thread_id < i)
		{
			shared_memory[thread_id] += shared_memory[thread_id + i];
		}

		__syncthreads();
	}

	if (thread_id < 32)
	{
		warp_reduce(shared_memory, thread_id);
	}

	if (thread_id == 0)
	{
		partial_sums[blockIdx.x] = shared_memory[0];
	}
}


__device__ float partial_sums_block[MAX_REDUCTION_BLOCKS];

template<typename Dtype>
void gpu_sum(Dtype* data, Dtype* sum, uint64_t len)
{
	int num_blocks;
	int thread_per_block = DEFA_REDUCTION_THREADS;
	static Dtype* partial_sums = nullptr;


	num_blocks = min(MAX_REDUCTION_BLOCKS, static_cast<int>(len) / thread_per_block + 1);

	if (!partial_sums)
	{
		cudaGetSymbolAddress((void**)&partial_sums, partial_sums_block);
	}

	gpu_sum_kernel<Dtype> << <num_blocks, thread_per_block, thread_per_block * sizeof(Dtype)>> > (data, partial_sums, len); // level 0 reduction


	thread_per_block = 64; // one warp ( + an extra warp to initialize shared mem)
	gpu_sum_kernel<Dtype> << <1, thread_per_block, thread_per_block * sizeof(Dtype) >> > (partial_sums, sum, num_blocks); // level 1 reduction

}


template<typename Dtype>
__global__ void gpu_sum_kernel(Dtype* data, Dtype* partial_sums, uint64_t len, Dtype scale)
{
	uint64_t i;
	int thread_per_block;
	int thread_id;
	int grid_size;
	Dtype thread_sum;
	Dtype* shared_memory;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	thread_id = threadIdx.x;

	thread_per_block = blockDim.x;
	shared_memory = (Dtype*)shared_mem_block;

	grid_size = thread_per_block * gridDim.x;

	thread_sum = 0;
	while (i < len)
	{
		thread_sum += data[i];
		i += grid_size;
	}

	shared_memory[thread_id] = thread_sum;
	__syncthreads();

	for (i = thread_per_block / 2; i > 32; i >>= 1)
	{
		if (thread_id < i)
		{
			shared_memory[thread_id] += shared_memory[thread_id + i];
		}

		__syncthreads();
	}

	if (thread_id < 32)
	{
		warp_reduce(shared_memory, thread_id);
	}

	if (thread_id == 0)
	{
		partial_sums[blockIdx.x] = shared_memory[0] * scale;
	}
}


template<typename Dtype>
__global__ void gpu_nll_kernel(Dtype* data, Dtype* target, Dtype* partial_sums, uint64_t len) // same as gpu_sum_kernel but first multiplies data with target
{
	uint64_t i;
	int thread_per_block;
	int thread_id;
	int grid_size;
	Dtype thread_sum;
	Dtype* shared_memory;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	thread_id = threadIdx.x;

	thread_per_block = blockDim.x;
	shared_memory = (Dtype*)shared_mem_block;

	grid_size = thread_per_block * gridDim.x;

	thread_sum = 0;
	while (i < len)
	{
		thread_sum += (data[i] * target[i]);
		i += grid_size;
	}

	shared_memory[thread_id] = thread_sum;
	__syncthreads();

	for (i = thread_per_block / 2; i > 32; i >>= 1)
	{
		if (thread_id < i)
		{
			shared_memory[thread_id] += shared_memory[thread_id + i];
		}

		__syncthreads();
	}

	if (thread_id < 32)
	{
		warp_reduce(shared_memory, thread_id);
	}

	if (thread_id == 0)
	{
		partial_sums[blockIdx.x] = shared_memory[0];
	}
}

template<typename Dtype>
void gpu_nll(Dtype* input, Dtype* target, Dtype* loss, uint64_t len, uint64_t batches)
{
	int num_blocks;
	int thread_per_block = DEFA_REDUCTION_THREADS;
	static Dtype* partial_sums = nullptr;

	num_blocks = min(MAX_REDUCTION_BLOCKS, static_cast<int>(len) / thread_per_block + 1);

	if (!partial_sums)
	{
		cudaGetSymbolAddress((void**)&partial_sums, partial_sums_block);
	}

	gpu_nll_kernel<Dtype> << <num_blocks, thread_per_block, thread_per_block * sizeof(Dtype) >> > (input, target, partial_sums, len); // level 0 reduction

	thread_per_block = 64; // one warp ( + an extra warp to initialize shared mem)
	gpu_sum_kernel<Dtype> << <1, thread_per_block, thread_per_block * sizeof(Dtype) >> > (partial_sums, loss, num_blocks, static_cast<Dtype>(-1.0 / batches)); // level 1 reduction & scale

}


template<typename Dtype>
__global__ void gpu_scalar_mul_kernel(Dtype* A, Dtype* B, Dtype scalar, uint64_t len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		B[i] = A[i] * scalar;
	}
}

template<typename Dtype>
void gpu_scalar_mul(Dtype* A, Dtype* B, Dtype scalar, uint64_t len)
{
	int num_blocks;

	num_blocks = (static_cast<int>(len) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_scalar_mul_kernel<Dtype> << <num_blocks, DEFA_THREADS >> > (A, B, scalar, len);

}

template<typename Dtype>
__global__ void gpu_fill_kernel(Dtype* memory, uint64_t len, Dtype value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		memory[i] = value;
	}
}

template<typename Dtype>
void gpu_fill(Dtype* memory, uint64_t len, Dtype value)
{
	int num_blocks;

	num_blocks = (static_cast<int>(len) + DEFA_THREADS - 1) / DEFA_THREADS;


	gpu_fill_kernel<Dtype> << <num_blocks, DEFA_THREADS >> > (memory, len, value);
}


template<typename Dtype>
__global__ void gpu_fill_kernel(Dtype* memory, uint64_t len, Dtype* value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		memory[i] = *value;
	}
}


template<typename Dtype>
void gpu_fill(Dtype* memory, uint64_t len, Dtype* value)
{
	int num_blocks;

	num_blocks = (static_cast<int>(len) + DEFA_THREADS - 1) / DEFA_THREADS;


	gpu_fill_kernel<Dtype> << <num_blocks, DEFA_THREADS >> > (memory, len, value);
}


void gpu_scalar_mul(float alpha, float* A, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_C, uint64_t width_C)
{
	int max_height;
	int max_width;
	uint64_t len;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = DEFA_BLOCK_Y;
	dimBlock.x = DEFA_BLOCK_X;

	max_height = max((int)height_A, (int)height_C);
	max_width = max((int)width_A, (int)width_C);

	dimGrid.x = (int)ceil((float)max_width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)max_height / (float)dimBlock.y);

	len = height_A * width_A;
	gpu_scalar_mul<float>(A, A, alpha, len);

	gpu_axpy_kernel << <dimGrid, dimBlock >> > (A, C, (int)height_A, (int)width_A, (int)height_C, (int)width_C, max_height, max_width);
}


__global__ void gpu_sgd_step_kernel(float* weight_ptr, float* weight_grad_ptr, float* velocity_ptr, int64_t numels, float mo, float wd, float lr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numels)
	{
		weight_grad_ptr[i] = wd * weight_ptr[i] + weight_grad_ptr[i];
		velocity_ptr[i] = velocity_ptr[i] * mo + (1.0f - mo) * weight_grad_ptr[i];
		weight_ptr[i] = weight_ptr[i] - (velocity_ptr[i] * lr);
	}
}

void gpu_sgd_step(float* weight_ptr, float* weight_grad_ptr, float* velocity_ptr, int64_t numels, float mo, float wd, float lr)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_sgd_step_kernel << <num_blocks, DEFA_THREADS >> > (weight_ptr, weight_grad_ptr, velocity_ptr, numels, mo, wd, lr);
}


// The tensor function max(dim) generates a new tensor one dimension smaller than the original tensor by recoding the maximum values over dimension dim.
// For each location in the smaller tensor, cpu_max/gpu_max computes the correspoinding "first" location in the original larger tensor.
// In other words, given the offset of a coordinate a,b,d in the resualt tensor, and assuming dim=2, calculate the offset of a,b,0,d in the original tensor.
// If this offset is known, then from each location a,b,c we can scan along a,b,x,d from 0 to the size of dim to find the max.
// Once the max is found it is stored in location a,b,c
// The mapping between the destination location and the "first" source location is given by this formula:
// source_offset = (destination_offset - remainder) * ratio + remainder
// where:
// remainder = destination_offset % the stride of dimension dim
// ratio = stride of dim - 1 / stride of dim (which is the same as the size of dim) 
// Note: when dim = 0, then ratio = 1
template<typename Dtype>
__global__ void gpu_max_kernel(const Dtype* src, Dtype* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype max;
	Dtype val;
	uint64_t max_index;
	uint64_t rem;

	offset_dst = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset_dst >= numels)
	{
		return;
	}

	rem = offset_dst % stride;
	offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

	max = src[offset_src];
	max_index = 0;
	for (i = 1; i < dim_size; i++)  // iterate through required dimension
	{
		offset_src += stride;
		val = src[offset_src];
		if (val >= max)
		{
			max = val;
			max_index = i;
		}
	}

	indices[offset_dst] = max_index;
	dst[offset_dst] = max;

}

template<typename Dtype>
void gpu_max(const Dtype* src, Dtype* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_max_kernel << <num_blocks, DEFA_THREADS >> > (src, dst, indices, numels, ratio, dim_size, stride);
}


template<typename Dtype>
__global__ void gpu_max_backward_kernel(Dtype* dst, const Dtype* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t max_index;
	uint64_t rem;

	offset_src = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset_src >= numels)
	{
		return;
	}

	rem = offset_src % stride;
	offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

	max_index = indices[offset_src];
	offset_dst += stride * max_index;

	dst[offset_dst] += src[offset_src];
}

template<typename Dtype>
void gpu_max_backward(Dtype* dst, const Dtype* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_max_backward_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, indices, numels, ratio, dim_size, stride);
}



template<typename Dtype>
__global__ void gpu_powx_kernel(int N, const Dtype* A, Dtype x, Dtype* B)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		B[i] = static_cast<Dtype>(powf(A[i], x));
	}
}

template<typename Dtype>
void gpu_powx(uint64_t N, const Dtype* A, Dtype x, Dtype* B)
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_powx_kernel << <num_blocks, DEFA_THREADS >> > (static_cast<int>(N), A, x, B);
}

template<typename Dtype>
__global__ void gpu_exp_kernel(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	uint64_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numels)
	{
		dst[i] = expf(src[i]);
		//dst[i] = __expf(src[i]);
	}
}


template<typename Dtype>
void gpu_exp(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_exp_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, numels);
}


template<typename Dtype>
__global__ void gpu_log_kernel(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	uint64_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numels)
	{
		dst[i] = logf(src[i]);
		//dst[i] = __logf(src[i]);
	}
}

template<typename Dtype>
void gpu_log(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_log_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, numels);
}


template<typename Dtype>
__global__ void gpu_sig_kernel(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	uint64_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numels)
	{
		dst[i] = 1.0f / (1.0f + expf(-src[i]));
		//dst[i] = 1.0f / (1.0f + __expf(-src[i]));
	}
}

template<typename Dtype>
void gpu_sig(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_sig_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, numels);
}



template<typename Dtype>
__global__ void gpu_sig_backward_kernel(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels)
{
	uint64_t i;
	Dtype val;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numels)
	{
		val = middle[i];
		bottom[i] = top[i] * val * (1.0f - val);
	}
}


template<typename Dtype>
void gpu_sig_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_sig_backward_kernel << <num_blocks, DEFA_THREADS >> > (bottom, top, middle, numels);
}


template<typename Dtype>
__global__ void gpu_tanh_kernel(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	uint64_t i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numels)
	{
		dst[i] = tanhf(src[i]);
	}
}

template<typename Dtype>
void gpu_tanh(Dtype* dst, const Dtype* src, const uint64_t numels)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_tanh_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, numels);
}


template<typename Dtype>
__global__ void gpu_tanh_backward_kernel(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels)
{
	uint64_t i;
	Dtype val;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numels)
	{
		val = middle[i];
		bottom[i] = top[i] * (1.0f - val * val);
	}
}

template<typename Dtype>
void gpu_tanh_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_tanh_backward_kernel << <num_blocks, DEFA_THREADS >> > (bottom, top, middle, numels);
}

template<typename Dtype>
__global__ void gpu_sum_kernel(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype sum;
	uint64_t rem;

	offset_dst = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset_dst >= numels)
	{
		return;
	}

	rem = offset_dst % stride;
	offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

	sum = static_cast<Dtype>(0);
	for (i = 0; i < dim_size; i++)  // iterate through required dimension
	{
		sum += src[offset_src];
		offset_src += stride;
	}
	dst[offset_dst] = sum;
}

template<typename Dtype>
void gpu_sum(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_sum_kernel << <num_blocks, DEFA_THREADS >> > (src, dst, numels, ratio, dim_size, stride);

}


template<typename Dtype>
__global__ void gpu_sum_backward_kernel(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t rem;
	Dtype val;
	uint64_t i;

	offset_src = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset_src >= numels)
	{
		return;
	}

	rem = offset_src % stride;
	offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

	val = src[offset_src];
	for (i = 0; i < dim_size; i++)  // iterate through required dimension
	{
		dst[offset_dst] += val;
		offset_dst += stride;
	}

}


template<typename Dtype>
void gpu_sum_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	int num_blocks;

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_sum_backward_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, numels, ratio, dim_size, stride);
}


template<typename Dtype>
__global__ void gpu_add_kernel(int N, Dtype alpha, Dtype* A, Dtype* B)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		B[i] = alpha + A[i];
	}


}

template<typename Dtype>
void gpu_add(uint64_t N, Dtype alpha, Dtype* A, Dtype* B)
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_add_kernel << <num_blocks, DEFA_THREADS >> > ((int)N, alpha, A, B);

}


template<typename Dtype>
__global__ void gpu_axpy_kernel(int N, Dtype alpha, Dtype* X, Dtype* Y, Dtype* C)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = Y[i] + alpha * X[i];
	}

}

template<typename Dtype>
void gpu_axpy(uint64_t N, Dtype alpha, Dtype* X, Dtype* Y, Dtype* C)
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_axpy_kernel << <num_blocks, DEFA_THREADS >> > ((int)N, alpha, X, Y, C);

}


template<typename Dtype>
__global__ void gpu_axpby_kernel(int N, Dtype alpha, Dtype* X, Dtype beta, Dtype* Y, Dtype* C)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		C[i] = alpha * X[i] + beta * Y[i];
	}

}

template<typename Dtype>
void gpu_axpby(uint64_t N, Dtype alpha, Dtype* X, Dtype beta, Dtype* Y, Dtype* C)
{
	int num_blocks;

	num_blocks = (static_cast<int>(N) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_axpby_kernel << <num_blocks, DEFA_THREADS >> > ((int)N, alpha, X, beta, Y, C);
}


template<typename Dtype>
__global__ void gpu_relu_kernel(Dtype* dst, Dtype* src, unsigned int len)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		dst[i] = max(src[i], Dtype(0));
	}

}

template<typename Dtype>
void gpu_relu(Dtype* dst, Dtype* src, uint64_t len)
{
	int num_blocks;

	num_blocks = (static_cast<int>(len) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_relu_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, (unsigned int)len);

}


template<typename Dtype>
__global__ void gpu_relu_backward_kernel(Dtype* bottom, const Dtype* top, const Dtype* middle, unsigned int len)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		bottom[i] = (middle[i] > 0) ? top[i] : 0;
		/*
		if (middle[i] > 0)
		{
			bottom[i] = top[i];
		}
		else
		{
			bottom[i] = 0;
		}
		*/
	}
}

template<typename Dtype>
void gpu_relu_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t len)
{
	int num_blocks;

	num_blocks = (static_cast<int>(len) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_relu_backward_kernel << <num_blocks, DEFA_THREADS >> > (bottom, top, middle, (unsigned int)len);

}


template<typename Dtype>
__global__ void gpu_dropout_kernel(Dtype* dst, Dtype* src, unsigned int* mask, unsigned int threshold, Dtype scale, uint64_t len)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		dst[i] = src[i] * (mask[i] > threshold) * scale;
	}
}

template<typename Dtype>
void gpu_dropout(Dtype* dst, Dtype* src, unsigned int* mask, unsigned int threshold, Dtype scale, uint64_t len)
{
	int num_blocks;

	num_blocks = (static_cast<int>(len) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_dropout_kernel << <num_blocks, DEFA_THREADS >> > (dst, src, mask, threshold, scale, (unsigned int)len);
}


template<typename Dtype>
__global__ void gpu_transpose_kernel(Dtype* src, Dtype* dst, int dim_1, int dim_2,
	int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1,
	int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, int numels)
{
	unsigned int pre;
	unsigned int mid;
	unsigned int post;
	unsigned int remaining;

	unsigned int  idx;
	unsigned int  idx_transpose;

	int coord1;
	int coord2;
	float temp;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numels)
	{

		if (dim_1 == 0)
		{
			remaining = idx;
		}
		else
		{
			remaining = idx % stride_src_dim_1_minus_1;
		}
		coord1 = remaining / stride_src_dim_1;

		remaining = idx % stride_src_dim_2_minus_1;
		coord2 = remaining / stride_src_dim_2;


		if (dim_1 == 0)
		{
			pre = 0;
		}
		else
		{
			pre = idx / stride_src_dim_1_minus_1;
		}
		remaining = idx % stride_src_dim_1;
		mid = remaining / stride_src_dim_2_minus_1;

		remaining = idx % stride_src_dim_2;
		post = remaining;

		if (dim_1 == 0)
		{
			idx_transpose = mid * stride_trn_dim_2_minus_1 + post;
		}
		else
		{
			idx_transpose = pre * stride_trn_dim_1_minus_1 + mid * stride_trn_dim_2_minus_1 + post;
		}

		idx_transpose += coord2 * stride_trn_dim_1 + coord1 * stride_trn_dim_2;

		temp = src[idx];
		src[idx] = dst[idx_transpose];
		dst[idx_transpose] = temp;
	}

}

template<typename Dtype>
void gpu_transpose(Dtype* src, Dtype* dst, int dim_1, int dim_2,
	int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1,
	int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels)
{
	int num_blocks;

	if (dim_1 >= dim_2)
	{
		LTEN_ERR("Second dimension must be strictly greater than first dimension");
	}

	num_blocks = (static_cast<int>(numels) + DEFA_THREADS - 1) / DEFA_THREADS;

	gpu_transpose_kernel << <num_blocks, DEFA_THREADS >> > (src, dst, dim_1, dim_2, stride_src_dim_1, stride_src_dim_1_minus_1, stride_src_dim_2, stride_src_dim_2_minus_1, stride_trn_dim_1, stride_trn_dim_1_minus_1, stride_trn_dim_2, stride_trn_dim_2_minus_1, (unsigned int)numels);

}




template void gpu_sum<float>(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_sub<float>(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_mul<float>(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, float beta);
template void gpu_mul<float>(uint64_t N, float alpha, float* A, float* B, float beta, float* C);
template void gpu_mul<float>(uint64_t N, float alpha, float* A, float* B);
template void gpu_div<float>(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_sum<float>(float* data, float* sum, uint64_t len);
template void gpu_nll<float>(float* input, float* target, float* loss, uint64_t len, uint64_t batches);
template void gpu_scalar_mul<float>(float* A, float* B, float scalar, uint64_t len);
template void gpu_fill<float>(float* memory, size_t size, float value);
template void gpu_fill<float>(float* memory, uint64_t len, float* value);
template void gpu_max<float>(const float* src, float* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_max_backward<float>(float* dst, const float* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_powx<float>(uint64_t N, const float* A_ptr, float x, float* B_ptr);
template void gpu_exp<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_log<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_sig<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_sig_backward<float>(float* bottom, const float* top, const float* middle, const uint64_t numels);
template void gpu_tanh<float>(float* dst, const float* src, const uint64_t numels);
template void gpu_tanh_backward<float>(float* bottom, const float* top, const float* middle, const uint64_t numels);
template void gpu_sum<float>(const float* src, float* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_sum_backward<float>(float* dst, const float* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_div<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_add<float>(uint64_t N, float alpha, float* A_ptr, float* B_ptr);
template void gpu_mul<float>(uint64_t N, float* A, float* B, float* C);
template void gpu_axpy<float>(uint64_t N, float alpha, float* X, float* Y, float* C);
template void gpu_axpby<float>(uint64_t N, float alpha, float* X, float beta, float* Y, float* C);
template void gpu_relu<float>(float* dst, float* src, uint64_t len);
template void gpu_relu_backward<float>(float* bottom, const float* top, const float* middle, const uint64_t len);
template void gpu_dropout<float>(float* dst, float* src, unsigned int* mask, unsigned int threshold, float scale, uint64_t len);
template void gpu_transpose<float>(float* src, float* dst, int dim_1, int dim_2, int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1, int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels);

template void gpu_sum<int>(int* A, int* B, int* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_sub<int>(int* A, int* B, int* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_mul<int>(int* A, int* B, int* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, int beta);
template void gpu_mul<int>(uint64_t N, int alpha, int* A, int* B, int beta, int* C);
template void gpu_mul<int>(uint64_t N, int alpha, int* A, int* B);
template void gpu_div<int>(int* A, int* B, int* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_sum<int>(int* data, int* sum, uint64_t len);
template void gpu_nll<int>(int* input, int* target, int* loss, uint64_t len, uint64_t batches);
template void gpu_scalar_mul<int>(int* A, int* B, int scalar, uint64_t len);
template void gpu_fill<int>(int* memory, size_t size, int value);
template void gpu_fill<int>(int* memory, uint64_t len, int* value);
template void gpu_max<int>(const int* src, int* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_max_backward<int>(int* dst, const int* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_powx<int>(uint64_t N, const int* A_ptr, int x, int* B_ptr);
template void gpu_exp<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_log<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_sig<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_sig_backward<int>(int* bottom, const int* top, const int* middle, const uint64_t numels);
template void gpu_tanh<int>(int* dst, const int* src, const uint64_t numels);
template void gpu_tanh_backward<int>(int* bottom, const int* top, const int* middle, const uint64_t numels);
template void gpu_sum<int>(const int* src, int* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_sum_backward<int>(int* dst, const int* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_div<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_add<int>(uint64_t N, int alpha, int* A_ptr, int* B_ptr);
template void gpu_mul<int>(uint64_t N, int* A, int* B, int* C);
template void gpu_axpy<int>(uint64_t N, int alpha, int* X, int* Y, int* C);
template void gpu_axpby<int>(uint64_t N, int alpha, int* X, int beta, int* Y, int* C);
template void gpu_relu<int>(int* dst, int* src, uint64_t len);
template void gpu_relu_backward<int>(int* bottom, const int* top, const int* middle, const uint64_t len);
template void gpu_dropout<int>(int* dst, int* src, unsigned int* mask, unsigned int threshold, int scale, uint64_t len);
template void gpu_transpose<int>(int* src, int* dst, int dim_1, int dim_2, int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1, int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels);

template void gpu_sum<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_sub<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_mul<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint8_t beta);
template void gpu_mul<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A, uint8_t* B, uint8_t beta, uint8_t* C);
template void gpu_mul<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A, uint8_t* B);
template void gpu_div<uint8_t>(uint8_t* A, uint8_t* B, uint8_t* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);
template void gpu_sum<uint8_t>(uint8_t* data, uint8_t* sum, uint64_t len);
template void gpu_nll<uint8_t>(uint8_t* input, uint8_t* target, uint8_t* loss, uint64_t len, uint64_t batches);
template void gpu_scalar_mul<uint8_t>(uint8_t* A, uint8_t* B, uint8_t scalar, uint64_t len);
template void gpu_fill<uint8_t>(uint8_t* memory, size_t size, uint8_t value);
template void gpu_fill<uint8_t>(uint8_t* memory, uint64_t len, uint8_t* value);
template void gpu_max<uint8_t>(const uint8_t* src, uint8_t* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_max_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_powx<uint8_t>(uint64_t N, const uint8_t* A_ptr, uint8_t x, uint8_t* B_ptr);
template void gpu_exp<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);
template void gpu_log<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);
template void gpu_sig<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);
template void gpu_sig_backward<uint8_t>(uint8_t* bottom, const uint8_t* top, const uint8_t* middle, const uint64_t numels);
template void gpu_tanh<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels);
template void gpu_tanh_backward<uint8_t>(uint8_t* bottom, const uint8_t* top, const uint8_t* middle, const uint64_t numels);
template void gpu_sum<uint8_t>(const uint8_t* src, uint8_t* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_sum_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void gpu_div<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);
template void gpu_add<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A_ptr, uint8_t* B_ptr);
template void gpu_mul<uint8_t>(uint64_t N, uint8_t* A, uint8_t* B, uint8_t* C);
template void gpu_axpy<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* X, uint8_t* Y, uint8_t* C);
template void gpu_axpby<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* X, uint8_t beta, uint8_t* Y, uint8_t* C);
template void gpu_relu<uint8_t>(uint8_t* dst, uint8_t* src, uint64_t len);
template void gpu_relu_backward<uint8_t>(uint8_t* bottom, const uint8_t* top, const uint8_t* middle, const uint64_t len);
template void gpu_dropout<uint8_t>(uint8_t* dst, uint8_t* src, unsigned int* mask, unsigned int threshold, uint8_t scale, uint64_t len);
template void gpu_transpose<uint8_t>(uint8_t* src, uint8_t* dst, int dim_1, int dim_2, int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1, int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels);
