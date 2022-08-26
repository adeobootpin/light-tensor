#include "md_array.h"
#include "utils.h"

/*
MIT License

Copyright (c) 2021 adeobootpin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


template<typename Dtype>
class CUDA_MultiDimArray : public MultiDimArray<Dtype>
{
public:
	CUDA_MultiDimArray() {}

	CUDA_MultiDimArray(const std::initializer_list<uint64_t>& dims, Dtype* buffer_to_use_ptr, bool own_memory = true)
	{
		Allocate(dims, buffer_to_use_ptr, own_memory);
	}

	CUDA_MultiDimArray(const uint64_t* dims_ptr, int ndims, Dtype* buffer_to_use_ptr, bool own_memory = true)
	{
		Allocate(dims_ptr, ndims, buffer_to_use_ptr, own_memory);
	}


	CUDA_MultiDimArray(const CUDA_MultiDimArray& other)
	{
		Allocate(other.GetSizes(), other.GetNDims());
		memcpy(data_ptr_, other.GetDataPtr(), sizeof(Dtype) * numels_);
	}


	~CUDA_MultiDimArray()
	{
		if (own_memory_)
		{
			FreeMemoryOnGPU(data_ptr_);
			data_ptr_ = nullptr;
		}
	}


#ifdef USE_MEMORYPOOL
	void* operator new(size_t size)
	{
		return lten::MISC_globals::singleton()->get_cpu_memorypool()->AllocateMemory(size);
	}


	void operator delete(void* memory)
	{
		return lten::MISC_globals::singleton()->get_cpu_memorypool()->FreeMemory(memory);
	}
#endif

	int Allocate(const uint64_t* dims_ptr, int ndims, Dtype* buffer_to_use_ptr = nullptr, bool own_memory = true)
	{
		int i;
		int64_t numels;
		int ret;

		if (ndims > MAX_DIMS)
		{
			LTEN_ERR("ndims > MAX_DIMS");
		}

		ndims_ = ndims;

		numels = 1;
		for (i = ndims - 1; i >= 0; i--)
		{
			strides_array_[i] = numels;
			dims_array_[i] = dims_ptr[i];
			numels *= dims_array_[i];
		}

		if (buffer_to_use_ptr)
		{
			if (data_ptr_ && own_memory_)
			{
				FreeMemoryOnGPU(data_ptr_);
			}
			data_ptr_ = buffer_to_use_ptr;
		}
		else
		{
			if (!own_memory_ || (numels_ != numels)) // (conservatively) avoid a memory allocation where possible
			{
				if (data_ptr_ && own_memory_)
				{
					FreeMemoryOnGPU(data_ptr_);
				}

				ret = AllocateMemoryOnGPU((void**)&data_ptr_, sizeof(Dtype) * numels, false);
				if (!data_ptr_ || ret)
				{
					return -1;
				}
			}
		}

		numels_ = numels;
		own_memory_ = own_memory;

		GetDevice(&device_index_);

		return 0;

	}

	int Allocate(const std::initializer_list<uint64_t>& dims, Dtype* buffer_to_use_ptr = nullptr, bool own_memory = true)
	{
		uint64_t dims_array[MAX_DIMS];
		int i;
		size_t ndims;
		int ret;

		ndims = dims.size();

		if (ndims > MAX_DIMS)
		{
			LTEN_ERR("ndims > MAX_DIMS");
		}

		i = 0;
		for (uint64_t dim : dims)
		{
			dims_array[i++] = dim;
		}

		ret = Allocate(dims_array, static_cast<int>(ndims), buffer_to_use_ptr, own_memory);

		return ret;
	}

	void ReleaseResources()
	{
		if (own_memory_)
		{
			FreeMemoryOnGPU(data_ptr_);
		}
		data_ptr_ = nullptr;
	}

	virtual CUDA_MultiDimArray& operator=(const CUDA_MultiDimArray& other)
	{
		Allocate(other.GetSizes(), other.GetNDims());
		GPUToGPUCopy(data_ptr_, other.GetDataPtr(), sizeof(Dtype) * numels_);
		return *this;
	}

	virtual CUDA_MultiDimArray& operator=(CUDA_MultiDimArray&& other)
	{
		if (own_memory_)
		{
			FreeMemoryOnGPU(data_ptr_);
		}
		Allocate(other.GetSizes(), other.GetNDims(), other.data_ptr_, other.own_memory_);
		other.own_memory_ = false;
		other.data_ptr_ = nullptr;
		return *this;
	}

	CUDA_MultiDimArray operator+(const CUDA_MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		CUDA_MultiDimArray result;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, true);

					gpu_sum(lhs_data, rhs_data, result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1]);
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);

					gpu_sum(lhs_data, rhs_data, result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1]);
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}


	CUDA_MultiDimArray operator-(const CUDA_MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		CUDA_MultiDimArray result;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, true);

					gpu_sub((float*)lhs_data, (float*)rhs_data, (float*)result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1]);
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);

					gpu_sub((float*)lhs_data, (float*)rhs_data, (float*)result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1]);
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}


	CUDA_MultiDimArray operator*(const CUDA_MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		CUDA_MultiDimArray result;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, true);

					gpu_mul((Dtype*)lhs_data, (Dtype*)rhs_data, (Dtype*)result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1], static_cast<Dtype>(0));
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);

					gpu_mul((Dtype*)lhs_data, (Dtype*)rhs_data, (Dtype*)result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1], static_cast<Dtype>(0));
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	CUDA_MultiDimArray operator/(const CUDA_MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		CUDA_MultiDimArray result;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, true);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, true);

					gpu_div((Dtype*)lhs_data, (Dtype*)rhs_data, (Dtype*)result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1]);
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);

					gpu_div((Dtype*)lhs_data, (Dtype*)rhs_data, (Dtype*)result_data, dims_array_[ndims_ - 2], dims_array_[ndims_ - 1], other_dims_array[ndims_ - 2], other_dims_array[ndims_ - 1]);
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	CUDA_MultiDimArray matmul(CUDA_MultiDimArray& other, POINTER_ARRAYS* pointer_array = nullptr)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		CUDA_MultiDimArray result;
		uint64_t M;
		uint64_t N;
		uint64_t K;
		float alpha;
		float beta;
		int lda;
		int ldb;
		int ldc;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 2)
		{
			original_ndims = ndims_;
			Reshape(2);
			other.Reshape(2);
		}

		other_dims_array = other.GetSizes();
		if (dims_array_[ndims_ - 1] != other_dims_array[ndims_ - 2]) // check matrix dimension compatibility
		{
			LTEN_ERR("MultiDimArrays must have compatiple dimensions");
		}


		broadcast_required = check_broadcast_required(other_dims_array, dims_result, true);

		result.Allocate(dims_result, ndims_, nullptr, false);


		cublasStatus_t status;
		cublasHandle_t hCuBlas;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(device_index_);

		alpha = 1.0f;
		beta = 0.0f;

		M = dims_result[ndims_ - 2];
		N = dims_result[ndims_ - 1];
		K = dims_array_[ndims_ - 1];

		lda = static_cast<int>(N);
		ldb = static_cast<int>(K);
		ldc = static_cast<int>(N);

		if (!broadcast_required)
		{
			int num_batches;

			num_batches = numels_ / (dims_array_[ndims_ - 1] * dims_array_[ndims_ - 2]);

			if (ndims_ < 3 || num_batches == 1)
			{
				Dtype* result_data = result.GetDataPtr();
				Dtype* lhs_data = GetDataPtr();
				Dtype* rhs_data = other.GetDataPtr();

				//status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
				//	(float*)rhs_data, lda, (float*)lhs_data, ldb, &beta, (float*)result_data, ldc);

				lda = K;
				status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
					(float*)rhs_data, lda, (float*)lhs_data, ldb, &beta, (float*)result_data, ldc);
			}
			else
			{
				Dtype* result_data = result.GetDataPtr();
				Dtype* lhs_data = GetDataPtr();
				Dtype* rhs_data = other.GetDataPtr();

				int stridea = N * K;
				int strideb = K * M;
				int stridec = N * M;

				status = cublasSgemmStridedBatched(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
					(float*)rhs_data, lda, stridea, (float*)lhs_data, ldb, strideb, &beta, (float*)result_data, ldc, stridec, num_batches);
			}

			cudaDeviceSynchronize();
		}
		else
		{
			Dtype* result_data = result.GetDataPtr();
			Dtype* lhs_data = GetDataPtr();
			Dtype* rhs_data = other.GetDataPtr();
			int num_batches;

			num_batches = result.GetNumels() / (dims_result[ndims_ - 1] * dims_result[ndims_ - 2]);

			if (!pointer_array)
			{
				POINTER_ARRAYS pa_gpu;
				POINTER_ARRAYS pa_cpu;

				AllocateMemoryOnGPU(&pa_gpu.buffer, 3 * sizeof(float*) * num_batches, false);
				pa_gpu.a_array = (void**)pa_gpu.buffer;
				pa_gpu.b_array = (void**)pa_gpu.buffer + num_batches;
				pa_gpu.c_array = (void**)pa_gpu.buffer + 2 * num_batches;

				pa_cpu.buffer = new float*[3 * num_batches];
				pa_cpu.a_array = (void**)pa_cpu.buffer;
				pa_cpu.b_array = (void**)pa_cpu.buffer + num_batches;
				pa_cpu.c_array = (void**)pa_cpu.buffer + 2 * num_batches;


				int index = 0;
				md_array_dim_iterator it(dims_result, ndims_ - 2);
				for (auto higher_indices : it)
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);

					pa_cpu.a_array[index] = (float*)lhs_data;
					pa_cpu.b_array[index] = (float*)rhs_data;
					pa_cpu.c_array[index] = (float*)result_data;
					index++;
				}

				assert(index == num_batches);
				CopyDataToGPU(pa_gpu.buffer, pa_cpu.buffer, 3 * sizeof(float*) * num_batches);

				pointer_array = &pa_gpu;

				//status = cublasSgemmBatched(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
				//	(float**)pointer_array->b_array, lda, (float**)pointer_array->a_array, ldb, &beta, (float**)pointer_array->c_array, ldc, num_batches);

				lda = K;
				status = cublasSgemmBatched(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
					(float**)pointer_array->b_array, lda, (float**)pointer_array->a_array, ldb, &beta, (float**)pointer_array->c_array, ldc, num_batches);

				delete pa_cpu.buffer;
				FreeMemoryOnGPU(pa_gpu.buffer);
			}
			else
			{
				status = cublasSgemmBatched(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
					(float**)pointer_array->b_array, lda, (float**)pointer_array->a_array, ldb, &beta, (float**)pointer_array->c_array, ldc, num_batches);
			}

		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}
	/*
	CUDA_MultiDimArray matmul(CUDA_MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		CUDA_MultiDimArray result;
		uint64_t M;
		uint64_t N;
		uint64_t K;
		float alpha;
		float beta;
		int lda;
		int ldb;
		int ldc;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();
		if (dims_array_[ndims_ - 1] != other_dims_array[ndims_ - 2]) // check matrix dimension compatibility
		{
			LTEN_ERR("MultiDimArrays must have compatiple dimensions");
		}


		broadcast_required = check_broadcast_required(other_dims_array, dims_result, true);

		result.Allocate(dims_result, ndims_, nullptr, false);


		md_array_dim_iterator it(dims_result, ndims_ - 2);

		M = dims_result[ndims_ - 2];
		N = dims_result[ndims_ - 1];
		K = dims_array_[ndims_ - 1];

		cublasStatus_t status;
		cublasHandle_t hCuBlas;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(device_index_);

		alpha = 1.0f;
		beta = 0.0f;
		lda = static_cast<int>(N);
		ldb = static_cast<int>(K);
		ldc = static_cast<int>(N);

		for (auto higher_indices : it)
		{
			Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);
			Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);
			Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);

			status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
				(float*)rhs_data, lda, (float*)lhs_data, ldb, &beta, (float*)result_data, ldc);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}
	*/

	CUDA_MultiDimArray transpose(int dim_1, int dim_2)
	{
		CUDA_MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		uint64_t temp;

		if (dim_1 == dim_2 || dim_1 >= ndims_ || dim_2 >= ndims_)
		{
			LTEN_ERR("Invalid args");
		}

		memcpy(dims_result, dims_array_, sizeof(uint64_t) * ndims_);
		temp = dims_result[dim_1];
		dims_result[dim_1] = dims_result[dim_2];
		dims_result[dim_2] = temp;

		result.Allocate(dims_result, ndims_, nullptr, false);


		uint64_t strides[MAX_DIMS];
		memcpy(strides, strides_array_, sizeof(uint64_t) * ndims_);

		temp = strides[dim_1];
		strides[dim_1] = strides[dim_2];
		strides[dim_2] = temp;
		
		gpu_transpose((float*)data_ptr_, (float*)result.GetDataPtr(), numels_, strides, result.GetStrides(), ndims_);

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	CUDA_MultiDimArray cat(const CUDA_MultiDimArray& other, int dim)
	{
		CUDA_MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		int i;
		const uint64_t* other_dims_array;
		const uint64_t* other_strides_array;
		const uint64_t* result_strides_array;
		Dtype* other_data_ptr;
		uint64_t other_numels;
		uint64_t dim_offset;
		Dtype* data;


		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		other_dims_array = other.GetSizes();
		for (i = 0; i < ndims_; i++)
		{
			if (i != dim)
			{
				if (dims_array_[i] != other_dims_array[i])
				{
					LTEN_ERR("MultiDimArrays must have compatiple dimensions");
				}
				dims_result[i] = dims_array_[i];
			}
			else
			{
				dims_result[i] = dims_array_[i] + other_dims_array[i];
			}
		}

		result.Allocate(dims_result, ndims_, nullptr, false);

		result_strides_array = result.GetStrides();

		data = result.GetDataPtr();
		other_data_ptr = other.GetDataPtr();

		result_strides_array[dim - 1];
		strides_array_[dim - 1];
		other_strides_array[dim - 1];
		other_strides_array[dim];

		dim_offset = dims_array_[dim];
		other_numels = other.GetNumels();
		other_strides_array = other.GetStrides();

		if (dim > 0)
		{
			gpu_cat(result.GetDataPtr(), data_ptr_, other.GetDataPtr(), result_strides_array[dim - 1], result_strides_array[dim], strides_array_[dim - 1], other_strides_array[dim - 1], other_strides_array[dim], dim_offset, numels_, other.GetNumels());
		}
		else
		{
			gpu_cat(result.GetDataPtr(), data_ptr_, other.GetDataPtr(), 0, 0, 0, 0, 0, dim_offset, numels_, other.GetNumels());
		}


		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);

	}

	CUDA_MultiDimArray index(const CUDA_MultiDimArray<int>& other)
	{
		CUDA_MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		const uint64_t* other_dims;
		int ndims;
		int other_ndims;
		int i;
		uint64_t u64i;
		Dtype* dst;
		Dtype* src;
		int* indices;
		uint64_t copy_len;
		uint64_t numels;


		copy_len = 1;
		for (i = 1; i < ndims_; i++)
		{
			copy_len *= dims_array_[i];
		}

		other_dims = other.GetSizes();
		other_ndims = other.GetNDims();
		ndims = other.GetNDims() + ndims_ - 1;

		memcpy(dims_result, other_dims, sizeof(uint64_t) * other_ndims);

		for (i = other_ndims; i < ndims; i++)
		{
			dims_result[i] = dims_array_[i - other_ndims + 1]; // add dims after 1st one to result dims array
		}

		result.Allocate(dims_result, ndims, nullptr, false);

		dst = result.GetDataPtr();
		src = data_ptr_;
		indices = other.GetDataPtr();
		numels = result.GetNumels();

		gpu_index(dst, src, indices, copy_len, numels);

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	CUDA_MultiDimArray repeat(const uint32_t* repeats, int nrepeats)
	{
		CUDA_MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		uint64_t dims_buffer[MAX_DIMS];
		uint64_t strides_buffer[MAX_DIMS];
		const uint64_t* strides_dst;
		uint64_t* dims_src; // need 'squeezed' src dims if ndims > ndims_ 
		uint64_t* strides_src;
		uint64_t numels;
		int i;
		int j;
		int ndims;

		ndims = nrepeats;

		if (ndims < ndims_)
		{
			LTEN_ERR("Number of repeat dimensions can not be less than number of tensor dimensions");
		}


		if (ndims > ndims_) // unsqueeze dims_array_
		{
			int len_diff = ndims - ndims_;
			dims_src = dims_buffer;
			for (i = 0; i < len_diff; i++)
			{
				dims_src[i] = 1;
			}

			memcpy(&dims_src[len_diff], dims_array_, sizeof(uint64_t) * ndims_);
		}
		else
		{
			dims_src = dims_array_;
		}

		for (i = 0; i < ndims; i++) // generate result dimesions
		{
			if (repeats[i] < 0)
			{
				LTEN_ERR("Repeat values must be greater than or equal to zero");
			}

			dims_result[i] = dims_src[i] * repeats[i];
		}

		ndims = std::max(ndims_, ndims);


		if (ndims != ndims_)
		{
			strides_src = strides_buffer; // generate src strides in case 'sqeezing' was required
			numels = 1;
			for (i = ndims - 1; i >= 0; i--)
			{
				strides_src[i] = numels;
				numels *= dims_src[i];
			}
		}
		else
		{
			strides_src = strides_array_; // just use strides_array_ if 'squeezing' was not performed
		}


		result.Allocate(dims_result, ndims, nullptr, false);
		numels = result.GetNumels();
		if (numels)
		{
			gpu_repeat(result.GetDataPtr(), data_ptr_, numels, result.GetStrides(), strides_src, dims_src, ndims);
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	CUDA_MultiDimArray repeat_interleave(const uint32_t* repeats, int nrepeats, int dim, uint32_t* scratch)
	{
		CUDA_MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		const uint64_t* strides_dst;
		int sum;
		Dtype* dst;
		uint32_t* cummulative_times;
		uint64_t numels;
		Dtype* src;
		int i;
		int j;
		int k;

		if (dim >= ndims_)
		{
			LTEN_ERR("Dimesion parameter is out of range");
		}

		if (nrepeats != dims_array_[dim])
		{
			LTEN_ERR("Sum of repeat values must equal size of dimension");
		}

		memcpy(dims_result, dims_array_, sizeof(uint64_t) * ndims_);
		sum = 0;
		for (i = 0; i < nrepeats; i++)
		{
			if (repeats[i] < 0)
			{
				LTEN_ERR("Repeat values must be greater than or equal to zero");
			}
			sum += repeats[i];
		}
		dims_result[dim] = sum;

		cummulative_times = scratch;
		if (!cummulative_times)
		{
			cummulative_times = new uint32_t[nrepeats + 1];
		}

		cummulative_times[0] = 0; // need this array for index look-up
		for (i = 1; i < nrepeats; i++)
		{
			cummulative_times[i] = cummulative_times[i - 1] + repeats[i - 1];
		}
		cummulative_times[nrepeats] = INT_MAX; // need stopper value so that linear scan (below) works

		result.Allocate(dims_result, ndims_, nullptr, false);

		dst = result.GetDataPtr();
		src = data_ptr_;
		strides_dst = result.GetStrides();

		numels = result.GetNumels();

		if (numels)
		{
			gpu_repeat_interleave(result.GetDataPtr(), data_ptr_, numels, result.GetStrides(), strides_array_, cummulative_times, ndims_, nrepeats, dim);
		}

		if (cummulative_times != scratch)
		{
			delete cummulative_times;
		}

		return CUDA_MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	using MultiDimArray<Dtype>::ndims_;
	using MultiDimArray<Dtype>::numels_;
	using MultiDimArray<Dtype>::own_memory_;
	using MultiDimArray<Dtype>::data_ptr_;
	using MultiDimArray<Dtype>::dims_array_;
	using MultiDimArray<Dtype>::strides_array_;

	using MultiDimArray<Dtype>::check_broadcast_required;
	using MultiDimArray<Dtype>::GetDataPtr;
	using MultiDimArray<Dtype>::Reset;
	using MultiDimArray<Dtype>::Reshape;

private:
	int device_index_;

};


