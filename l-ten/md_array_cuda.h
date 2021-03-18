#include "md_array.h"
#include "utils.h"


template<typename Dtype>
class CUDA_MultiDimArray :  public MultiDimArray<Dtype>
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
		Reset();
		Allocate(other.GetSizes(), other.GetNDims());
		GPUToGPUCopy(data_ptr_, other.GetDataPtr(), sizeof(Dtype) * numels_);
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


