#include <cmath>
#include "tensorimpl.h"
#include "utils.h"

namespace lten {

	template<typename Dtype>
	int TensorImpl<Dtype>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory, TensorOps* options_ptr)
	{
		int ret;
		bool reallocate_md_array = false;

		if (options_ptr)
		{
			if (device_ != options_ptr->device_type ||
				device_index_ != options_ptr->device_index ||
				data_type_ != options_ptr->data_type)
			{
				reallocate_md_array = true;
			}

			device_ = options_ptr->device_type;
			device_index_ = options_ptr->device_index;
			data_type_ = options_ptr->data_type;
		}

		if (!md_array_base_ || reallocate_md_array)
		{
			delete md_array_base_;
			delete gradient_ptr_;

			if (device_ == GPU)
			{
#ifdef USE_CUDA
				md_array_base_ = new CUDA_MultiDimArray<Dtype>;
				if (options_ptr && options_ptr->alloc_gradient_buffer)
				{
					gradient_ptr_ = new CUDA_MultiDimArray<Dtype>;
				}
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				md_array_base_ = new MultiDimArray<Dtype>;
				if (options_ptr && options_ptr->alloc_gradient_buffer)
				{
					gradient_ptr_ = new MultiDimArray<Dtype>;
				}
			}
		}

		if (gradient_ptr_)
		{
			ret = gradient_ptr_->Allocate(dims, nullptr, true);
			if (ret)
			{
				return ret;
			}

			if (device_ == CPU)
			{
				::FillBuffer<Dtype>(gradient_ptr_->GetDataPtr(), gradient_ptr_->GetNumels(), 0);
			}
			else
			{
				if (GPU == get_device())
				{
#ifdef USE_CUDA
					ZeroMemoryOnGPU(gradient_ptr_->GetDataPtr(), sizeof(Dtype) * gradient_ptr_->GetNumels()); // zero gradients (backward logic expects gradients to be initialized to 0)
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}
		}

		return md_array_base_->Allocate(dims, (Dtype*)data_ptr, own_memory);
	}


	template<typename Dtype>
	int TensorImpl<Dtype>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr)
	{
		int ret;
		bool reallocate_md_arrays = false;

		if (options_ptr)
		{
			if (device_ != options_ptr->device_type ||
				device_index_ != options_ptr->device_index ||
				data_type_ != options_ptr->data_type)
			{
				reallocate_md_arrays = true;
			}

			device_ = options_ptr->device_type;
			device_index_ = options_ptr->device_index;
			data_type_ = options_ptr->data_type;
		}

		if (!md_array_base_ || reallocate_md_arrays)
		{
			delete md_array_base_;
			delete gradient_ptr_;

			if (device_ == GPU)
			{
#ifdef USE_CUDA
				md_array_base_ = new CUDA_MultiDimArray<Dtype>;
				gradient_ptr_ = new CUDA_MultiDimArray<Dtype>;
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				md_array_base_ = new MultiDimArray<Dtype>;
				gradient_ptr_ = new MultiDimArray<Dtype>;
			}
		}

		ret = gradient_ptr_->Allocate(dims, (Dtype*)gradient_ptr, own_gradient_memory);
		if (ret)
		{
			return ret;
		}

		return md_array_base_->Allocate(dims, (Dtype*)data_ptr, own_data_memory);
	}


	template<typename Dtype>
	int TensorImpl<Dtype>::allocate_from_buffer(const uint64_t* dims_ptr, int ndims, void* data_ptr, bool own_memory, TensorOps* options_ptr)
	{
		int ret;
		bool reallocate_md_array = false;

		if (options_ptr)
		{
			if (device_ != options_ptr->device_type ||
				device_index_ != options_ptr->device_index ||
				data_type_ != options_ptr->data_type)
			{
				reallocate_md_array = true;
			}

			device_ = options_ptr->device_type;
			device_index_ = options_ptr->device_index;
			data_type_ = options_ptr->data_type;
		}

		if (!md_array_base_ || reallocate_md_array)
		{
			delete md_array_base_;
			delete gradient_ptr_;

			if (device_ == GPU)
			{
#ifdef USE_CUDA
				md_array_base_ = new CUDA_MultiDimArray<Dtype>;
				if (options_ptr && options_ptr->alloc_gradient_buffer)
				{
					gradient_ptr_ = new CUDA_MultiDimArray<Dtype>;
					ZeroMemoryOnGPU(gradient_ptr_->GetDataPtr(), sizeof(Dtype) * gradient_ptr_->GetNumels()); // zero gradients (backward logic expects gradients to be initialized to 0)
				}
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				md_array_base_ = new MultiDimArray<Dtype>;
				if (options_ptr && options_ptr->alloc_gradient_buffer)
				{
					gradient_ptr_ = new MultiDimArray<Dtype>;
					memset(gradient_ptr_->GetDataPtr(), 0, sizeof(Dtype) * gradient_ptr_->GetNumels());
				}
			}
		}

		if (gradient_ptr_)
		{
			ret = gradient_ptr_->Allocate(dims_ptr, ndims, nullptr, true);
			if (ret)
			{
				return ret;
			}

			if (device_ == CPU)
			{
				::FillBuffer<Dtype>(gradient_ptr_->GetDataPtr(), gradient_ptr_->GetNumels(), 0);
			}
			else
			{
				if (GPU == get_device())
				{
#ifdef USE_CUDA
					ZeroMemoryOnGPU(gradient_ptr_->GetDataPtr(), sizeof(Dtype) * gradient_ptr_->GetNumels());
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}
		}

		return md_array_base_->Allocate(dims_ptr, ndims, (Dtype*)data_ptr, own_memory);
	}


	template<typename Dtype>
	int TensorImpl<Dtype>::allocate(const std::initializer_list<uint64_t>& dims, TensorOps* options_ptr)
	{
		return allocate_from_buffer(dims, nullptr, true, options_ptr);
	}

	template<typename Dtype>
	int TensorImpl<Dtype>::allocate(const uint64_t* dims_ptr, int ndims, TensorOps* options_ptr)
	{
		return allocate_from_buffer(dims_ptr, ndims, nullptr, true, options_ptr);
	}


	template<typename Dtype>
	void TensorImpl<Dtype>::release_resources()
	{
		int i;

		delete md_array_base_;
		delete gradient_ptr_;

		if (own_misc_ptr1_)
		{
			if (CPU == device_)
			{
				delete misc_ptr1_;
			}
			else
			{
				if (GPU == device_)
				{
#ifdef USE_CUDA
					FreeMemoryOnGPU(misc_ptr1_);
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}
		}

		if (own_misc_ptr2_)
		{
			if (CPU == device_)
			{
				delete misc_ptr2_;
			}
			else
			{
				if (GPU == device_)
				{
#ifdef USE_CUDA
					FreeMemoryOnGPU(misc_ptr2_);
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}
		}

		gradient_ptr_ = nullptr;
		misc_ptr1_ = nullptr;
		misc_ptr2_ = nullptr;

		for (i = 0; i < num_children_; i++)
		{
			delete children_lock_[i];
		}

		if (view_src_)
		{
			int ref_count;

			ref_count = view_src_->release();
			if (ref_count == 0)
			{
				view_src_->release_resources();
				delete view_src_;
			}
		}

		reset();
	}

	template<typename Dtype>
	void TensorImpl<Dtype>::to(TensorImpl<Dtype>& operand1, device target_device, int target_device_index)
	{
		TensorOps options;

		options.data_type = operand1.get_data_type();
		options.device_type = target_device;
		options.device_index = target_device_index;
		set_autograd(operand1.autograd_on());

		LTEN_ERR_CHECK(allocate(operand1.get_sizes(), operand1.get_ndims(), &options));

		if (operand1.gradient_ptr_)
		{
			if (CPU == target_device)
			{
				gradient_ptr_ = new MultiDimArray<Dtype>;
				if (!gradient_ptr_)
				{
					std::terminate(); // no hope, bail
				}
				LTEN_ERR_CHECK(gradient_ptr_->Allocate(get_sizes(), get_ndims()));
			}
			else
			{
				if (GPU == target_device)
				{
#ifdef USE_CUDA
					gradient_ptr_ = new CUDA_MultiDimArray<Dtype>;
					if (!gradient_ptr_)
					{
						std::terminate(); // no hope, bail
					}
					LTEN_ERR_CHECK(gradient_ptr_->Allocate(get_sizes(), get_ndims()));
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
			}
		}



		if (GPU == target_device)
		{
#ifdef USE_CUDA
			switch (operand1.get_device())
			{
			case GPU:
				GPUToGPUCopy(get_data_ptr(), operand1.get_data_ptr(), sizeof(Dtype) * get_numels());
				if (gradient_ptr_)
				{
					GPUToGPUCopy(gradient_ptr_->GetDataPtr(), operand1.gradient_ptr_->GetDataPtr(), sizeof(Dtype) * get_numels());
				}
				break;
			case CPU:
				CopyDataToGPU(get_data_ptr(), operand1.get_data_ptr(), sizeof(Dtype) * get_numels());
				if (gradient_ptr_)
				{
					CopyDataToGPU(gradient_ptr_->GetDataPtr(), operand1.gradient_ptr_->GetDataPtr(), sizeof(Dtype) * get_numels());
				}
				break;
			default:
				LTEN_ERR("Invalid device type");
				break;
			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			return;
		}

		if (CPU == target_device)
		{
			switch (operand1.get_device())
			{
			case GPU:
#ifdef USE_CUDA
				CopyDataFromGPU(get_data_ptr(), operand1.get_data_ptr(), sizeof(Dtype) * get_numels());
				if (gradient_ptr_)
				{
					CopyDataFromGPU(gradient_ptr_->GetDataPtr(), operand1.gradient_ptr_->GetDataPtr(), sizeof(Dtype) * get_numels());
				}
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				break;
			case CPU:
				memcpy(get_data_ptr(), operand1.get_data_ptr(), sizeof(Dtype) * get_numels());
				if (gradient_ptr_)
				{
					memcpy(gradient_ptr_->GetDataPtr(), operand1.gradient_ptr_->GetDataPtr(), sizeof(Dtype) * get_numels());
				}
				break;
			default:
				LTEN_ERR("Invalid device type");
				break;
			}

			return;
		}
	}


	template<typename Dtype>
	void TensorImpl<Dtype>::matmul(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2)
	{
		int ndims;
		int i;
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		const uint64_t* result_sizes_ptr;
		const uint64_t* op1_sizes_ptr;
		const uint64_t* op2_sizes_ptr;
		bool broadcast_required;
		TensorOps options;

		ndims = operand1.get_ndims();

		if (ndims != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}


		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_sizes_ptr = operand1.get_sizes();
		op2_sizes_ptr = operand2.get_sizes();

		if (operand1.autograd_on() || operand2.autograd_on())
		{
			broadcast_required = false;
			for (i = 0; i < ndims - 2; i++)
			{
				if (op1_sizes_ptr[i] != op2_sizes_ptr[i])
				{
					broadcast_required = true;
					break;
				}
			}
		}

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();

		if (CPU == options.device_type)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array).matmul(*op2_md_array);

			result_sizes_ptr = result.GetSizes();
			ndims = result.GetNDims();

			LTEN_ERR_CHECK(allocate_from_buffer(result_sizes_ptr, ndims, result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))).matmul((*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array))));

				result_sizes_ptr = result.GetSizes();
				ndims = result.GetNDims();

				LTEN_ERR_CHECK(allocate_from_buffer(result_sizes_ptr, ndims, result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on() || operand2.autograd_on())
		{
			add_child(operand1);
			add_child(operand2);
			grad_fn_ = ::matmul_backward;
			set_autograd(true);
		}
	}

	template<typename Dtype>
	void TensorImpl<Dtype>::exp(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		Dtype* data_ptr;
		const Dtype* operand1_data_ptr;
		uint64_t i;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		TensorOps options;


		ndims = operand1.get_ndims();

		sizes_operand1_ptr = operand1.get_sizes();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate(sizes_operand1_ptr, ndims, &options));

		len = operand1.get_numels();
		operand1_data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());
		data_ptr = md_array_base_->GetDataPtr();

		if (CPU == options.device_type)
		{
			for (i = 0; i < len; i++)
			{
				data_ptr[i] = static_cast<Dtype>(expf(static_cast<float>(operand1_data_ptr[i])));
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_exp(data_ptr, operand1_data_ptr, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::exp_backward;
			set_autograd(true);
		}
	}


	template<typename Dtype>
	void TensorImpl<Dtype>::scalar_mul(TensorImpl<Dtype>& operand1, double scalar)
	{
		uint64_t len;
		Dtype* data_ptr;
		Dtype* operand1_data_ptr;
		uint64_t i;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		TensorOps options;

		ndims = operand1.get_ndims();

		sizes_operand1_ptr = operand1.get_sizes();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate(sizes_operand1_ptr, ndims, &options));

		len = operand1.get_numels();
		operand1_data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());
		data_ptr = md_array_base_->GetDataPtr();

		if (options.device_type == CPU)
		{
			for (i = 0; i < len; i++)
			{
				data_ptr[i] = static_cast<Dtype>(operand1_data_ptr[i] * scalar);
			}
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				gpu_scalar_mul(operand1_data_ptr, data_ptr, (Dtype)scalar, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			misc2_ = static_cast<Dtype>(scalar);
			add_child(operand1);
			grad_fn_ = ::scalar_mul_backward;
			set_autograd(true);
		}

	}



	template<typename Dtype>
	void TensorImpl<Dtype>::log(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		Dtype* data_ptr;
		const Dtype* operand1_data_ptr;
		uint64_t i;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		TensorOps options;


		ndims = operand1.get_ndims();

		sizes_operand1_ptr = operand1.get_sizes();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate(sizes_operand1_ptr, ndims, &options));

		len = operand1.get_numels();
		operand1_data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());
		data_ptr = md_array_base_->GetDataPtr();

		if (CPU == options.device_type)
		{
			for (i = 0; i < len; i++)
			{
				data_ptr[i] = static_cast<Dtype>(logf(static_cast<float>(operand1_data_ptr[i])));
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_log(data_ptr, operand1_data_ptr, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::log_backward;
			set_autograd(true);
		}
	}


	template<typename Dtype>
	void TensorImpl<Dtype>::sig(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		Dtype* data_ptr;
		const Dtype* operand1_data_ptr;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		TensorOps options;


		ndims = operand1.get_ndims();

		sizes_operand1_ptr = operand1.get_sizes();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate(sizes_operand1_ptr, ndims, &options));

		len = operand1.get_numels();
		operand1_data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());
		data_ptr = md_array_base_->GetDataPtr();

		if (CPU == options.device_type)
		{
			cpu_sig(len, operand1_data_ptr, data_ptr);
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_sig(data_ptr, operand1_data_ptr, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::sig_backward;
			set_autograd(true);
		}

	}

	template<typename Dtype>
	void TensorImpl<Dtype>::tanh(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		Dtype* data_ptr;
		const Dtype* operand1_data_ptr;
		uint64_t i;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		TensorOps options;


		ndims = operand1.get_ndims();

		sizes_operand1_ptr = operand1.get_sizes();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate(sizes_operand1_ptr, ndims, &options));

		len = operand1.get_numels();
		operand1_data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());
		data_ptr = md_array_base_->GetDataPtr();

		if (CPU == options.device_type)
		{
			for (i = 0; i < len; i++)
			{
				data_ptr[i] = static_cast<Dtype>(tanhf(static_cast<float>(operand1_data_ptr[i])));
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_tanh(data_ptr, operand1_data_ptr, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}



		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::tanh_backward;
			set_autograd(true);
		}

	}



	template<typename Dtype>
	void TensorImpl<Dtype>::max(TensorImpl<Dtype>& operand1, int dim)
	{
		uint64_t len;
		uint64_t dims[MAX_DIMS];
		MultiDimArray<Dtype>* md_array_ptr;
		MultiDimArray<Dtype>* op1_md_array;
		uint64_t i;
		int index;
		const uint64_t* src_sizes;
		const uint64_t* src_strides;
		int ndims;
		TensorOps options;
		uint64_t dim_size;
		uint64_t ratio;


		ndims = operand1.get_ndims();
		if (dim >= ndims)
		{
			LTEN_ERR("Invalid index");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();

		src_strides = op1_md_array->GetStrides();
		src_sizes = op1_md_array->GetSizes();

		len = 0;
		index = 0;
		for (i = 0; i < ndims; i++)
		{
			if (i != dim)
			{
				dims[index] = src_sizes[i];
				len *= dims[index];
				index++;
			}
		}

		LTEN_ERR_CHECK(allocate(dims, ndims - 1, &options));

		md_array_ptr = get_mdarray();

		dim_size = src_sizes[dim];
		src_sizes = operand1.get_sizes();

		if (dim > 0)
		{
			ratio = src_strides[dim - 1] / src_strides[dim];
		}
		else
		{
			ratio = 1;
		}

		if (CPU == options.device_type)
		{
			MultiDimArray<uint64_t> indices;

			indices.Allocate(dims, ndims - 1); // md_array to hold max indices

			cpu_max(op1_md_array->GetDataPtr(), md_array_ptr->GetDataPtr(), indices.GetDataPtr(), get_numels(), ratio, dim_size, src_strides[dim]);

			indices.Reshape(ndims);
			assert(misc_ptr1_ == nullptr);
			misc_ptr1_ = indices.GetDataPtr();
			indices.SetMemoryOwnership(false); // prevent md_array memory from getting freed
			own_misc_ptr1_ = true; // take ownership of memory
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<uint64_t> indices;

				indices.Allocate(dims, ndims - 1); // md_array to hold max indices

				gpu_max(op1_md_array->GetDataPtr(), md_array_ptr->GetDataPtr(), indices.GetDataPtr(), get_numels(), ratio, dim_size, src_strides[dim]);

				indices.Reshape(ndims);
				assert(misc_ptr1_ == nullptr);
				misc_ptr1_ = indices.GetDataPtr();
				indices.SetMemoryOwnership(false); // prevent md_array memory from getting freed
				own_misc_ptr1_ = true; // take ownership of memory
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		md_array_ptr->Reshape(ndims);

		if (operand1.autograd_on())
		{
			misc1_ = dim; // save this for back prop
			add_child(operand1);
			grad_fn_ = ::max_backward1;
			set_autograd(true);
		}
	}

	template<typename Dtype>
	void TensorImpl<Dtype>::max(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		const Dtype* data_ptr;
		uint64_t dims[MAX_DIMS];
		Dtype* value_ptr;
		const uint64_t* sizes_ptr;
		uint64_t i;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		Dtype max;
		uint64_t max_count;
		TensorOps options;

		ndims = operand1.get_ndims();
		for (i = 0; i < ndims; i++)
		{
			dims[i] = 1;
		}

		sizes_ptr = operand1.get_sizes();

		data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());

		len = operand1.get_numels();
		max_count = 1;
		max = data_ptr[0];
		for (i = 1; i < len; i++)
		{
			if (data_ptr[i] > max)
			{
				max_count = 1;
				max = data_ptr[i];
			}
			else
			{
				if (data_ptr[i] == max)
				{
					max_count++;
				}
			}
		}

		value_ptr = new Dtype;
		*value_ptr = max;

		options.data_type = operand1.get_data_type();
		LTEN_ERR_CHECK(allocate_from_buffer(dims, ndims, value_ptr, true, &options));

		sizes_operand1_ptr = operand1.get_sizes();

		if (operand1.autograd_on())
		{
			misc1_ = max_count;
			misc2_ = max;

			add_child(operand1);
			grad_fn_ = ::max_backward2;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::sum(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		Dtype* data_ptr;
		uint64_t dims[MAX_DIMS];
		Dtype* value_ptr;
		const uint64_t* sizes_ptr;
		uint64_t i;
		int ndims;
		Dtype sum;
		TensorOps options;

		ndims = operand1.get_ndims();
		for (i = 0; i < ndims; i++)
		{
			dims[i] = 1;
		}

		sizes_ptr = operand1.get_sizes();

		data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		len = operand1.get_numels();

		if (options.device_type == CPU)
		{
			sum = 0;
			for (i = 0; i < len; i++)
			{
				sum += data_ptr[i];
			}

			value_ptr = new Dtype;
			*value_ptr = sum;

			LTEN_ERR_CHECK(allocate_from_buffer(dims, ndims, value_ptr, true, &options));
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				AllocateMemoryOnGPU((void**)&value_ptr, sizeof(Dtype), false);
				gpu_sum(data_ptr, value_ptr, len);

				LTEN_ERR_CHECK(allocate_from_buffer(dims, ndims, value_ptr, true, &options));
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::sum_backward2;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::sum(TensorImpl<Dtype>& operand1, int dim)
	{
		uint64_t len;
		uint64_t dims[MAX_DIMS];
		MultiDimArray<Dtype>* md_array_ptr;
		MultiDimArray<Dtype>* op1_md_array;
		uint64_t i;
		int index;
		const uint64_t* src_sizes;
		const uint64_t* src_strides;
		int ndims;
		TensorOps options;
		uint64_t dim_size;
		uint64_t ratio;

		ndims = operand1.get_ndims();
		if (dim >= ndims)
		{
			LTEN_ERR("Invalid index");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();

		src_strides = op1_md_array->GetStrides();
		src_sizes = operand1.get_sizes();

		len = 0;
		index = 0;
		for (i = 0; i < ndims; i++)
		{
			if (i != dim)
			{
				dims[index] = src_sizes[i];
				len *= dims[index];
				index++;
			}
		}

		LTEN_ERR_CHECK(allocate(dims, ndims - 1, &options));

		md_array_ptr = get_mdarray();
		dim_size = src_sizes[dim];

		if (dim > 0)
		{
			ratio = src_strides[dim - 1] / src_strides[dim];
		}
		else
		{
			ratio = 1;
		}


		if (CPU == options.device_type)
		{
			cpu_sum(op1_md_array->GetDataPtr(), md_array_ptr->GetDataPtr(), get_numels(), ratio, dim_size, src_strides[dim]);
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_sum(op1_md_array->GetDataPtr(), md_array_ptr->GetDataPtr(), get_numels(), ratio, dim_size, src_strides[dim]);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		md_array_ptr->Reshape(ndims);

		if (operand1.autograd_on())
		{
			misc1_ = dim; // save this for back prop

			add_child(operand1);
			grad_fn_ = ::sum_backward1;
			set_autograd(true);
		}

	}



	template<typename Dtype>
	void TensorImpl<Dtype>::add(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2)
	{
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		TensorOps options;



		if (operand1.get_ndims() != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array) + (*op2_md_array);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))) + (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array)));

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::add_backward;
			set_autograd(true);
		}

		if (operand2.autograd_on())
		{
			add_child(operand2);
			grad_fn_ = ::add_backward;
			set_autograd(true);
		}

	}



	template<typename Dtype>
	void TensorImpl<Dtype>::sub(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2)
	{
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		TensorOps options;



		if (operand1.get_ndims() != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array) - (*op2_md_array);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))) - (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array)));

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::sub_backward;
			set_autograd(true);
		}

		if (operand2.autograd_on())
		{
			add_child(operand2);
			grad_fn_ = ::sub_backward;
			set_autograd(true);
		}

	}

	// element-wize multiplication
	template<typename Dtype>
	void TensorImpl<Dtype>::mul(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2)
	{
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		TensorOps options;



		if (operand1.get_ndims() != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array) * (*op2_md_array);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))) * (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array)));

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on() || operand2.autograd_on())
		{
			add_child(operand1);
			add_child(operand2);
			grad_fn_ = ::mul_backward;
			set_autograd(true);
		}
	}


	// element-wise division
	template<typename Dtype>
	void TensorImpl<Dtype>::div(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2)
	{
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		TensorOps options;



		if (operand1.get_ndims() != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array) / (*op2_md_array);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))) / (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array)));

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on() || operand2.autograd_on())
		{
			add_child(operand1);
			add_child(operand2);
			grad_fn_ = ::div_backward;
			set_autograd(true);
		}

	}

	template<typename Dtype>
	void TensorImpl<Dtype>::sqrt(TensorImpl<Dtype>& operand1)
	{
		uint64_t len;
		Dtype* data_ptr;
		const Dtype* operand1_data_ptr;
		uint64_t i;
		const uint64_t* sizes_operand1_ptr;
		int ndims;
		TensorOps options;


		ndims = operand1.get_ndims();

		sizes_operand1_ptr = operand1.get_sizes();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate(sizes_operand1_ptr, ndims, &options));

		len = operand1.get_numels();
		operand1_data_ptr = static_cast<Dtype*>(operand1.get_data_ptr());
		data_ptr = md_array_base_->GetDataPtr();

		if (CPU == options.device_type)
		{
			for (i = 0; i < len; i++)
			{
				data_ptr[i] = static_cast<Dtype>(sqrtf(static_cast<float>(operand1_data_ptr[i])));
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_sqrt(data_ptr, operand1_data_ptr, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}



		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::sqrt_backward;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::mean(TensorImpl<Dtype>& operand1)
	{
		uint64_t dims[MAX_DIMS];
		TensorOps options;


		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		dims[0] = 1;
		LTEN_ERR_CHECK(allocate(dims, 1, &options));

		if (CPU == options.device_type)
		{
			assert(0); // TODO implement cpu version
			LTEN_ERR("Not yet implemented: mean for cpu");
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA			
				gpu_mean(static_cast<Dtype*>(get_data_ptr()), static_cast<Dtype*>(operand1.get_data_ptr()), operand1.get_numels());
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			misc1_ = 0; // naxes is zero for gloabl mean
			add_child(operand1);
			grad_fn_ = ::mean_backward;
			set_autograd(true);
		}
	}

	template<typename Dtype>
	void TensorImpl<Dtype>::mean(TensorImpl<Dtype>& operand1, const uint32_t* axes, int naxes)
	{
		int i;
		TensorOps options;
		uint64_t dims_dst[MAX_DIMS];
		int ndims_src;
		int ndims_dst;
		uint64_t bitmask;
		const uint64_t* dims_src;

		ndims_src = operand1.get_ndims();

		for (i = 0; i < naxes; i++)
		{
			if (axes[i] >= static_cast<uint32_t>(ndims_src))
			{
				LTEN_ERR("Invalid index");
			}
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		bitmask = 0;
		for (i = 0; i < naxes; i++)
		{
			bitmask |= (1 << axes[i]);
		}

		dims_src = operand1.get_sizes();
		ndims_dst = 0;

		for (i = 0; i < ndims_src; i++)
		{
			if (!(bitmask & (1 << i)))
			{
				dims_dst[ndims_dst++] = dims_src[i];
			}
		}

		LTEN_ERR_CHECK(allocate(dims_dst, ndims_dst, &options));

		if (CPU == options.device_type)
		{
			cpu_mean((Dtype*)get_data_ptr(), (Dtype*)operand1.get_data_ptr(), get_numels(), get_strides(), operand1.get_strides(), ndims_dst, ndims_src, dims_src, axes);
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_mean((Dtype*)get_data_ptr(), (Dtype*)operand1.get_data_ptr(), get_numels(), get_strides(), operand1.get_strides(), ndims_dst, ndims_src, operand1.get_sizes(), axes);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::mean_backward;
			set_autograd(true);
		}
	}


	template<typename Dtype>
	void TensorImpl<Dtype>::var(TensorImpl& operand1, const uint32_t* axes, int naxes)
	{
		int i;
		TensorOps options;
		uint64_t dims_dst[MAX_DIMS];
		uint64_t dims_tmp[MAX_DIMS];
		int ndims_src;
		int ndims_dst;


		ndims_src = operand1.get_ndims();

		for (i = 0; i < naxes; i++)
		{
			if (axes[i] >= static_cast<uint32_t>(ndims_src))
			{
				LTEN_ERR("Invalid index");
			}
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();


		memcpy(dims_tmp, operand1.get_sizes(), sizeof(uint64_t) * ndims_src);

		for (i = 0; i < naxes; i++)
		{
			dims_tmp[axes[i]] = 0; // set to an invalid value
		}

		ndims_dst = 0;
		for (i = 0; i < ndims_src; i++)
		{
			if (dims_tmp[i])
			{
				dims_dst[ndims_dst] = dims_tmp[i];
				ndims_dst++;
			}
		}

		LTEN_ERR_CHECK(allocate(dims_dst, ndims_dst, &options));

		if (CPU == options.device_type)
		{
			//cpu_var((Dtype*)get_data_ptr(), (Dtype*)operand1.get_data_ptr(), get_numels(), get_strides(), operand1.get_strides(), ndims_dst, ndims_src, operand1.get_sizes(), axes);
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_var((Dtype*)get_data_ptr(), (Dtype*)operand1.get_data_ptr(), get_numels(), get_strides(), operand1.get_strides(), ndims_dst, ndims_src, operand1.get_sizes(), axes);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::var_backward;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::std(TensorImpl<Dtype>& operand1, int dim)
	{
		uint64_t len;
		uint64_t dims[MAX_DIMS];
		MultiDimArray<Dtype>* md_array_ptr;
		MultiDimArray<Dtype>* op1_md_array;
		uint64_t i;
		int index;
		const uint64_t* src_sizes;
		const uint64_t* src_strides;
		int ndims;
		TensorOps options;
		uint64_t dim_size;
		uint64_t ratio;

		ndims = operand1.get_ndims();
		if (dim >= ndims)
		{
			LTEN_ERR("Invalid index");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();

		src_strides = op1_md_array->GetStrides();
		src_sizes = operand1.get_sizes();

		len = 0;
		index = 0;
		for (i = 0; i < ndims; i++)
		{
			if (i != dim)
			{
				dims[index] = src_sizes[i];
				len *= dims[index];
				index++;
			}
		}

		LTEN_ERR_CHECK(allocate(dims, ndims - 1, &options));

		md_array_ptr = get_mdarray();
		dim_size = src_sizes[dim];

		if (dim > 0)
		{
			ratio = src_strides[dim - 1] / src_strides[dim];
		}
		else
		{
			ratio = 1;
		}


		if (CPU == options.device_type)
		{
			cpu_std(op1_md_array->GetDataPtr(), md_array_ptr->GetDataPtr(), get_numels(), ratio, dim_size, src_strides[dim]);
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_std(op1_md_array->GetDataPtr(), md_array_ptr->GetDataPtr(), get_numels(), ratio, dim_size, src_strides[dim]);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		md_array_ptr->Reshape(ndims);

		if (operand1.autograd_on())
		{
			misc1_ = dim; // save this for back prop

			add_child(operand1);
			grad_fn_ = ::std_backward;
			set_autograd(true);
		}

	}


	// tensor concatenation
	template<typename Dtype>
	void TensorImpl<Dtype>::cat(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2, int dim)
	{
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		TensorOps options;



		if (operand1.get_ndims() != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array).cat(*op2_md_array, dim);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))).cat(*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array)), dim);

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on() || operand2.autograd_on())
		{
			misc1_ = dim; // save this for back prop

			add_child(operand1);
			add_child(operand2);
			grad_fn_ = ::cat_backward;
			set_autograd(true);
		}

	}

	template<typename Dtype>
	void TensorImpl<Dtype>::squeeze(TensorImpl<Dtype>& operand1, int dim)
	{
		uint64_t dims[MAX_DIMS];
		int i;
		int ndims;
		int index;
		const uint64_t* op1_sizes_ptr;
		TensorOps options;

		ndims = operand1.get_ndims();
		op1_sizes_ptr = operand1.get_sizes();

		index = 0;
		for (i = 0; i < ndims; i++)
		{
			if ((op1_sizes_ptr[i] != 1) || (i != dim))
			{
				dims[index++] = op1_sizes_ptr[i];
			}
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		LTEN_ERR_CHECK(allocate_from_buffer(dims, index, operand1.get_data_ptr(), false, &options));
		operand1.add_ref();

		view_src_ = &operand1;

		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::squeeze_backward;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::unsqueeze(TensorImpl<Dtype>& operand1, int dim)
	{
		uint64_t dims[MAX_DIMS];
		int index;
		int i;
		int ndims;
		const uint64_t* op1_sizes_ptr;
		TensorOps options;

		ndims = operand1.get_ndims();
		op1_sizes_ptr = operand1.get_sizes();

		if (ndims + 1 > MAX_DIMS)
		{
			LTEN_ERR("Resulting tensor will have more dimensions than MAX_DIMS");
		}

		index = 0;

		for (i = 0; i < ndims + 1; i++)
		{
			if (i == dim)
			{
				dims[i] = 1;
			}
			else
			{
				dims[i] = op1_sizes_ptr[index++];
			}
		}

		assert(index == ndims);

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();


		LTEN_ERR_CHECK(allocate_from_buffer(dims, ndims + 1, operand1.get_data_ptr(), false, &options));
		operand1.add_ref();

		view_src_ = &operand1;


		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = ::unsqueeze_backward;
			set_autograd(true);
		}

	}
	template<typename Dtype>
	void TensorImpl<Dtype>::transpose(TensorImpl& operand1, int dim1, int dim2)
	{
		TensorOps options;
		MultiDimArray<Dtype>* op1_md_array;


		op1_md_array = operand1.get_mdarray();

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		if (CPU == options.device_type)
		{
			MultiDimArray<Dtype> result;

			result = op1_md_array->transpose(dim1, dim2);
			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));

			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;


				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))).transpose(dim1, dim2);

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			misc1_ = dim1;
			misc2_ = dim2;
			add_child(operand1);
			grad_fn_ = ::transpose_backward;
			set_autograd(true);
		}

	}

	template<typename Dtype>
	void TensorImpl<Dtype>::reshape(TensorImpl<Dtype>& operand1, const uint64_t* dims, int ndims)
	{
		TensorOps options;

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		allocate_from_buffer(dims, ndims, operand1.get_data_ptr(), false, &options);


		operand1.add_ref();
		view_src_ = &operand1;

		if (operand1.autograd_on())
		{
			add_child(operand1);
			grad_fn_ = nullptr; // so layer is 'skipped' during backprop for speed
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::sub_array(TensorImpl<Dtype>& operand1, int index)
	{
		uint64_t stride;
		int ndims;
		const uint64_t* dims_ptr;
		int i;
		uint64_t dim;
		dtype data_type;
		TensorOps options;

		ndims = operand1.get_ndims();
		data_type = operand1.get_data_type();
		options.data_type = data_type;
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		if (ndims == 1)
		{
			dim = 1;
			allocate_from_buffer(&dim, 1, &((Dtype*)operand1.get_data_ptr())[index], false, &options);
		}
		else
		{
			dims_ptr = operand1.get_sizes();

			stride = 1;
			for (i = 1; i < ndims; i++)
			{
				stride *= dims_ptr[i];
			}

			allocate_from_buffer(&dims_ptr[1], ndims - 1, &((Dtype*)operand1.get_data_ptr())[stride * index], false, &options);
		}


		if (operand1.autograd_on())
		{
			misc1_ = index;
			add_child(operand1);
			grad_fn_ = ::sub_array_backward;
			set_autograd(true);
		}

	}

	template<typename Dtype>
	void TensorImpl<Dtype>::index(TensorImpl<Dtype>& operand1, TensorImpl<int>& index_operand)
	{
		const uint64_t* result_sizes_ptr;
		TensorOps options;
		int ndims;
		MultiDimArray<Dtype>* op1_md_array;

		if (index_operand.get_data_type() != INT32)
		{
			LTEN_ERR("Index tensors must be of type INT32");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();

		if (CPU == options.device_type)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array).index(*(index_operand.get_mdarray()));

			result_sizes_ptr = result.GetSizes();
			ndims = result.GetNDims();

			LTEN_ERR_CHECK(allocate_from_buffer(result_sizes_ptr, ndims, result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))).index((*(static_cast<CUDA_MultiDimArray<int>*>(index_operand.get_mdarray()))));

				result_sizes_ptr = result.GetSizes();
				ndims = result.GetNDims();

				LTEN_ERR_CHECK(allocate_from_buffer(result_sizes_ptr, ndims, result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			add_child(operand1);
			index_operand.add_ref();
			//index_ = &index_operand;
			//grad_fn_ = ::index_backward;
			set_autograd(true);
		}

	}

	template<typename Dtype>
	void TensorImpl<Dtype>::masked_fill(TensorImpl<Dtype>& operand1, TensorImpl<Dtype>& operand2, double value)
	{
		MultiDimArray<Dtype>* op1_md_array;
		MultiDimArray<Dtype>* op2_md_array;
		TensorOps options;



		if (operand1.get_ndims() != operand2.get_ndims())
		{
			LTEN_ERR("Tensors must have the same number of dimensions");
		}

		if (operand1.get_device() != operand2.get_device() ||
			operand1.get_device_index() != operand2.get_device_index() ||
			operand1.get_data_type() != operand2.get_data_type())
		{
			LTEN_ERR("Input tensors have different device type, device indices or data type");
		}

		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();
		op2_md_array = operand2.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array).masked_fill(*op2_md_array, static_cast<Dtype>(value));

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				//result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))) * (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op2_md_array)));

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		if (operand1.autograd_on())
		{
			add_child(operand1);
			add_child(operand2);
			grad_fn_ = ::masked_fill_backward;
			set_autograd(true);
		}
	}

	template<typename Dtype>
	void TensorImpl<Dtype>::repeat(TensorImpl& operand1, const uint32_t* repeats, int nrepeats)
	{
		MultiDimArray<Dtype>* op1_md_array;
		TensorOps options;


		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array).repeat(repeats, nrepeats);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))).repeat(repeats, nrepeats);

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			add_child(operand1);
			misc1_ = nrepeats;
			//grad_fn_ = ::repeat_backward;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::repeat_interleave(TensorImpl& operand1, const uint32_t* repeats, int nrepeats, int dim, uint32_t* scratch)
	{
		MultiDimArray<Dtype>* op1_md_array;
		TensorOps options;


		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		op1_md_array = operand1.get_mdarray();


		if (options.device_type == CPU)
		{
			MultiDimArray<Dtype> result;

			result = (*op1_md_array).repeat_interleave(repeats, nrepeats, dim, scratch);

			LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
			result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> result;

				result = (*(static_cast<CUDA_MultiDimArray<Dtype>*>(op1_md_array))).repeat_interleave(repeats, nrepeats, dim, scratch);

				LTEN_ERR_CHECK(allocate_from_buffer(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true, &options));
				result.SetMemoryOwnership(false); // this TensorImpl<Dtype> needs to keep the md_array's buffer so set ownership accordingly
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}

		if (operand1.autograd_on())
		{
			if (!scratch)
			{
				LTEN_ERR("scratch buffer cannot be non-null if autograd is on");
			}
			add_child(operand1);
			misc1_ = nrepeats;
			misc2_ = dim;
			misc_ptr1_ = scratch;
			//grad_fn_ = ::repeat_interleave_backward;
			set_autograd(true);
		}

	}


	template<typename Dtype>
	void TensorImpl<Dtype>::permute(TensorImpl& operand1, const uint32_t* permutations, int npermutations)
	{
		TensorOps options;
		int ndims;
		int i;
		const uint64_t* src_dims;
		uint64_t dims[MAX_DIMS];
		uint64_t numels;

		ndims = operand1.get_ndims();

		if (ndims != npermutations)
		{
			LTEN_ERR("Permutations must be the same number as tensor dimensions");
		}


		options.data_type = operand1.get_data_type();
		options.device_type = operand1.get_device();
		options.device_index = operand1.get_device_index();

		src_dims = operand1.get_sizes();

		for (i = 0; i < ndims; i++)
		{
			dims[i] = src_dims[permutations[i]];
		}

		
		allocate(dims, ndims, &options);

		numels = operand1.get_numels();

		if (options.device_type == CPU)
		{
			//TODO: move to approprite file and multithread
			OffsetCalc_permutaion ofs(get_strides(), operand1.get_strides(), permutations, ndims);
			Dtype* src = (Dtype*)operand1.get_data_ptr();
			Dtype* dst = (Dtype*)get_data_ptr();

			for (i = 0; i < numels; i++)
			{
				uint32_t offset;
				offset = ofs.GetOffset(i);
				dst[offset] = src[i];
			}
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA

				gpu_permute((Dtype*)get_data_ptr(), (Dtype*)operand1.get_data_ptr(), ndims, operand1.get_numels(), get_strides(), operand1.get_strides(), permutations);

#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


	}


	template<typename Dtype>
	void TensorImpl<Dtype>::add_child(TensorImpl<Dtype>& child)
	{
		intrusive_ptr<TensorImpl>* temp_ptr;

		if (num_children_ >= MAX_CHILDREN)
		{
			LTEN_ERR("More child nodes added to computational graph than maximum allowed");
		}

		temp_ptr = new intrusive_ptr<TensorImpl>(&child);

		children_lock_[num_children_] = temp_ptr;

		children_[num_children_++] = &child;
	}

	template<typename Dtype>
	void TensorImpl<Dtype>::clear_gradients()
	{
		device device_type;

		device_type = get_device();

		if (gradient_ptr_)
		{
			if (CPU == device_type)
			{
				::FillBuffer<Dtype>(gradient_ptr_->GetDataPtr(), gradient_ptr_->GetNumels(), 0);
			}
			else
			{
				if (GPU == device_type)
				{
#ifdef USE_CUDA
					ZeroMemoryOnGPU(gradient_ptr_->GetDataPtr(), sizeof(Dtype) * gradient_ptr_->GetNumels());
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}
		}
	}

	template int TensorImpl<float>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory, TensorOps* options_ptr);
	template int TensorImpl<float>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr);
	template int TensorImpl<float>::allocate(const std::initializer_list<uint64_t>& dims, TensorOps* options_ptr);
	template int TensorImpl<float>::allocate(const uint64_t* dims_ptr, int ndims, TensorOps* options_ptr);
	template void TensorImpl<float>::release_resources();
	template void TensorImpl<float>::matmul(TensorImpl<float>& operand1, TensorImpl<float>& operand2);
	template void TensorImpl<float>::sub_array(TensorImpl<float>& operand1, int index);
	template void TensorImpl<float>::add_child(TensorImpl<float>& child);
	template void TensorImpl<float>::add(TensorImpl<float>& lhs, TensorImpl<float>& rhs);
	template void TensorImpl<float>::sub(TensorImpl<float>& lhs, TensorImpl<float>& rhs);
	template void TensorImpl<float>::div(TensorImpl<float>& operand1, TensorImpl<float>& operand2);
	template void TensorImpl<float>::cat(TensorImpl<float>& operand1, TensorImpl<float>& operand2, int dim);
	template void TensorImpl<float>::exp(TensorImpl<float>& operand1);
	template void TensorImpl<float>::max(TensorImpl<float>& operand1);
	template void TensorImpl<float>::max(TensorImpl<float>& operand1, int dim);
	template void TensorImpl<float>::sum(TensorImpl<float>& operand1);
	template void TensorImpl<float>::sum(TensorImpl<float>& operand1, int dim);
	template void TensorImpl<float>::mean(TensorImpl<float>& operand1);
	template void TensorImpl<float>::mean(TensorImpl<float>& operand1, const uint32_t* axes, int naxes);
	template void TensorImpl<float>::var(TensorImpl<float>& operand1, const uint32_t* axes, int naxes);
	template void TensorImpl<float>::std(TensorImpl<float>& operand1, int dim);
	template void TensorImpl<float>::log(TensorImpl<float>& operand1);
	template void TensorImpl<float>::sig(TensorImpl<float>& operand1);
	template void TensorImpl<float>::tanh(TensorImpl<float>& operand1);
	template void TensorImpl<float>::scalar_mul(TensorImpl<float>& operand1, double scalar);
	template void TensorImpl<float>::mul(TensorImpl<float>& operand1, TensorImpl<float>& operand2);
	template void TensorImpl<float>::sqrt(TensorImpl<float>& operand1);
	template void TensorImpl<float>::clear_gradients();
	template void TensorImpl<float>::squeeze(TensorImpl<float>& operand1, int dim);
	template void TensorImpl<float>::unsqueeze(TensorImpl<float>& operand1, int dim);
	template void TensorImpl<float>::reshape(TensorImpl<float>& operand1, const uint64_t* dims, int ndims);
	template void TensorImpl<float>::to(TensorImpl<float>& operand1, device target_device, int target_device_index);
	template void TensorImpl<float>::transpose(TensorImpl<float>& operand1, int dim1, int dim2);
	template void TensorImpl<float>::masked_fill(TensorImpl<float>& operand1, TensorImpl<float>& mask, double value);
	template void TensorImpl<float>::index(TensorImpl<float>& operand1, TensorImpl<int>& index_operand);
	template void TensorImpl<float>::repeat(TensorImpl<float>& operand1, const uint32_t* repeats, int nrepeats);
	template void TensorImpl<float>::repeat_interleave(TensorImpl<float>& operand1, const uint32_t* dims_times, int ndims, int dim, uint32_t* scratch);
	template void TensorImpl<float>::permute(TensorImpl<float>& operand1, const uint32_t* permutations, int npermutations);

	template int TensorImpl<int>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory, TensorOps* options_ptr);
	template int TensorImpl<int>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr);
	template int TensorImpl<int>::allocate(const std::initializer_list<uint64_t>& dims, TensorOps* options_ptr);
	template int TensorImpl<int>::allocate(const uint64_t* dims_ptr, int ndims, TensorOps* options_ptr);
	template void TensorImpl<int>::release_resources();
	template void TensorImpl<int>::matmul(TensorImpl<int>& operand1, TensorImpl<int>& operand2);
	template void TensorImpl<int>::sub_array(TensorImpl<int>& operand1, int index);
	template void TensorImpl<int>::add_child(TensorImpl<int>& child);
	template void TensorImpl<int>::add(TensorImpl<int>& lhs, TensorImpl<int>& rhs);
	template void TensorImpl<int>::sub(TensorImpl<int>& lhs, TensorImpl<int>& rhs);
	template void TensorImpl<int>::div(TensorImpl<int>& operand1, TensorImpl<int>& operand2);
	template void TensorImpl<int>::cat(TensorImpl<int>& operand1, TensorImpl<int>& operand2, int dim);
	template void TensorImpl<int>::exp(TensorImpl<int>& operand1);
	template void TensorImpl<int>::max(TensorImpl<int>& operand1);
	template void TensorImpl<int>::max(TensorImpl<int>& operand1, int dim);
	template void TensorImpl<int>::sum(TensorImpl<int>& operand1);
	template void TensorImpl<int>::sum(TensorImpl<int>& operand1, int dim);
	template void TensorImpl<int>::mean(TensorImpl<int>& operand1);
	template void TensorImpl<int>::mean(TensorImpl<int>& operand1, const uint32_t* axes, int naxes);
	template void TensorImpl<int>::var(TensorImpl<int>& operand1, const uint32_t* axes, int naxes);
	template void TensorImpl<int>::std(TensorImpl<int>& operand1, int dim);
	template void TensorImpl<int>::log(TensorImpl<int>& operand1);
	template void TensorImpl<int>::sig(TensorImpl<int>& operand1);
	template void TensorImpl<int>::tanh(TensorImpl<int>& operand1);
	template void TensorImpl<int>::scalar_mul(TensorImpl<int>& operand1, double scalar);
	template void TensorImpl<int>::mul(TensorImpl<int>& operand1, TensorImpl<int>& operand2);
	template void TensorImpl<int>::sqrt(TensorImpl<int>& operand1);
	template void TensorImpl<int>::clear_gradients();
	template void TensorImpl<int>::squeeze(TensorImpl<int>& operand1, int dim);
	template void TensorImpl<int>::unsqueeze(TensorImpl<int>& operand1, int dim);
	template void TensorImpl<int>::reshape(TensorImpl<int>& operand1, const uint64_t* dims, int ndims);
	template void TensorImpl<int>::to(TensorImpl<int>& operand1, device target_device, int target_device_index);
	template void TensorImpl<int>::transpose(TensorImpl<int>& operand1, int dim1, int dim2);
	template void TensorImpl<int>::masked_fill(TensorImpl<int>& operand1, TensorImpl<int>& mask, double value);
	template void TensorImpl<int>::index(TensorImpl<int>& operand1, TensorImpl<int>& index_operand);
	template void TensorImpl<int>::repeat(TensorImpl<int>& operand1, const uint32_t* repeats, int nrepeats);
	template void TensorImpl<int>::repeat_interleave(TensorImpl<int>& operand1, const uint32_t* dims_times, int ndims, int dim, uint32_t* scratch);
	template void TensorImpl<int>::permute(TensorImpl<int>& operand1, const uint32_t* permutations, int npermutations);

	template int TensorImpl<uint8_t>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory, TensorOps* options_ptr);
	template int TensorImpl<uint8_t>::allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr);
	template int TensorImpl<uint8_t>::allocate(const std::initializer_list<uint64_t>& dims, TensorOps* options_ptr);
	template int TensorImpl<uint8_t>::allocate(const uint64_t* dims_ptr, int ndims, TensorOps* options_ptr);
	template void TensorImpl<uint8_t>::release_resources();
	template void TensorImpl<uint8_t>::matmul(TensorImpl<uint8_t>& operand1, TensorImpl<uint8_t>& operand2);
	template void TensorImpl<uint8_t>::sub_array(TensorImpl<uint8_t>& operand1, int index);
	template void TensorImpl<uint8_t>::add_child(TensorImpl<uint8_t>& child);
	template void TensorImpl<uint8_t>::add(TensorImpl<uint8_t>& lhs, TensorImpl<uint8_t>& rhs);
	template void TensorImpl<uint8_t>::sub(TensorImpl<uint8_t>& lhs, TensorImpl<uint8_t>& rhs);
	template void TensorImpl<uint8_t>::div(TensorImpl<uint8_t>& operand1, TensorImpl<uint8_t>& operand2);
	template void TensorImpl<uint8_t>::cat(TensorImpl<uint8_t>& operand1, TensorImpl<uint8_t>& operand2, int dim);
	template void TensorImpl<uint8_t>::exp(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::max(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::max(TensorImpl<uint8_t>& operand1, int dim);
	template void TensorImpl<uint8_t>::sum(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::sum(TensorImpl<uint8_t>& operand1, int dim);
	template void TensorImpl<uint8_t>::mean(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::mean(TensorImpl<uint8_t>& operand1, const uint32_t* axes, int naxes);
	template void TensorImpl<uint8_t>::var(TensorImpl<uint8_t>& operand1, const uint32_t* axes, int naxes);
	template void TensorImpl<uint8_t>::std(TensorImpl<uint8_t>& operand1, int dim);
	template void TensorImpl<uint8_t>::log(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::sig(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::tanh(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::scalar_mul(TensorImpl<uint8_t>& operand1, double scalar);
	template void TensorImpl<uint8_t>::mul(TensorImpl<uint8_t>& operand1, TensorImpl<uint8_t>& operand2);
	template void TensorImpl<uint8_t>::sqrt(TensorImpl<uint8_t>& operand1);
	template void TensorImpl<uint8_t>::clear_gradients();
	template void TensorImpl<uint8_t>::squeeze(TensorImpl<uint8_t>& operand1, int dim);
	template void TensorImpl<uint8_t>::unsqueeze(TensorImpl<uint8_t>& operand1, int dim);
	template void TensorImpl<uint8_t>::reshape(TensorImpl<uint8_t>& operand1, const uint64_t* dims, int ndims);
	template void TensorImpl<uint8_t>::to(TensorImpl<uint8_t>& operand1, device target_device, int target_device_index);
	template void TensorImpl<uint8_t>::transpose(TensorImpl<uint8_t>& operand1, int dim1, int dim2);
	template void TensorImpl<uint8_t>::masked_fill(TensorImpl<uint8_t>& operand1, TensorImpl<uint8_t>& mask, double value);
	template void TensorImpl<uint8_t>::index(TensorImpl<uint8_t>& operand1, TensorImpl<int>& index_operand);
	template void TensorImpl<uint8_t>::repeat(TensorImpl<uint8_t>& operand1, const uint32_t* repeats, int nrepeats);
	template void TensorImpl<uint8_t>::repeat_interleave(TensorImpl<uint8_t>& operand1, const uint32_t* dims_times, int ndims, int dim, uint32_t* scratch);
	template void TensorImpl<uint8_t>::permute(TensorImpl<uint8_t>& operand1, const uint32_t* permutations, int npermutations);
} // namespace