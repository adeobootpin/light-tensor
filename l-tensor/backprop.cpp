#include "tensorimpl.h"
#include "layers.h"
#include "im_col.h"
#include "utils.h"

//-------------------------------------------------------------------------------------------------------------------------------------------------
// The function backward() calls do_backward() to recursively traverse the computational graph.  
// Each node has a specific backpropagation handler (e.g. matmul_backward for matrix multiplication) which reads top_gradient and uses it to 
// compute bottom_gradient (bottom_gradient subsequently becomes top_gradient for the node's children).
// Leaf nodes accumulate gradients in their gradient buffers from all the paths that they are a part of.
// A few special cases do not follow this logic.  CUDNN nodes for example use CUDNN to compute gradients and these are accumulated in the 
// backpropagation handler (rather that ouside the handler as usual).  Another example is fc_backward which special cases the handling of bias
// gradients.
//-------------------------------------------------------------------------------------------------------------------------------------------------

namespace lten {
	template<typename Dtype>
	void TensorImpl<Dtype>::backward(MultiDimArray<Dtype>* top_gradient_ptr)
	{
		if (data_type_ != FLOAT32)  // gradient computation only supported for float tensors for now
		{
			LTEN_ERR("Backpropagation is only supported for FLOAT32 tensors");
		}

		if (!autograd_on_)
		{
			LTEN_ERR("autograd is off for this tensor");
		}

		do_backward(top_gradient_ptr);
	}


	template<typename Dtype>
	void TensorImpl<Dtype>::do_backward(MultiDimArray<Dtype>* top_gradient_ptr)
	{
		int ret;
		int i;
		MultiDimArray<Dtype> bottom_gradient_cpu;
#ifdef USE_CUDA
		CUDA_MultiDimArray<Dtype> bottom_gradient_gpu;
#endif
		MultiDimArray<Dtype>* bottom_gradient = nullptr;
		TensorImpl* child_ptr;

		if (!gradient_ptr_ && !num_children_) // store gradients only for leaf nodes
		{
			if (CPU == get_device())
			{
				gradient_ptr_ = new MultiDimArray<Dtype>;
				if (!gradient_ptr_)
				{
					std::terminate(); // no hope, bail
				}
				ret = gradient_ptr_->Allocate(get_sizes(), get_ndims());
				if (ret)
				{
					std::terminate(); // no hope, bail
				}
				::FillBuffer<Dtype>(gradient_ptr_->GetDataPtr(), gradient_ptr_->GetNumels(), 0);
			}
			else
			{
				if (GPU == get_device())
				{
#ifdef USE_CUDA
					gradient_ptr_ = new CUDA_MultiDimArray<Dtype>;
					if (!gradient_ptr_)
					{
						std::terminate();
					}
					ret = gradient_ptr_->Allocate(get_sizes(), get_ndims());
					if (ret)
					{
						std::terminate();
					}
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

		if (!num_children_) // accumulate gradients only for root nodes
		{
			if (top_gradient_ptr)
			{
				assert(gradient_ptr_->GetNumels() == top_gradient_ptr->GetNumels());
				if (CPU == get_device())
				{
					cpu_axpy(gradient_ptr_->GetNumels(), static_cast<Dtype>(1), top_gradient_ptr->GetDataPtr(), gradient_ptr_->GetDataPtr(), gradient_ptr_->GetDataPtr());
				}
				else
				{
					if (GPU == get_device())
					{
#ifdef USE_CUDA
						float alpha;
						cublasHandle_t hCuBlas;
						alpha = 1;

						hCuBlas = CUDA_globlas::singleton()->get_cublas_handle(get_device_index());

						cublasSaxpy(hCuBlas, static_cast<int>(gradient_ptr_->GetNumels()), &alpha, (float*)top_gradient_ptr->GetDataPtr(), 1, (float*)gradient_ptr_->GetDataPtr(), 1);
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
			else
			{
				if (CPU == get_device())
				{
					::FillBuffer<Dtype>(gradient_ptr_->GetDataPtr(), gradient_ptr_->GetNumels(), 1);
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
		}

		for (i = 0; i < num_children_; i++)
		{
			child_ptr = children_[i];

			if (CPU == get_device())
			{
				bottom_gradient = &bottom_gradient_cpu;
				bottom_gradient->Allocate(child_ptr->get_sizes(), child_ptr->get_ndims());
			}
			else
			{
				if (GPU == get_device())
				{
#ifdef USE_CUDA
					bottom_gradient = &bottom_gradient_gpu;
					bottom_gradient->Allocate(child_ptr->get_sizes(), child_ptr->get_ndims());
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}

			if (grad_fn_)
			{
				(this->grad_fn_)(bottom_gradient, top_gradient_ptr, children_, i, this);
			}
			else
			{
				LTEN_ERR("No backpropagation function defined");
			}
			children_[i]->do_backward(bottom_gradient);
		}
	}


	template void TensorImpl<float>::backward(MultiDimArray<float>* top_gradient_ptr);
	template void TensorImpl<int>::backward(MultiDimArray<int>* top_gradient_ptr);
	template void TensorImpl<uint8_t>::backward(MultiDimArray<uint8_t>* top_gradient_ptr);

	template void TensorImpl<float>::do_backward(MultiDimArray<float>* top_gradient_ptr);
	template void TensorImpl<int>::do_backward(MultiDimArray<int>* top_gradient_ptr);
	template void TensorImpl<uint8_t>::do_backward(MultiDimArray<uint8_t>* top_gradient_ptr);

} // namespace lten


template<typename Dtype>
void matmul_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* operand1_md_array;
	MultiDimArray<Dtype>* operand2_md_array;
	Dtype val;
	uint64_t M, N, K;
	const uint64_t* dims_array = nullptr;
	bool broadcast_required;
	uint64_t dims[MAX_DIMS];
	int i;
	int ndims;
	int device_index;
	lten::device device_type;
	int original_ndims = 0;

	device_type = parent_ptr->get_device();
	device_index = parent_ptr->get_device_index();

	operand1_md_array = children_ptr_array[0]->get_mdarray();
	operand2_md_array = children_ptr_array[1]->get_mdarray();

	ndims = parent_ptr->get_ndims();

	if (ndims < 3)
	{
		original_ndims = ndims;
		if (top_gradient_ptr)
		{
			top_gradient_ptr->Reshape(3);
		}
		bottom_gradient_ptr->Reshape(3);
		operand1_md_array->Reshape(3);
		operand2_md_array->Reshape(3);
		ndims = 3;
	}

	if (!top_gradient_ptr)
	{
		for (i = 0; i < ndims; i++)
		{
			dims[i] = 1;
		}

		if (lten::CPU == device_type)
		{
			MultiDimArray<Dtype> fake_top_gradient;

			val = static_cast<Dtype>(1);

			fake_top_gradient.Allocate(dims, ndims, &val, false);
			dims_array = dims;
			top_gradient_ptr = &fake_top_gradient;
		}
		else
		{
			if (lten::GPU == device_type)
			{
#ifdef USE_CUDA
				CUDA_MultiDimArray<Dtype> fake_top_gradient;

				val = static_cast<Dtype>(1);

				fake_top_gradient.Allocate(dims, ndims);
				gpu_fill(fake_top_gradient.GetDataPtr(), 1, val);
				dims_array = dims;
				top_gradient_ptr = &fake_top_gradient;
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
	else
	{
		dims_array = top_gradient_ptr->GetSizes();
	}


	K = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width
	assert(K == children_ptr_array[1]->get_sizes()[children_ptr_array[1]->get_ndims() - 2]);


	assert(ndims == operand1_md_array->GetNDims());

	broadcast_required = top_gradient_ptr->check_broadcast_required(children_ptr_array[0]->get_sizes(), nullptr, true) || top_gradient_ptr->check_broadcast_required(children_ptr_array[1]->get_sizes(), nullptr, true);

	if (lten::CPU == device_type)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);

		md_array_dim_iterator it(dims_array, ndims - 2);
		for (auto higher_indices : it)
		{
			Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

			if (child_index == 0) // operand 1
			{
				M = dims_array[ndims - 2];
				K = dims_array[ndims - 1];
				N = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

				Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), top_data, op2_data, static_cast<Dtype>(1), bottom_data);
			}
			else
			{
				K = dims_array[ndims - 2];
				N = dims_array[ndims - 1];
				M = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

				Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				Dtype* op1_data = operand1_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), op1_data, top_data, static_cast<Dtype>(1), bottom_data);
			}
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			cublasHandle_t hCuBlas;

			hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(device_index);

			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());

			md_array_dim_iterator it(dims_array, ndims - 2);
			for (auto higher_indices : it)
			{
				float alpha;
				float beta;
				int lda;
				int ldb;
				int ldc;

				Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				if (child_index == 0) // operand 1
				{
					alpha = 1.0f;
					beta = 1.0f;

					M = dims_array[ndims - 2];
					K = dims_array[ndims - 1];
					N = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

					lda = static_cast<int>(K);
					ldb = static_cast<int>(K);
					ldc = static_cast<int>(N);

					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)op2_data, lda, (float*)top_data, ldb, &beta, (float*)bottom_data, ldc));
				}
				else
				{
					alpha = 1.0f;
					beta = 1.0f;

					K = dims_array[ndims - 2];
					N = dims_array[ndims - 1];
					M = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

					lda = static_cast<int>(N);
					ldb = static_cast<int>(M);
					ldc = static_cast<int>(N);

					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* op1_data = operand1_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)top_data, lda, (float*)op1_data, ldb, &beta, (float*)bottom_data, ldc));
				}
			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}

	if (original_ndims)
	{
		top_gradient_ptr->Reshape(original_ndims);
		bottom_gradient_ptr->Reshape(original_ndims);
		operand1_md_array->Reshape(original_ndims);
		operand2_md_array->Reshape(original_ndims);
	}

}


template<typename Dtype>
void add_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	uint64_t N;
	const uint64_t* dims_array_top;
	const uint64_t* dims_array_btm;
	int ndims;
	lten::device device_type;
	int original_ndims = 0;
	bool broadcast_required;

	ndims = top_gradient_ptr->GetNDims();

	assert(ndims == bottom_gradient_ptr->GetNDims());
	if (ndims < 3)
	{
		original_ndims = ndims;
		top_gradient_ptr->Reshape(3);
		bottom_gradient_ptr->Reshape(3);
		ndims = 3;
	}

	dims_array_top = top_gradient_ptr->GetSizes();
	dims_array_btm = bottom_gradient_ptr->GetSizes();


	broadcast_required = top_gradient_ptr->check_broadcast_required(bottom_gradient_ptr->GetSizes());

	device_type = parent_ptr->get_device();

	if (device_type == lten::CPU)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);
		if (dims_array_top[ndims - 1] == dims_array_btm[ndims - 1] && dims_array_top[ndims - 2] == dims_array_btm[ndims - 2]) // if same H x W then use faster path
		{
			N = dims_array_top[ndims - 2] * dims_array_top[ndims - 1];

			md_array_dim_iterator it(dims_array_top, ndims - 2);
			for (auto higher_indices : it)
			{
				Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				cpu_axpy(N, static_cast<Dtype>(1), top_data, bottom_data, bottom_data);
			}
		}
		else
		{
			md_array_dim_iterator it(dims_array_top, ndims);

			for (auto indices : it)
			{
				(*bottom_gradient_ptr)(indices, ndims, broadcast_required) += (*top_gradient_ptr)(indices, ndims, broadcast_required);
			}
		}
	}
	else
	{
		if (device_type == lten::GPU)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());
			md_array_dim_iterator it(dims_array_top, ndims - 2);
			for (auto higher_indices : it)
			{
				float* bottom_data = (float*)bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				float* top_data = (float*)top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				gpu_scalar_mul(1.0f, top_data, bottom_data, dims_array_top[ndims - 2], dims_array_top[ndims - 1], dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}

	if (original_ndims)
	{
		top_gradient_ptr->Reshape(original_ndims);
		bottom_gradient_ptr->Reshape(original_ndims);
	}
}

template<typename Dtype>
void sub_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	uint64_t N;
	const uint64_t* dims_array_top;
	const uint64_t* dims_array_btm;
	int ndims;
	lten::device device_type;
	bool broadcast_required;
	int original_ndims = 0;

	ndims = top_gradient_ptr->GetNDims();
	assert(ndims == bottom_gradient_ptr->GetNDims());

	if (ndims < 3)
	{
		original_ndims = ndims;
		top_gradient_ptr->Reshape(3);
		bottom_gradient_ptr->Reshape(3);
		ndims = 3;
	}

	dims_array_top = top_gradient_ptr->GetSizes();
	dims_array_btm = bottom_gradient_ptr->GetSizes();


	broadcast_required = top_gradient_ptr->check_broadcast_required(bottom_gradient_ptr->GetSizes());

	device_type = parent_ptr->get_device();

	if (device_type == lten::CPU)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);

		if (dims_array_top[ndims - 1] == dims_array_btm[ndims - 1] && dims_array_top[ndims - 2] == dims_array_btm[ndims - 2]) // if same H x W then use faster path
		{
			N = dims_array_top[ndims - 2] * dims_array_top[ndims - 1];

			md_array_dim_iterator it(dims_array_top, ndims - 2);
			for (auto higher_indices : it)
			{
				Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				if (child_index == 0) // operand 1
				{
					cpu_axpy(N, static_cast<Dtype>(1), top_data, bottom_data, bottom_data);
				}
				else
				{
					cpu_axpy(N, static_cast<Dtype>(-1), top_data, bottom_data, bottom_data);
				}

			}
		}
		else
		{
			md_array_dim_iterator it(dims_array_top, ndims);

			for (auto indices : it)
			{
				if (child_index == 0) // operand 1
				{
					(*bottom_gradient_ptr)(indices, ndims, broadcast_required) += (*top_gradient_ptr)(indices, ndims, broadcast_required);
				}
				else
				{
					(*bottom_gradient_ptr)(indices, ndims, broadcast_required) -= (*top_gradient_ptr)(indices, ndims, broadcast_required);
				}
			}
		}
	}
	else
	{
		if (device_type == lten::GPU)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());
			md_array_dim_iterator it(dims_array_top, ndims - 2);
			for (auto higher_indices : it)
			{
				float* bottom_data = (float*)bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				float* top_data = (float*)top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				if (child_index == 0) // operand 1
				{
					gpu_scalar_mul(1.0f, top_data, bottom_data,
						dims_array_top[ndims - 2], dims_array_top[ndims - 1],
						dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
				}
				else
				{
					gpu_scalar_mul(-1.0f, top_data, bottom_data,
						dims_array_top[ndims - 2], dims_array_top[ndims - 1],
						dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
				}

			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}

	if (original_ndims)
	{
		top_gradient_ptr->Reshape(original_ndims);
		bottom_gradient_ptr->Reshape(original_ndims);
	}


}


template<typename Dtype>
void mul_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* operand1_md_array;
	MultiDimArray<Dtype>* operand2_md_array;
	uint64_t N;
	lten::device device_type;
	const uint64_t* dims_array_top;
	int original_ndims = 0;
	bool broadcast_required;
	int ndims;

	operand1_md_array = children_ptr_array[0]->get_mdarray();
	operand2_md_array = children_ptr_array[1]->get_mdarray();

	ndims = top_gradient_ptr->GetNDims();
	assert(ndims == bottom_gradient_ptr->GetNDims());
	assert(ndims == children_ptr_array[child_index]->get_ndims());
	if (ndims < 3)
	{
		original_ndims = ndims;
		top_gradient_ptr->Reshape(3);
		bottom_gradient_ptr->Reshape(3);
		operand1_md_array->Reshape(3);
		operand2_md_array->Reshape(3);
		ndims = 3;
	}

	dims_array_top = top_gradient_ptr->GetSizes();

	broadcast_required = top_gradient_ptr->check_broadcast_required(children_ptr_array[0]->get_sizes()) || top_gradient_ptr->check_broadcast_required(children_ptr_array[1]->get_sizes());

	device_type = parent_ptr->get_device();

	if (device_type == lten::CPU)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);

		if (child_index == 0) // operand 1
		{
			if (dims_array_top[ndims - 1] == bottom_gradient_ptr->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == bottom_gradient_ptr->GetSizes()[ndims - 2] &&
				dims_array_top[ndims - 1] == operand2_md_array->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == operand2_md_array->GetSizes()[ndims - 2]) // if same H x W then use faster path
			{
				N = dims_array_top[ndims - 2] * dims_array_top[ndims - 1];

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					cpu_mul(N, static_cast<Dtype>(1), top_data, op2_data, static_cast<Dtype>(1), bottom_data);
				}
			}
			else
			{
				md_array_dim_iterator it(dims_array_top, ndims);
				for (auto indices : it)
				{
					(*bottom_gradient_ptr)(indices, ndims, broadcast_required) += (*top_gradient_ptr)(indices, ndims) * (*operand2_md_array)(indices, ndims, broadcast_required);
				}
			}
		}
		else
		{
			assert(child_index == 1);
			if (dims_array_top[ndims - 1] == bottom_gradient_ptr->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == bottom_gradient_ptr->GetSizes()[ndims - 2] &&
				dims_array_top[ndims - 1] == operand1_md_array->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == operand1_md_array->GetSizes()[ndims - 2]) // if same H x W then use faster path
			{
				N = dims_array_top[ndims - 2] * dims_array_top[ndims - 1];

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op1_data = operand1_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					cpu_mul(N, static_cast<Dtype>(1), top_data, op1_data, static_cast<Dtype>(1), bottom_data);
				}
			}
			else
			{
				md_array_dim_iterator it(dims_array_top, ndims);
				for (auto indices : it)
				{
					(*bottom_gradient_ptr)(indices, ndims, broadcast_required) += (*top_gradient_ptr)(indices, ndims) * (*operand1_md_array)(indices, ndims, broadcast_required);
				}
			}
		}
	}
	else
	{
		if (device_type == lten::GPU)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());
			if (child_index == 0) // operand 1
			{
				const uint64_t* dims_array_btm;
				dims_array_btm = bottom_gradient_ptr->GetSizes();

				const uint64_t* dims_array_op2;
				dims_array_op2 = operand2_md_array->GetSizes();

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					gpu_mul((float*)top_data, (float*)op2_data, (float*)bottom_data,
						dims_array_top[ndims - 2], dims_array_top[ndims - 1],
						dims_array_op2[ndims - 2], dims_array_op2[ndims - 1],
						dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
				}
			}
			else
			{
				assert(child_index == 1);

				const uint64_t* dims_array_btm;
				dims_array_btm = bottom_gradient_ptr->GetSizes();

				const uint64_t* dims_array_op1;
				dims_array_op1 = operand1_md_array->GetSizes();

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op1_data = operand1_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					gpu_mul((float*)top_data, (float*)op1_data, (float*)bottom_data,
						dims_array_top[ndims - 2], dims_array_top[ndims - 1],
						dims_array_op1[ndims - 2], dims_array_op1[ndims - 1],
						dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
				}
			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}


	if (original_ndims)
	{
		top_gradient_ptr->Reshape(original_ndims);
		bottom_gradient_ptr->Reshape(original_ndims);
		operand1_md_array->Reshape(original_ndims);
		operand2_md_array->Reshape(original_ndims);
	}

}

template<typename Dtype>
void div_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* operand1_md_array;
	MultiDimArray<Dtype>* operand2_md_array;
	uint64_t N;
	lten::device device_type;
	const uint64_t* dims_array_top;
	int original_ndims = 0;
	bool broadcast_required;
	int ndims;

	ndims = top_gradient_ptr->GetNDims();
	assert(ndims == bottom_gradient_ptr->GetNDims());
	assert(ndims == children_ptr_array[0]->get_ndims());
	assert(ndims == children_ptr_array[1]->get_ndims());

	operand1_md_array = children_ptr_array[0]->get_mdarray();
	operand2_md_array = children_ptr_array[1]->get_mdarray();

	if (ndims < 3)
	{
		original_ndims = ndims;
		top_gradient_ptr->Reshape(3);
		bottom_gradient_ptr->Reshape(3);
		operand1_md_array->Reshape(3);
		operand2_md_array->Reshape(3);
		ndims = 3;
	}

	dims_array_top = top_gradient_ptr->GetSizes();

	broadcast_required = top_gradient_ptr->check_broadcast_required(children_ptr_array[0]->get_mdarray()->GetSizes()) || top_gradient_ptr->check_broadcast_required(children_ptr_array[1]->get_mdarray()->GetSizes());


	device_type = parent_ptr->get_device();

	if (lten::CPU == device_type)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);

		if (child_index == 0) // operand 1
		{
			if (dims_array_top[ndims - 1] == bottom_gradient_ptr->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == bottom_gradient_ptr->GetSizes()[ndims - 2] &&
				dims_array_top[ndims - 1] == operand2_md_array->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == operand2_md_array->GetSizes()[ndims - 2]) // if same H x W then use faster path
			{
				N = dims_array_top[ndims - 2] * dims_array_top[ndims - 1];

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					//cpu_div(N, top_data, op2_data, bottom_data);
					cpu_div(N, static_cast<Dtype>(1), top_data, op2_data, static_cast<Dtype>(1), bottom_data);
				}
			}
			else
			{
				md_array_dim_iterator it(dims_array_top, ndims);
				for (auto indices : it)
				{
					(*bottom_gradient_ptr)(indices, ndims, broadcast_required) += (*top_gradient_ptr)(indices, ndims, broadcast_required) / (*operand2_md_array)(indices, ndims, broadcast_required);
				}
			}
		}
		else
		{
			assert(child_index == 1);
			if (dims_array_top[ndims - 1] == bottom_gradient_ptr->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == bottom_gradient_ptr->GetSizes()[ndims - 2] &&
				dims_array_top[ndims - 1] == operand1_md_array->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == operand1_md_array->GetSizes()[ndims - 2] &&
				dims_array_top[ndims - 1] == operand2_md_array->GetSizes()[ndims - 1] &&
				dims_array_top[ndims - 2] == operand2_md_array->GetSizes()[ndims - 2]) // if same H x W then use faster path

			{
				N = dims_array_top[ndims - 2] * dims_array_top[ndims - 1];

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op1_data = operand1_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					cpu_div_back(N, top_data, op1_data, op2_data, bottom_data);
				}
			}
			else
			{
				md_array_dim_iterator it(dims_array_top, ndims);
				for (auto indices : it)
				{
					(*bottom_gradient_ptr)(indices, ndims, broadcast_required) += (*top_gradient_ptr)(indices, ndims, broadcast_required) * (static_cast<Dtype>(-1) * (*operand1_md_array)(indices, ndims, broadcast_required) / ((*operand2_md_array)(indices, ndims, broadcast_required) * (*operand2_md_array)(indices, ndims, broadcast_required)));
				}
			}
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());
			if (child_index == 0) // operand 1
			{
				const uint64_t* dims_array_btm;
				dims_array_btm = bottom_gradient_ptr->GetSizes();

				const uint64_t* dims_array_op2;
				dims_array_op2 = operand2_md_array->GetSizes();

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					gpu_div((float*)top_data, (float*)op2_data, (float*)bottom_data,
						dims_array_top[ndims - 2], dims_array_top[ndims - 1],
						dims_array_op2[ndims - 2], dims_array_op2[ndims - 1],
						dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
				}
			}
			else
			{
				const uint64_t* dims_array_btm;
				dims_array_btm = bottom_gradient_ptr->GetSizes();

				const uint64_t* dims_array_op1;
				dims_array_op1 = operand1_md_array->GetSizes();

				const uint64_t* dims_array_op2;
				dims_array_op2 = operand2_md_array->GetSizes();

				md_array_dim_iterator it(dims_array_top, ndims - 2);
				for (auto higher_indices : it)
				{
					Dtype* op1_data = operand1_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* op2_data = operand2_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

					//gpu_div_back(N, top_data, op1_data, op2_data, bottom_data);
					gpu_div_back((float*)top_data, (float*)op1_data, (float*)op2_data, (float*)bottom_data,
						dims_array_top[ndims - 2], dims_array_top[ndims - 1],
						dims_array_op1[ndims - 2], dims_array_op1[ndims - 1],
						dims_array_op2[ndims - 2], dims_array_op2[ndims - 1],
						dims_array_btm[ndims - 2], dims_array_btm[ndims - 1]);
				}
			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}

	if (original_ndims)
	{
		top_gradient_ptr->Reshape(original_ndims);
		bottom_gradient_ptr->Reshape(original_ndims);
		operand1_md_array->Reshape(original_ndims);
		operand2_md_array->Reshape(original_ndims);
	}
}


template<typename Dtype>
void max_backward1(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	uint64_t dim;
	MultiDimArray<uint64_t> indices;
	const uint64_t* dst_sizes;
	const uint64_t* dst_strides;
	uint64_t dim_size;
	uint64_t ratio;
	lten::device device_type;
	int ndims;

	device_type = parent_ptr->get_device();

	ndims = parent_ptr->get_ndims();

	dim = static_cast<uint64_t>(parent_ptr->misc1_);

	dst_sizes = bottom_gradient_ptr->GetSizes();
	dst_strides = bottom_gradient_ptr->GetStrides();

	indices.Allocate(parent_ptr->get_sizes(), parent_ptr->get_ndims(), static_cast<uint64_t*>(parent_ptr->misc_ptr1_), false);


	indices.Reshape(ndims - 1);
	top_gradient_ptr->Reshape(ndims - 1);

	dim_size = dst_sizes[dim];

	if (dim > 0)
	{
		ratio = dst_strides[dim - 1] / dst_strides[dim];
	}
	else
	{
		ratio = 1;
	}


	assert(top_gradient_ptr->GetNDims() == bottom_gradient_ptr->GetNDims() - 1);

	if (lten::CPU == device_type)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);
		cpu_max_backward(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), indices.GetDataPtr(), top_gradient_ptr->GetNumels(), ratio, dim_size, dst_strides[dim]);
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());
			gpu_max_backward(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), indices.GetDataPtr(), top_gradient_ptr->GetNumels(), ratio, dim_size, dst_strides[dim]);
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}


	top_gradient_ptr->Reshape(ndims);
	indices.Reshape(ndims);

}


template<typename Dtype>
void sum_backward1(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	int dim;
	//uint64_t n, c, h, w;
	//uint64_t N, C, H, W;
	const uint64_t* dst_sizes;
	const uint64_t* dst_strides;
	uint64_t dim_size;
	uint64_t ratio;
	lten::device device_type;
	int ndims;

	device_type = parent_ptr->get_device();

	ndims = parent_ptr->get_ndims();

	dim = static_cast<int>(parent_ptr->misc1_);

	dst_sizes = bottom_gradient_ptr->GetSizes();
	dst_strides = bottom_gradient_ptr->GetStrides();

	top_gradient_ptr->Reshape(ndims - 1);

	dim_size = dst_sizes[dim];

	if (dim > 0)
	{
		ratio = dst_strides[dim - 1] / dst_strides[dim];
	}
	else
	{
		ratio = 1;
	}

	if (lten::CPU == device_type)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);
		cpu_sum_backward(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetNumels(), ratio, dim_size, dst_strides[dim]);
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());
			gpu_sum_backward(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetNumels(), ratio, dim_size, dst_strides[dim]);
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tensor device type");
		}
	}


	top_gradient_ptr->Reshape(ndims);

}

template<typename Dtype>
void max_backward2(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* operand1_md_array;
	uint64_t len;
	uint64_t i;
	Dtype gradient;
	Dtype max;

	max = static_cast<Dtype>(parent_ptr->misc2_);
	gradient = static_cast<Dtype>(1) / static_cast<Dtype>(parent_ptr->misc1_); // 1.0 / max_count (pytorch compatible)
	//gradient = static_cast<Dtype>(1); // using unscaled gradient like libtorch does (this does not give the same result as pytorch which uses 1.0/max_count)

	operand1_md_array = children_ptr_array[0]->get_mdarray();

	len = bottom_gradient_ptr->GetNumels();

	for (i = 0; i < len; i++)
	{
		if (operand1_md_array->GetDataPtr()[i] == max)
		{
			bottom_gradient_ptr->GetDataPtr()[i] = gradient;
		}
		else
		{
			bottom_gradient_ptr->GetDataPtr()[i] = 0;
		}
	}

}

template<typename Dtype>
void sum_backward2(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	lten::device device_type = parent_ptr->get_device();

	if (top_gradient_ptr)
	{
		assert(top_gradient_ptr->GetNumels() == 1);
		if (lten::CPU == device_type)
		{
			FillBuffer(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), top_gradient_ptr->GetDataPtr()[0]);
		}
		else
		{
			if (lten::GPU == device_type)
			{
#ifdef USE_CUDA
				gpu_fill<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), top_gradient_ptr->GetDataPtr());
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
	else
	{
		if (lten::CPU == device_type)
		{
			FillBuffer(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), static_cast<Dtype>(1));
		}
		else
		{
			if (lten::GPU == device_type)
			{
#ifdef USE_CUDA
				gpu_fill(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), static_cast<Dtype>(1));
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


template<typename Dtype>
void scalar_mul_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	uint64_t N;
	MultiDimArray<Dtype>* operand1_md_array;
	const uint64_t* dims_array;
	lten::device device_type;
	Dtype scale;
	int ndims;

	assert(child_index == 0);

	device_type = parent_ptr->get_device();

	scale = static_cast<Dtype>(parent_ptr->misc2_);

	operand1_md_array = children_ptr_array[0]->get_mdarray();

	ndims = bottom_gradient_ptr->GetNDims();
	assert(ndims <= MAX_DIMS);

	if (!top_gradient_ptr)
	{
		assert(bottom_gradient_ptr->GetNumels() == 1);
		if (device_type == lten::CPU)
		{
			bottom_gradient_ptr->GetDataPtr()[0] = scale;
		}
		else
		{
			if (device_type == lten::GPU)
			{
#ifdef USE_CUDA
				CopyDataToGPU(bottom_gradient_ptr->GetDataPtr(), &scale, sizeof(float));
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
	else
	{
		assert(bottom_gradient_ptr->GetNumels() == top_gradient_ptr->GetNumels());
		dims_array = top_gradient_ptr->GetSizes();

		N = dims_array[ndims - 2] * dims_array[ndims - 1];
		md_array_dim_iterator it(dims_array, ndims - 2);

		if (device_type == lten::CPU)
		{
			FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);
			
			for (auto higher_indices : it)
			{
				Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2);
				Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2);

				cpu_axpy(N, scale, top_data, bottom_data, bottom_data);
			}
		}
		else
		{
			if (device_type == lten::GPU)
			{
#ifdef USE_CUDA
				ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());

				for (auto higher_indices : it)
				{
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 2);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 2);

					gpu_axpy(N, scale, top_data, bottom_data, bottom_data);
				}
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


template<typename Dtype>
void conv2_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	lten::Conv2d* conv2d_ptr;
	const uint64_t* input_dims_ptr;
	const uint64_t* top_dims_array;
	uint64_t dims[MAX_DIMS];
	int ndims;
	int i;
	uint64_t batches_in;
	uint64_t stride_in;
	uint64_t channels_out;
	uint64_t channels_in;
	uint64_t height_in;
	uint64_t width_in;
	uint64_t kernel_h;
	uint64_t kernel_w;
	uint64_t pad_h;
	uint64_t pad_w;
	uint64_t stride_h;
	uint64_t stride_w;
	uint64_t height_out;
	uint64_t width_out;
	lten::Tensor* col_buffer_ptr;
	MultiDimArray<Dtype> col_buffer_md_array;
	MultiDimArray<Dtype> bottom_gradient_view;
	MultiDimArray<Dtype> top_gradient_view;
	MultiDimArray<Dtype> fake_top_gradient;
	MultiDimArray<Dtype>* weights_md_array_ptr;
	MultiDimArray<Dtype>* inputs_md_array_ptr;

	Dtype val;
	uint64_t n;
	uint64_t N, H, W, K;
	int device_index;
	lten::device device_type;


	device_index = parent_ptr->get_device_index();
	device_type = parent_ptr->get_device();


	assert(parent_ptr->get_data_type() == lten::FLOAT32 && children_ptr_array[0]->get_data_type() == lten::FLOAT32);

	ndims = parent_ptr->get_ndims();

	if (!top_gradient_ptr)
	{
		for (i = 0; i < ndims; i++)
		{
			dims[i] = 1;
		}

		val = static_cast<Dtype>(1);
		fake_top_gradient.Allocate(dims, ndims, &val, false);
		top_dims_array = dims;
		top_gradient_ptr = &fake_top_gradient;
	}
	else
	{
		top_dims_array = top_gradient_ptr->GetSizes();
	}

	weights_md_array_ptr = children_ptr_array[0]->get_mdarray(); // weights
	inputs_md_array_ptr = children_ptr_array[1]->get_mdarray(); // input

	conv2d_ptr = static_cast<lten::Conv2d*>(parent_ptr->misc_ptr1_);

	channels_out = conv2d_ptr->get_channels_out();
	channels_in = conv2d_ptr->get_channels_in();
	kernel_h = conv2d_ptr->get_kernel_h();
	kernel_w = conv2d_ptr->get_kernel_w();
	pad_h = conv2d_ptr->get_pad_h();
	pad_w = conv2d_ptr->get_pad_w();
	stride_h = conv2d_ptr->get_stride_h();
	stride_w = conv2d_ptr->get_stride_w();


	input_dims_ptr = children_ptr_array[1]->get_sizes();
	batches_in = input_dims_ptr[0];
	height_in = input_dims_ptr[2];
	width_in = input_dims_ptr[3];

	height_out = (height_in + 2 * pad_h - kernel_h) / stride_h + 1;
	width_out = (width_in + 2 * pad_w - kernel_w) / stride_w + 1;


	col_buffer_ptr = conv2d_ptr->get_col_buffer_ptr();

	stride_in = channels_in * height_in * width_in;

	N = top_dims_array[0];
	H = channels_out;
	W = height_out * width_out;
	K = channels_in * kernel_h * kernel_w;

	assert(N == batches_in);

	if (lten::CPU == device_type)
	{
		if (child_index == 0) // weights
		{
			FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);

			col_buffer_md_array.Allocate({ 1, 1, channels_in * kernel_h * kernel_w, height_out * width_out }, static_cast<Dtype*>(col_buffer_ptr->get_data_ptr()), false);
			bottom_gradient_view.Allocate({ 1, 1, channels_out, channels_in * kernel_h * kernel_w }, static_cast<Dtype*>(bottom_gradient_ptr->GetDataPtr()), false);
			top_gradient_view.Allocate({ N, 1, channels_out, height_out * width_out }, static_cast<Dtype*>(top_gradient_ptr->GetDataPtr()), false);

			for (n = 0; n < N; n++)
			{
				im2col_cpu((float*)children_ptr_array[1]->get_data_ptr() + n * stride_in, channels_in, height_in, width_in, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, (float*)col_buffer_ptr->get_data_ptr());
				cpu_gemm(false, true, H, K, W, static_cast<Dtype>(1), top_gradient_view.GetDataPtr(&n, 1), col_buffer_md_array.GetDataPtr(), static_cast<Dtype>(1), bottom_gradient_ptr->GetDataPtr());
			}
		}
		else
		{
			FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);

			weights_md_array_ptr = conv2d_ptr->get_weights()->get_mdarray<Dtype>();
			uint64_t stride_top = top_gradient_ptr->GetNumels() / top_gradient_ptr->GetSizes()[0];
			uint64_t stride_bottom = bottom_gradient_ptr->GetNumels() / bottom_gradient_ptr->GetSizes()[0];

			for (n = 0; n < N; n++)
			{
				cpu_gemm(true, false, channels_in * kernel_h * kernel_w, height_out * width_out, channels_out, static_cast<Dtype>(1), weights_md_array_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr() + stride_top * n, static_cast <Dtype>(0), static_cast<Dtype*>(col_buffer_ptr->get_data_ptr()));
				col2im_cpu(static_cast<Dtype*>(col_buffer_ptr->get_data_ptr()), channels_in, height_in, width_in, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, bottom_gradient_ptr->GetDataPtr() + stride_bottom * n);
			}
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			float alpha;
			float beta;
			int lda;
			int ldb;
			int ldc;

			cublasHandle_t hCuBlas;


			hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(device_index);

			if (child_index == 0) // weights
			{
				alpha = 1.0f;
				beta = 0.0f;

				lda = static_cast<int>(W);
				ldb = static_cast<int>(W);
				ldc = static_cast<int>(K);

				col_buffer_md_array.Allocate({ 1, 1, channels_in * kernel_h * kernel_w, height_out * width_out }, static_cast<Dtype*>(col_buffer_ptr->get_data_ptr()), false);
				bottom_gradient_view.Allocate({ 1, 1, channels_out, channels_in * kernel_h * kernel_w }, static_cast<Dtype*>(bottom_gradient_ptr->GetDataPtr()), false);
				top_gradient_view.Allocate({ N, 1, channels_out, height_out * width_out }, static_cast<Dtype*>(top_gradient_ptr->GetDataPtr()), false);

				for (n = 0; n < N; n++)
				{
					im2col_gpu((float*)children_ptr_array[1]->get_data_ptr() + n * stride_in, (int)channels_in, (int)height_in, (int)width_in, (int)kernel_h, (int)kernel_w, (int)pad_h, (int)pad_w, (int)stride_h, (int)stride_w, 1, 1, (float*)col_buffer_ptr->get_data_ptr());

					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(K), static_cast<int>(H), static_cast<int>(W), &alpha,
						(float*)col_buffer_md_array.GetDataPtr(), lda, (float*)top_gradient_view.GetDataPtr(&n, 1), ldb, &beta, (float*)bottom_gradient_ptr->GetDataPtr(), ldc));
				}
			}
			else
			{
				alpha = 1.0f;
				beta = 0.0f;

				lda = static_cast<int>(W);
				ldb = static_cast<int>(K);
				ldc = static_cast<int>(W);

				ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());

				weights_md_array_ptr = conv2d_ptr->get_weights()->get_mdarray<Dtype>();
				uint64_t stride_top = top_gradient_ptr->GetNumels() / top_gradient_ptr->GetSizes()[0];
				uint64_t stride_bottom = bottom_gradient_ptr->GetNumels() / bottom_gradient_ptr->GetSizes()[0];

				for (n = 0; n < N; n++)
				{
					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(W), static_cast<int>(K), static_cast<int>(H), &alpha,
						(float*)top_gradient_ptr->GetDataPtr() + stride_top * n, lda, (float*)weights_md_array_ptr->GetDataPtr(), ldb, &beta, (float*)col_buffer_ptr->get_data_ptr(), ldc));

					col2im_gpu(static_cast<Dtype*>(col_buffer_ptr->get_data_ptr()), (int)channels_in, (int)height_in, (int)width_in, (int)kernel_h, (int)kernel_w, (int)pad_h, (int)pad_w, (int)stride_h, (int)stride_w, 1, 1, bottom_gradient_ptr->GetDataPtr() + stride_bottom * n);
				}
			}
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

#ifdef USE_CUDA
template<typename Dtype>
void conv2_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	lten::conv2d_CUDNN* conv2d_ptr;
	float alpha;
	float beta;
	cudnnHandle_t cudnnHandle;

	conv2d_ptr = static_cast<lten::conv2d_CUDNN*>(parent_ptr->misc_ptr1_);

	cudnnHandle = lten::CUDA_globlas::singleton()->get_cudnn_handle(0);

	alpha = 1.0f;
	beta = 1.0f; // so perform additive processing here since these are the *actual* gradient buffers (i.e. not gradients to be passed to children since weights and biases are termnal leaves)

	if (conv2d_ptr->is_using_bias())
	{
		cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2d_ptr->get_outputDesc(), top_gradient_ptr->GetDataPtr(), &beta, conv2d_ptr->get_biasDesc(), conv2d_ptr->get_bias()->get_grad_ptr());
	}
	
	assert(child_index == 0);
	cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, conv2d_ptr->get_inputDesc(), children_ptr_array[child_index]->get_data_ptr(), conv2d_ptr->get_outputDesc(), top_gradient_ptr->GetDataPtr(), 
		conv2d_ptr->get_convDesc(), 
		conv2d_ptr->get_bwf_algo(),
		conv2d_ptr->get_bwf_workspace(),
		conv2d_ptr->get_bwf_workspace_size(),
		&beta, conv2d_ptr->get_wtDesc(), conv2d_ptr->get_weights()->get_grad_ptr());


	beta = 0.0f; // write gradients for children into uninitialized scratch buffer (additive processing handled by caller)
	cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2d_ptr->get_wtDesc(), conv2d_ptr->get_weights()->get_data_ptr(), conv2d_ptr->get_outputDesc(), top_gradient_ptr->GetDataPtr(), 
		conv2d_ptr->get_convDesc(), 
		conv2d_ptr->get_bwd_algo(),
		conv2d_ptr->get_bwd_workspace(),
		conv2d_ptr->get_bwd_workspace_size(),
		&beta, conv2d_ptr->get_inputDesc(), bottom_gradient_ptr->GetDataPtr());

}
#endif

template<typename Dtype>
void fc_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	lten::FullyConnected* fc;
	MultiDimArray<Dtype>* operand1_md_array;
	MultiDimArray<Dtype>* operand2_md_array;
	MultiDimArray<Dtype> fake_top_gradient;
	Dtype val;
	uint64_t M, N, K;
	const uint64_t* dims_array;
	uint64_t dims[MAX_DIMS];
	int i;
	int device_index;
	lten::device device_type;
	int ndims;

	device_index = parent_ptr->get_device_index();
	device_type = parent_ptr->get_device();

	ndims = parent_ptr->get_ndims();

	if (!top_gradient_ptr)
	{
		for (i = 0; i < ndims; i++)
		{
			dims[i] = 1;
		}

		val = static_cast<Dtype>(1);

		fake_top_gradient.Allocate(dims, ndims, &val, false);
		dims_array = dims;
		top_gradient_ptr = &fake_top_gradient;
	}
	else
	{
		dims_array = top_gradient_ptr->GetSizes();
	}


	K = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width
	assert(K == children_ptr_array[1]->get_sizes()[children_ptr_array[1]->get_ndims() - 1]); // (-1 instead of -2 since weights are transposed for performace)

	operand1_md_array = children_ptr_array[0]->get_mdarray();
	operand2_md_array = children_ptr_array[1]->get_mdarray();

	assert(ndims == operand1_md_array->GetNDims());

	fc = static_cast<lten::FullyConnected*>(parent_ptr->misc_ptr1_);

	if (device_type == lten::CPU)
	{
		Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();

		if (child_index == 0) // operand 1
		{
			M = 1;
			for (i = 0; i < ndims - 2; i++) // fold in all batches so that only one gemm call is needed
			{
				M *= dims_array[i];
			}

			M *= dims_array[ndims - 2];
			K = dims_array[ndims - 1];
			N = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

			Dtype* top_data = top_gradient_ptr->GetDataPtr();
			Dtype* op2_data = operand2_md_array->GetDataPtr();
			cpu_gemm(false, false, M, N, K, static_cast<Dtype>(1), top_data, op2_data, static_cast<Dtype>(0), bottom_data);
		}
		else
		{
			K = 1;
			for (i = 0; i < ndims - 2; i++) // fold in all batches so that only one gemm call is needed
			{
				K *= dims_array[i];
			}

			K *= dims_array[ndims - 2];
			M = dims_array[ndims - 1];
			N = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

			Dtype* top_data = top_gradient_ptr->GetDataPtr();
			Dtype* op1_data = operand1_md_array->GetDataPtr();
			cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), top_data, op1_data, static_cast<Dtype>(0), bottom_data);


			if (fc->is_using_bias())
			{
				M = 1;
				for (i = 0; i < ndims - 2; i++) // fold in all batches so that only one gemm call is needed
				{
					M *= dims_array[i];
				}

				M *= dims_array[ndims - 2];
				K = dims_array[ndims - 1];

				Dtype* bias_grad = (Dtype*)fc->get_bias()->get_grad_ptr();
				cpu_gemm(false, false, 1, K, M, static_cast<Dtype>(1), static_cast<Dtype*>(fc->get_bias_multiplier()->get_data_ptr()), top_data, static_cast<Dtype>(1), bias_grad);
			}

		}
	}
	else
	{
		if (device_type == lten::GPU)
		{
#ifdef USE_CUDA
			float alpha;
			float beta;
			int lda;
			int ldb;
			int ldc;

			cublasHandle_t hCuBlas;

			assert(lten::FLOAT32 == parent_ptr->get_data_type());


			hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(device_index);

			Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();

			if (child_index == 0) // operand 1
			{
				M = 1;
				for (i = 0; i < ndims - 2; i++) // fold in all batches so that only one gemm call is needed
				{
					M *= dims_array[i];
				}

				M *= dims_array[ndims - 2];
				K = dims_array[ndims - 1];
				N = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

				lda = static_cast<int>(N);
				ldb = static_cast<int>(K);
				ldc = static_cast<int>(N);

				alpha = 1;
				beta = 0;

				float* top_data = (float*)top_gradient_ptr->GetDataPtr();
				float* op2_data = (float*)operand2_md_array->GetDataPtr();
				LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha, op2_data, lda, top_data, ldb, &beta, (float*)bottom_data, ldc));
			}
			else
			{
				K = 1;
				for (i = 0; i < ndims - 2; i++) // fold in all batches so that only one gemm call is needed
				{
					K *= dims_array[i];
				}

				K *= dims_array[ndims - 2];
				M = dims_array[ndims - 1];
				N = children_ptr_array[0]->get_sizes()[children_ptr_array[0]->get_ndims() - 1]; // lhs width

				lda = static_cast<int>(N);
				ldb = static_cast<int>(M);
				ldc = static_cast<int>(N);

				alpha = 1;
				beta = 0;

				float* top_data = (float*)top_gradient_ptr->GetDataPtr();
				float* op1_data = (float*)operand1_md_array->GetDataPtr();

				LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha, op1_data, lda, top_data, ldb, &beta, (float*)bottom_data, ldc));

				if (fc->is_using_bias())
				{
					M = 1;
					for (i = 0; i < ndims - 2; i++) // fold in all batches so that only one gemm call is needed
					{
						M *= dims_array[i];
					}

					M *= dims_array[ndims - 2];
					K = dims_array[ndims - 1];

					Dtype* bias_grad = (Dtype*)fc->get_bias()->get_grad_ptr();
					beta = 1.0f;
					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(K), static_cast<int>(1), static_cast<int>(M), &alpha,	top_data, static_cast<int>(K), static_cast<float*>(fc->get_bias_multiplier()->get_data_ptr()),
						static_cast<int>(M), &beta, (float*)bias_grad, static_cast<int>(K)));
				}
			}
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
void relu_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	int64_t numels;
	int64_t i;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	numels = children_ptr_array[0]->get_numels();
	assert(numels == bottom_gradient_ptr->GetNumels());

	Dtype* op1_data = children_ptr_array[0]->get_mdarray()->GetDataPtr();
	Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();
	Dtype* top_data = top_gradient_ptr->GetDataPtr();

	if (lten::CPU == device_type)
	{
		for (i = 0; i < numels; i++)
		{
			if (op1_data[i] > 0)
			{
				bottom_data[i] = top_data[i];
			}
			else
			{
				bottom_data[i] = 0;
			}
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_relu_backward(bottom_data, top_data, op1_data, numels);
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
void dropout_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	lten::Dropout* dropout;
	MultiDimArray<Dtype>* operand1_md_array;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	operand1_md_array = children_ptr_array[0]->get_mdarray();

	assert(child_index == 0);
	assert(top_gradient_ptr->GetNumels() == bottom_gradient_ptr->GetNumels());
	assert(top_gradient_ptr->GetNumels() == operand1_md_array->GetNumels());

	dropout = static_cast<lten::Dropout*>(parent_ptr->misc_ptr1_);

	if (lten::CPU == device_type)
	{
		cpu_dropout((float*)bottom_gradient_ptr->GetDataPtr(), (float*)top_gradient_ptr->GetDataPtr(), dropout->get_mask()->GetDataPtr(), dropout->get_threshold(), dropout->get_scale(), bottom_gradient_ptr->GetNumels());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_dropout((float*)bottom_gradient_ptr->GetDataPtr(), (float*)top_gradient_ptr->GetDataPtr(), dropout->get_mask()->GetDataPtr(), dropout->get_threshold(), dropout->get_scale(), bottom_gradient_ptr->GetNumels());
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

#ifdef USE_CUDA
template<typename Dtype>
void softmax_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	cudnnHandle_t cudnnHandle;
	lten::softmax_CUDNN* softmax_cudnn;
	float alpha;
	float beta;

	softmax_cudnn = static_cast<lten::softmax_CUDNN*>(parent_ptr->misc_ptr1_);


	cudnnHandle = lten::CUDA_globlas::singleton()->get_cudnn_handle(0);

	alpha = 1.0f;
	beta = 0;

	
	cudnnSoftmaxBackward(cudnnHandle,
		softmax_cudnn->get_algo(),
		CUDNN_SOFTMAX_MODE_INSTANCE,
		&alpha,
		*softmax_cudnn->get_inputDesc(),
		parent_ptr->get_data_ptr(),
		*softmax_cudnn->get_outputDesc(),
		top_gradient_ptr->GetDataPtr(),
		&beta,
		*softmax_cudnn->get_inputDesc(),
		bottom_gradient_ptr->GetDataPtr());


}
#endif

template<typename Dtype>
void squeeze_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	int64_t numels;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	numels = parent_ptr->get_numels();

	assert(numels == top_gradient_ptr->GetNumels());
	assert(numels == bottom_gradient_ptr->GetNumels());

	if (lten::CPU == device_type)
	{
		cpu_copy(numels, top_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetDataPtr());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			GPUToGPUCopy(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), numels * sizeof(Dtype));
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
void unsqueeze_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	int64_t numels;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	numels = parent_ptr->get_numels();

	assert(numels == top_gradient_ptr->GetNumels());
	assert(numels == bottom_gradient_ptr->GetNumels());

	if (lten::CPU == device_type)
	{
		cpu_copy(numels, top_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetDataPtr());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			GPUToGPUCopy(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), numels * sizeof(Dtype));
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
void reshape_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	int64_t numels;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	numels = parent_ptr->get_numels();

	assert(numels == top_gradient_ptr->GetNumels());
	assert(numels == bottom_gradient_ptr->GetNumels());

	if (lten::CPU == device_type)
	{
		cpu_copy(numels, top_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetDataPtr());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			GPUToGPUCopy(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), numels * sizeof(Dtype));
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
void sub_array_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	const uint64_t* op1_strides;
	MultiDimArray<Dtype>* operand1_md_array;
	int index;
	bool broadcast_required;
	const uint64_t* dims_array;
	uint64_t N;
	lten::device device_type;
	int ndims;
	int top_ndims;

	assert(child_index == 0);

	index = static_cast<int>(parent_ptr->misc1_);

	operand1_md_array = children_ptr_array[0]->get_mdarray();

	op1_strides = operand1_md_array->GetStrides();

	device_type = parent_ptr->get_device();

	if (lten::CPU == device_type)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr() + index * op1_strides[0], op1_strides[0], 1);
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_fill(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), static_cast<Dtype>(0));
			gpu_fill(bottom_gradient_ptr->GetDataPtr() + index * op1_strides[0], op1_strides[0], static_cast<Dtype>(1));
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

	if (top_gradient_ptr)
	{
		ndims = bottom_gradient_ptr->GetNDims();
		top_ndims = top_gradient_ptr->GetNDims();
		if (ndims != top_ndims)
		{
			assert(ndims >= top_ndims);
			top_gradient_ptr->Reshape(ndims);
		}

		broadcast_required = true;
		dims_array = bottom_gradient_ptr->GetSizes();

		if (dims_array[ndims - 1] == top_gradient_ptr->GetSizes()[ndims - 1])
		{
			N = dims_array[ndims - 1];

			md_array_dim_iterator it(dims_array, ndims - 1);

			if (lten::CPU == device_type)
			{
				for (auto higher_indices : it)
				{
					Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 1);
					Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 1, broadcast_required);

					cpu_mul(N, top_data, bottom_data, bottom_data);
				}
			}
			else
			{
				if (lten::GPU == device_type)
				{
#ifdef USE_CUDA
					for (auto higher_indices : it)
					{
						Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr(higher_indices, ndims - 1);
						Dtype* top_data = top_gradient_ptr->GetDataPtr(higher_indices, ndims - 1, broadcast_required);

						gpu_mul(top_data, bottom_data, bottom_data, N, 1, N, 1, static_cast<Dtype>(0));
					}

#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tesor data type");
				}
			}
		}
		else
		{
			assert(0);
		}

	}
}

template<typename Dtype>
void exp_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* md_array_ptr;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	md_array_ptr = parent_ptr->get_mdarray();

	assert(child_index == 0);
	assert(top_gradient_ptr->GetNumels() == bottom_gradient_ptr->GetNumels());
	assert(top_gradient_ptr->GetNumels() == md_array_ptr->GetNumels());

	if (lten::CPU == device_type)
	{
		cpu_mul(bottom_gradient_ptr->GetNumels(), static_cast<Dtype>(1), top_gradient_ptr->GetDataPtr(), md_array_ptr->GetDataPtr(), static_cast<Dtype>(0), bottom_gradient_ptr->GetDataPtr());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_mul(bottom_gradient_ptr->GetNumels(), top_gradient_ptr->GetDataPtr(), md_array_ptr->GetDataPtr(), bottom_gradient_ptr->GetDataPtr());
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

}


template<typename Dtype>
void log_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* operand1_md_array;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	operand1_md_array = children_ptr_array[0]->get_mdarray();

	assert(child_index == 0);
	assert(top_gradient_ptr->GetNumels() == bottom_gradient_ptr->GetNumels());
	assert(top_gradient_ptr->GetNumels() == operand1_md_array->GetNumels());

	if (lten::CPU == device_type)
	{
		cpu_div(bottom_gradient_ptr->GetNumels(), top_gradient_ptr->GetDataPtr(), operand1_md_array->GetDataPtr(), bottom_gradient_ptr->GetDataPtr());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_div(bottom_gradient_ptr->GetNumels(), top_gradient_ptr->GetDataPtr(), operand1_md_array->GetDataPtr(), bottom_gradient_ptr->GetDataPtr());
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

}

template<typename Dtype>
void sig_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* result_md_array;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	result_md_array = parent_ptr->get_mdarray();

	assert(child_index == 0);
	assert(top_gradient_ptr->GetNumels() == bottom_gradient_ptr->GetNumels());

	uint64_t len;
	uint64_t i;

	len = bottom_gradient_ptr->GetNumels();

	if (lten::CPU == device_type)
	{
		for (i = 0; i < len; i++)
		{
			Dtype val;
			val = result_md_array->GetDataPtr()[i];
			bottom_gradient_ptr->GetDataPtr()[i] = top_gradient_ptr->GetDataPtr()[i] * val * (1 - val);
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_sig_backward(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), result_md_array->GetDataPtr(), len);
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

}

template<typename Dtype>
void tanh_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* result_md_array;
	lten::device device_type;

	device_type = parent_ptr->get_device();

	result_md_array = parent_ptr->get_mdarray();

	assert(child_index == 0);
	assert(top_gradient_ptr->GetNumels() == bottom_gradient_ptr->GetNumels());

	uint64_t len;
	uint64_t i;

	len = bottom_gradient_ptr->GetNumels();

	if (lten::CPU == device_type)
	{
		for (i = 0; i < len; i++)
		{
			Dtype val;
			val = result_md_array->GetDataPtr()[i];
			bottom_gradient_ptr->GetDataPtr()[i] = top_gradient_ptr->GetDataPtr()[i] * (1 - val * val);
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			gpu_tanh_backward(bottom_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetDataPtr(), result_md_array->GetDataPtr(), len);
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

}

template<typename Dtype>
void gru_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	lten::GRU* gru_ptr;
	uint64_t sequence_len;
	int64_t j;
	uint64_t i;
	uint64_t batches;
	uint64_t hidden_dim;
	uint64_t input_dim;
	uint64_t M, N, K;
	bool bidirectional;
	bool use_bias;

	Dtype* top_gradient;
	Dtype* bottom_gradient;
	Dtype* hidden_gradient;
	Dtype* gradient_0;
	Dtype* gradient_1;
	Dtype* gradient_2;
	Dtype* gradient_3;
	Dtype* gradient_4;
	Dtype* scratch_gradient;
	Dtype* weights;
	Dtype* input;

	lten::Tensor* hidden_array;
	lten::Tensor* hidden_rev_array = nullptr; // keep compiler quiet
	lten::Tensor* z_t_array = nullptr;
	lten::Tensor* hc_t_array = nullptr;
	lten::Tensor* w_hc_h_array = nullptr;
	lten::Tensor* r_t_array = nullptr;
	lten::Tensor* tmp_5_array = nullptr;

	lten::Tensor* z_t_rev_array = nullptr;
	lten::Tensor* hc_t_rev_array = nullptr;
	lten::Tensor* w_hc_t_rev_array = nullptr;
	lten::Tensor* r_t_rev_array = nullptr;
	lten::Tensor* tmp_5_rev_array = nullptr;
	lten::Tensor* extra_workspace;

	lten::device device_type;
	int device_index;

	//::FillBuffer<Dtype>(top_gradient_ptr->GetDataPtr(), top_gradient_ptr->GetNumels(), 0.362f);

	device_type = parent_ptr->get_device();
	device_index = parent_ptr->get_device_index();

	gru_ptr = static_cast<lten::GRU*>(parent_ptr->misc_ptr1_);

	hidden_array = gru_ptr->get_hidden_array();

	z_t_array = gru_ptr->get_z_t_array();
	hc_t_array = gru_ptr->get_hc_t_array();
	w_hc_h_array = gru_ptr->get_w_hc_h_array();
	r_t_array = gru_ptr->get_r_t_array();
	tmp_5_array = gru_ptr->get_tmp_5_array();


	bidirectional = gru_ptr->is_bidirectional();
	use_bias = gru_ptr->is_using_bias();

	if (bidirectional)
	{
		z_t_rev_array = gru_ptr->get_z_t_rev_array();
		hc_t_rev_array = gru_ptr->get_hc_t_rev_array();
		w_hc_t_rev_array = gru_ptr->get_w_hc_t_rev_array();
		r_t_rev_array = gru_ptr->get_r_t_rev_array();
		tmp_5_rev_array = gru_ptr->get_tmp_5_rev_array();
	}

	extra_workspace = gru_ptr->get_extra_workspace();

	lten::Tensor* weights_w = gru_ptr->get_w_weights();
	lten::Tensor* weights_u = gru_ptr->get_u_weights();
	lten::Tensor* bias_w = gru_ptr->get_w_bias();
	lten::Tensor* bias_u = gru_ptr->get_u_bias();
	Dtype* weights_w_gradients;
	Dtype* weights_u_gradients;
	Dtype* bias_w_gradients = nullptr;
	Dtype* bias_u_gradients = nullptr;

	lten::Tensor* weights_w_rev = gru_ptr->get_w_rev_weights();
	lten::Tensor* weights_u_rev = gru_ptr->get_u_rev_weights();
	lten::Tensor* bias_w_rev = gru_ptr->get_w_rev_bias();
	lten::Tensor* bias_u_rev = gru_ptr->get_u_rev_bias();
	Dtype* weights_w_rev_gradients = nullptr;
	Dtype* weights_u_rev_gradients = nullptr;
	Dtype* bias_w_rev_gradients = nullptr;
	Dtype* bias_u_rev_gradients = nullptr;


	weights_w_gradients = weights_w->get_gradients_mdarray<Dtype>()->GetDataPtr();
	weights_u_gradients = weights_u->get_gradients_mdarray<Dtype>()->GetDataPtr();
	if (use_bias)
	{
		bias_w_gradients = bias_w->get_gradients_mdarray<Dtype>()->GetDataPtr();
		bias_u_gradients = bias_u->get_gradients_mdarray<Dtype>()->GetDataPtr();
	}

	if (bidirectional)
	{
		weights_w_rev_gradients = weights_w_rev->get_gradients_mdarray<Dtype>()->GetDataPtr();
		weights_u_rev_gradients = weights_u_rev->get_gradients_mdarray<Dtype>()->GetDataPtr();
		if (use_bias)
		{
			bias_w_rev_gradients = bias_w_rev->get_gradients_mdarray<Dtype>()->GetDataPtr();
			bias_u_rev_gradients = bias_u_rev->get_gradients_mdarray<Dtype>()->GetDataPtr();
		}
	}


	if (bidirectional)
	{
		hidden_rev_array = gru_ptr->get_hidden_rev_array();
	}


	input_dim = gru_ptr->get_input_dim();
	hidden_dim = gru_ptr->get_hidden_dim();
	sequence_len = parent_ptr->misc1_;

	batches = bottom_gradient_ptr->GetSizes()[0];


	hidden_gradient = static_cast<Dtype*>(extra_workspace[0].get_data_ptr());
	gradient_0 = static_cast<Dtype*>(extra_workspace[1].get_data_ptr());
	gradient_1 = static_cast<Dtype*>(extra_workspace[2].get_data_ptr());
	gradient_2 = static_cast<Dtype*>(extra_workspace[3].get_data_ptr());
	gradient_3 = static_cast<Dtype*>(extra_workspace[4].get_data_ptr());
	gradient_4 = static_cast<Dtype*>(extra_workspace[5].get_data_ptr());
	scratch_gradient = static_cast<Dtype*>(extra_workspace[6].get_data_ptr());

	//---------------------------------------------------
	/*
	MultiDimArray<Dtype> temp2;
	temp2.Allocate({ 1, 1, hidden_dim });
	Dtype* temp2_data;
	temp2_data = temp2.GetDataPtr();

	MultiDimArray<Dtype> temp3;
	temp3.Allocate({ 1, hidden_dim, hidden_dim * 3 });
	Dtype* temp3_data;
	temp3_data = temp3.GetDataPtr();

	MultiDimArray<Dtype> temp4;
	temp4.Allocate({ 1, 1, input_dim });
	Dtype* temp4_data;
	temp4_data = temp4.GetDataPtr();


	MultiDimArray<Dtype> temp5;
	temp5.Allocate({ 1, input_dim, hidden_dim * 3 });
	Dtype* temp5_data;
	temp5_data = temp5.GetDataPtr();
	*/
	//---------------------------------------------------

	if (lten::CPU == device_type)
	{
		FillBuffer<Dtype>(bottom_gradient_ptr->GetDataPtr(), bottom_gradient_ptr->GetNumels(), 0);
		for (i = 0; i < batches; i++)
		{
			memset(hidden_gradient, 0, sizeof(float) * hidden_dim);
			memset(scratch_gradient, 0, sizeof(float) * hidden_dim * 3);
			memset(gradient_3, 0, sizeof(float) * hidden_dim * 3);

			for (j = sequence_len - 1; j >= 0; j--)
			{
				input = children_ptr_array[0]->get_mdarray()->GetDataPtr() + i * sequence_len * input_dim + j * input_dim;

				if (bidirectional)
				{
					top_gradient = top_gradient_ptr->GetDataPtr() + i * sequence_len * 2 * hidden_dim + j * 2 * hidden_dim; // 'slice' out gradient for this sequence step
				}
				else
				{
					top_gradient = top_gradient_ptr->GetDataPtr() + i * sequence_len * hidden_dim + j * hidden_dim; // 'slice' out gradient for this sequence step
				}

				bottom_gradient = bottom_gradient_ptr->GetDataPtr() + i * sequence_len * input_dim + input_dim * j;


				cpu_axpy(hidden_dim, static_cast<Dtype>(1), hidden_gradient, top_gradient, gradient_0);

				cpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_0, static_cast<Dtype*>(z_t_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_1); // tmp5 * zt backward_left_child (combined with -1 * hc_t)

				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[6]*/
				cpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_1, hidden_gradient);
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------

				//std::cout << grad_1 << "\n";

				cpu_axpy(hidden_dim, static_cast<Dtype>(1), gradient_0, gradient_1, gradient_1); // sum gradients flowing into tmp3.tanh, sending correct gradients to everthing downstream

				cpu_tanh_backward(gradient_2, gradient_1, static_cast<Dtype*>(hc_t_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // h_c_t tanh backward

				//std::cout << grad_2 << "\n";

				cpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(w_hc_h_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_3); // r_t * w_hc_h backward_left_child 

				cpu_sig_backward(gradient_3, gradient_3, static_cast<Dtype*>(r_t_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // r_t sigmoid backward

				//std::cout << grad_3 << "\n";
				// temp_data branches to two paths here

				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[0]*/
				if (use_bias)
				{
					cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_w_gradients, bias_w_gradients);
				}

				weights = static_cast<Dtype*>(weights_w->get_data_ptr());

				M = 1;
				N = hidden_dim;
				K = hidden_dim * 3;

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(1), hidden_gradient);
				
				//std::cout << temp2 << "\n";

				M = hidden_dim;
				N = hidden_dim * 3;
				K = 1;

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), gradient_3, static_cast<Dtype>(1), weights_w_gradients);

				//std::cout << temp3 << "\n";
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------


				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[1]*/
				if (use_bias)
				{
					cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_u_gradients, bias_u_gradients);
				}

				weights = static_cast<Dtype*>(weights_u->get_data_ptr());

				M = 1;
				N = input_dim;
				K = hidden_dim * 3;

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(1), bottom_gradient);

				M = input_dim;
				N = hidden_dim * 3;
				K = 1;

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, gradient_3, static_cast<Dtype>(1), weights_u_gradients);
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------



				cpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(r_t_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient + 2 * hidden_dim); // r_t * w_hc_h backward_right_child (+ 2 * hidden_dim because this for temp_1[2])
				memset(scratch_gradient, 0, sizeof(Dtype) * 2 * hidden_dim);
				//std::cout << scratch_grad << "\n";
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[2]*/
				if (use_bias)
				{
					cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_gradients, bias_w_gradients);
				}

				weights = static_cast<Dtype*>(weights_w->get_data_ptr());

				M = 1;
				N = hidden_dim;
				K = hidden_dim * 3;

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), hidden_gradient);


				M = hidden_dim;
				N = hidden_dim * 3;
				K = 1;

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(1), weights_w_gradients);
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------

				memcpy(scratch_gradient + 2 * hidden_dim, gradient_2, sizeof(Dtype) * hidden_dim); // (+ 2 * hidden_dim because this is for temp_0[2])
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[3]*/
				if (use_bias)
				{
					cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_gradients, bias_u_gradients);
				}

				weights = static_cast<Dtype*>(weights_u->get_data_ptr());

				M = 1;
				N = input_dim;
				K = hidden_dim * 3;

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), bottom_gradient);


				M = input_dim;
				N = hidden_dim * 3;
				K = 1;

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(1), weights_u_gradients);
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------

				cpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_0, static_cast<Dtype*>(tmp_5_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient); // tmp5 * zt backward_right_child
				//std::cout << scratch_grad << "\n";

				cpu_sig_backward(scratch_gradient + hidden_dim, scratch_gradient, static_cast<Dtype*>(z_t_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // + hidden_dim for correct placement in temp_1[1]
				memset(scratch_gradient + 2 * hidden_dim, 0, sizeof(Dtype) * hidden_dim); // clear residual stuff
				memset(scratch_gradient, 0, sizeof(Dtype) * hidden_dim); // clear residual stuff
				//std::cout << scratch_grad << "\n";

				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[4]*/
				if (use_bias)
				{
					cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_gradients, bias_w_gradients);
				}

				weights = static_cast<Dtype*>(weights_w->get_data_ptr());

				M = 1;
				N = hidden_dim;
				K = hidden_dim * 3;

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), hidden_gradient);


				M = hidden_dim;
				N = hidden_dim * 3;
				K = 1;

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(1), weights_w_gradients);
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------


				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*[5]*/
				if (use_bias)
				{
					cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_gradients, bias_u_gradients);
				}

				weights = static_cast<Dtype*>(weights_u->get_data_ptr());

				M = 1;
				N = input_dim;
				K = hidden_dim * 3;

				cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), bottom_gradient);


				M = input_dim;
				N = hidden_dim * 3;
				K = 1;

				cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(1), weights_u_gradients);
				//--------------------------------------------------------------------------------------------------------------------------------------------------------------
			}
		}


		if (bidirectional)
		{
			for (i = 0; i < batches; i++)
			{
				memset(hidden_gradient, 0, sizeof(float) * hidden_dim);
				memset(scratch_gradient, 0, sizeof(float) * hidden_dim * 3);
				memset(gradient_3, 0, sizeof(float) * hidden_dim * 3);

				for (j = sequence_len - 1; j >= 0; j--)
				{
					input = children_ptr_array[0]->get_mdarray()->GetDataPtr() + i * sequence_len * input_dim + (sequence_len - j - 1) * input_dim;

					top_gradient = top_gradient_ptr->GetDataPtr() + i * sequence_len * 2 * hidden_dim + (sequence_len - j - 1) * 2 * hidden_dim + hidden_dim; // 'slice' out gradient for this sequence step

					bottom_gradient = bottom_gradient_ptr->GetDataPtr() + i * sequence_len * input_dim + (sequence_len - j - 1) * input_dim;


					cpu_axpy(hidden_dim, static_cast<Dtype>(1), hidden_gradient, top_gradient, gradient_0);

					cpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_0, static_cast<Dtype*>(z_t_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_1); // tmp5 * zt backward_left_child (combined with -1 * hc_t)

					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[6]*/
					cpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_1, hidden_gradient);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------


					//std::cout << grad_1 << "\n";

					cpu_axpy(hidden_dim, static_cast<Dtype>(1), gradient_0, gradient_1, gradient_1); // sum gradients flowing into tmp3.tanh, sending correct gradients to everthing downstream

					cpu_tanh_backward(gradient_2, gradient_1, static_cast<Dtype*>(hc_t_rev_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // h_c_t tanh backward

					//std::cout << grad_2 << "\n";

					cpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(w_hc_t_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_3); // r_t * w_hc_h backward_left_child 

					cpu_sig_backward(gradient_3, gradient_3, static_cast<Dtype*>(r_t_rev_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // r_t sigmoid backward

					//std::cout << grad_3 << "\n";
					// temp_data branches to two paths here

					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[0]*/
					if (use_bias)
					{
						cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_w_rev_gradients, bias_w_rev_gradients);
					}

					weights = static_cast<Dtype*>(weights_w_rev->get_data_ptr());

					M = 1;
					N = hidden_dim;
					K = hidden_dim * 3;

					cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(1), hidden_gradient);


					M = hidden_dim;
					N = hidden_dim * 3;
					K = 1;

					cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_rev_array[i * sequence_len + j].get_data_ptr()), gradient_3, static_cast<Dtype>(1), weights_w_rev_gradients);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------


					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[1]*/
					if (use_bias)
					{
						cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_u_rev_gradients, bias_u_rev_gradients);
					}

					weights = static_cast<Dtype*>(weights_u_rev->get_data_ptr());

					M = 1;
					N = input_dim;
					K = hidden_dim * 3;

					cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(1), bottom_gradient);


					M = input_dim;
					N = hidden_dim * 3;
					K = 1;

					cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, gradient_3, static_cast<Dtype>(1), weights_u_rev_gradients);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------



					cpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(r_t_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient + 2 * hidden_dim); // r_t * w_hc_h backward_right_child (+ 2 * hidden_dim because this for temp_1[2])
					memset(scratch_gradient, 0, sizeof(Dtype) * 2 * hidden_dim);
					//std::cout << scratch_grad << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[2]*/
					if (use_bias)
					{
						cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_rev_gradients, bias_w_rev_gradients);
					}

					weights = static_cast<Dtype*>(weights_w_rev->get_data_ptr());

					M = 1;
					N = hidden_dim;
					K = hidden_dim * 3;

					cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), hidden_gradient);

					M = hidden_dim;
					N = hidden_dim * 3;
					K = 1;

					cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_rev_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(1), weights_w_rev_gradients);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------

					memcpy(scratch_gradient + 2 * hidden_dim, gradient_2, sizeof(Dtype) * hidden_dim); // (+ 2 * hidden_dim because this is for temp_0[2])
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[3]*/
					if (use_bias)
					{
						cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_rev_gradients, bias_u_rev_gradients);
					}

					weights = static_cast<Dtype*>(weights_u_rev->get_data_ptr());

					M = 1;
					N = input_dim;
					K = hidden_dim * 3;

					cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), bottom_gradient);


					M = input_dim;
					N = hidden_dim * 3;
					K = 1;

					cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(1), weights_u_rev_gradients);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------

					cpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_0, static_cast<Dtype*>(tmp_5_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient); // tmp5 * zt backward_right_child
					//std::cout << scratch_grad << "\n";

					cpu_sig_backward(scratch_gradient + hidden_dim, scratch_gradient, static_cast<Dtype*>(z_t_rev_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // + hidden_dim for correct placement in temp_1[1]
					memset(scratch_gradient + 2 * hidden_dim, 0, sizeof(Dtype) * hidden_dim); // clear residual stuff
					memset(scratch_gradient, 0, sizeof(Dtype) * hidden_dim); // clear residual stuff
					//std::cout << scratch_grad << "\n";

					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[4]*/
					if (use_bias)
					{
						cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_rev_gradients, bias_w_rev_gradients);
					}

					weights = static_cast<Dtype*>(weights_w_rev->get_data_ptr());

					M = 1;
					N = hidden_dim;
					K = hidden_dim * 3;

					cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), hidden_gradient);


					M = hidden_dim;
					N = hidden_dim * 3;
					K = 1;

					cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_rev_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(1), weights_w_rev_gradients);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------


					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[5]*/
					if (use_bias)
					{
						cpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_rev_gradients, bias_u_rev_gradients);
					}

					weights = static_cast<Dtype*>(weights_u_rev->get_data_ptr());

					M = 1;
					N = input_dim;
					K = hidden_dim * 3;

					cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), bottom_gradient);

					M = input_dim;
					N = hidden_dim * 3;
					K = 1;

					cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(1), weights_u_rev_gradients);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				}
			}
		}

	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			ZeroMemoryOnGPU(bottom_gradient_ptr->GetDataPtr(), sizeof(Dtype) * bottom_gradient_ptr->GetNumels());

			cublasStatus_t status;
			cublasHandle_t hCuBlas;

			hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(device_index);

			float alpha = 1.0f;
			float beta = 1.0f;


			for (i = 0; i < batches; i++)
			{
				ZeroMemoryOnGPU(hidden_gradient, sizeof(float) * hidden_dim);
				ZeroMemoryOnGPU(scratch_gradient, sizeof(float) * hidden_dim * 3);
				ZeroMemoryOnGPU(gradient_3, sizeof(float) * hidden_dim * 3);

				for (j = sequence_len - 1; j >= 0; j--)
				{
					input = children_ptr_array[0]->get_mdarray()->GetDataPtr() + i * sequence_len * input_dim + j * input_dim;

					if (bidirectional)
					{
						top_gradient = top_gradient_ptr->GetDataPtr() + i * sequence_len * 2 * hidden_dim + j * 2 * hidden_dim; // 'slice' out gradient for this sequence step
					}
					else
					{
						top_gradient = top_gradient_ptr->GetDataPtr() + i * sequence_len * hidden_dim + j * hidden_dim; // 'slice' out gradient for this sequence step
					}

					bottom_gradient = bottom_gradient_ptr->GetDataPtr() + i * sequence_len * input_dim + input_dim * j;


					gpu_axpy(hidden_dim, static_cast<Dtype>(1), hidden_gradient, top_gradient, gradient_0);

					gpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_0, static_cast<Dtype*>(z_t_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_1); // tmp5 * zt backward_left_child (combined with -1 * hc_t)

					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[6]*/
					gpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_1, hidden_gradient);
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------

					//std::cout << grad_1 << "\n";

					gpu_axpy(hidden_dim, static_cast<Dtype>(1), gradient_0, gradient_1, gradient_1); // sum gradients flowing into tmp3.tanh, sending correct gradients to everthing downstream

					gpu_tanh_backward(gradient_2, gradient_1, static_cast<Dtype*>(hc_t_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // h_c_t tanh backward

					//std::cout << grad_2 << "\n";

					gpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(w_hc_h_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_3); // r_t * w_hc_h backward_left_child 

					gpu_sig_backward(gradient_3, gradient_3, static_cast<Dtype*>(r_t_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // r_t sigmoid backward

					//std::cout << grad_3 << "\n";
					// temp_data branches to two paths here

					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[0]*/
					if (use_bias)
					{
						gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_w_gradients, bias_w_gradients);
					}

					weights = static_cast<Dtype*>(weights_w->get_data_ptr());

					M = 1;
					N = hidden_dim;
					K = hidden_dim * 3;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)gradient_3, (int)K, &beta, (float*)hidden_gradient, (int)N);

					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(1), hidden_gradient);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(0), temp2_data);

					//std::cout << temp2 << "\n";

					M = hidden_dim;
					N = hidden_dim * 3;
					K = 1;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)gradient_3, (int)N, (float*)(hidden_array[i * sequence_len + j].get_data_ptr()), (int)M, &beta, (float*)weights_w_gradients, (int)N);

					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), gradient_3, static_cast<Dtype>(1), weights_w_gradients);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), gradient_3, static_cast<Dtype>(0), temp3_data);

					//std::cout << temp3 << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------


					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[1]*/
					if (use_bias)
					{
						gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_u_gradients, bias_u_gradients);
					}

					weights = static_cast<Dtype*>(weights_u->get_data_ptr());

					M = 1;
					N = input_dim;
					K = hidden_dim * 3;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)gradient_3, (int)K, &beta, (float*)bottom_gradient, (int)N);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(1), bottom_gradient);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), gradient_3, weights, static_cast<Dtype>(0), temp4_data);
					//std::cout << temp4 << "\n";


					M = input_dim;
					N = hidden_dim * 3;
					K = 1;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)gradient_3, (int)N, (float*)input, (int)M, &beta, (float*)weights_u_gradients, (int)N);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, gradient_3, static_cast<Dtype>(1), weights_u_gradients);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, gradient_3, static_cast<Dtype>(0), temp5_data);
					//std::cout << temp5 << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------



					gpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(r_t_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient + 2 * hidden_dim); // r_t * w_hc_h backward_right_child (+ 2 * hidden_dim because this for temp_1[2])
					ZeroMemoryOnGPU(scratch_gradient, sizeof(Dtype) * 2 * hidden_dim);
					//std::cout << scratch_grad << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[2]*/
					if (use_bias)
					{
						gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_gradients, bias_w_gradients);
					}

					weights = static_cast<Dtype*>(weights_w->get_data_ptr());

					M = 1;
					N = hidden_dim;
					K = hidden_dim * 3;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)hidden_gradient, (int)N);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), hidden_gradient);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(0), temp2_data);

					//std::cout << temp2 << "\n";


					M = hidden_dim;
					N = hidden_dim * 3;
					K = 1;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)(hidden_array[i * sequence_len + j].get_data_ptr()), (int)M, &beta, (float*)weights_w_gradients, (int)N);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(1), weights_w_gradients);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(0), temp3_data);

					//std::cout << temp3 << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------

					GPUToGPUCopy(scratch_gradient + 2 * hidden_dim, gradient_2, sizeof(Dtype) * hidden_dim); // (+ 2 * hidden_dim because this is for temp_0[2])
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[3]*/
					if (use_bias)
					{
						gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_gradients, bias_u_gradients);
					}

					weights = static_cast<Dtype*>(weights_u->get_data_ptr());

					M = 1;
					N = input_dim;
					K = hidden_dim * 3;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)bottom_gradient, (int)N);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), bottom_gradient);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(0), temp4_data);
					//std::cout << temp4 << "\n";



					M = input_dim;
					N = hidden_dim * 3;
					K = 1;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)input, (int)M, &beta, (float*)weights_u_gradients, (int)N);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(1), weights_u_gradients);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(0), temp5_data);
					//std::cout << temp5 << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------

					gpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_0, static_cast<Dtype*>(tmp_5_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient); // tmp5 * zt backward_right_child
					//std::cout << scratch_grad << "\n";

					gpu_sig_backward(scratch_gradient + hidden_dim, scratch_gradient, static_cast<Dtype*>(z_t_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // + hidden_dim for correct placement in temp_1[1]
					ZeroMemoryOnGPU(scratch_gradient + 2 * hidden_dim, sizeof(Dtype) * hidden_dim); // clear residual stuff
					ZeroMemoryOnGPU(scratch_gradient, sizeof(Dtype) * hidden_dim); // clear residual stuff
					//std::cout << scratch_grad << "\n";

					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[4]*/
					if (use_bias)
					{
						gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_gradients, bias_w_gradients);
					}

					weights = static_cast<Dtype*>(weights_w->get_data_ptr());

					M = 1;
					N = hidden_dim;
					K = hidden_dim * 3;


					status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)hidden_gradient, (int)N);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), hidden_gradient);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(0), temp2_data);

					//std::cout << temp2 << "\n";


					M = hidden_dim;
					N = hidden_dim * 3;
					K = 1;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)(hidden_array[i * sequence_len + j].get_data_ptr()), (int)M, &beta, (float*)weights_w_gradients, (int)N);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(1), weights_w_gradients);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), (Dtype*)(hidden_array[i * sequence_len + j].get_data_ptr()), scratch_gradient, static_cast<Dtype>(0), temp3_data);

					//std::cout << temp3 << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------


					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					/*[5]*/
					if (use_bias)
					{
						gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_gradients, bias_u_gradients);
					}

					weights = static_cast<Dtype*>(weights_u->get_data_ptr());

					M = 1;
					N = input_dim;
					K = hidden_dim * 3;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)bottom_gradient, (int)N);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(1), bottom_gradient);
					//cpu_gemm(false, true, M, N, K, static_cast<Dtype>(1), scratch_gradient, weights, static_cast<Dtype>(0), temp4_data);
					//std::cout << temp4 << "\n";



					M = input_dim;
					N = hidden_dim * 3;
					K = 1;

					status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)input, (int)M, &beta, (float*)weights_u_gradients, (int)N);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(1), weights_u_gradients);
					//cpu_gemm(true, false, M, N, K, static_cast<Dtype>(1), input, scratch_gradient, static_cast<Dtype>(0), temp5_data);
					//std::cout << temp5 << "\n";
					//--------------------------------------------------------------------------------------------------------------------------------------------------------------
				}
			}


			if (bidirectional)
			{
				for (i = 0; i < batches; i++)
				{
					ZeroMemoryOnGPU(hidden_gradient, sizeof(float) * hidden_dim);
					ZeroMemoryOnGPU(scratch_gradient, sizeof(float) * hidden_dim * 3);
					ZeroMemoryOnGPU(gradient_3, sizeof(float) * hidden_dim * 3);

					for (j = sequence_len - 1; j >= 0; j--)
					{
						input = children_ptr_array[0]->get_mdarray()->GetDataPtr() + i * sequence_len * input_dim + (sequence_len - j - 1) * input_dim;

						top_gradient = top_gradient_ptr->GetDataPtr() + i * sequence_len * 2 * hidden_dim + (sequence_len - j - 1) * 2 * hidden_dim + hidden_dim; // 'slice' out gradient for this sequence step

						bottom_gradient = bottom_gradient_ptr->GetDataPtr() + i * sequence_len * input_dim + (sequence_len - j - 1) * input_dim;


						gpu_axpy(hidden_dim, static_cast<Dtype>(1), hidden_gradient, top_gradient, gradient_0);

						gpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_0, static_cast<Dtype*>(z_t_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_1); // tmp5 * zt backward_left_child (combined with -1 * hc_t)

						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[6]*/
						gpu_mul(hidden_dim, static_cast<Dtype>(-1), gradient_1, hidden_gradient);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------


						gpu_axpy(hidden_dim, static_cast<Dtype>(1), gradient_0, gradient_1, gradient_1); // sum gradients flowing into tmp3.tanh, sending correct gradients to everthing downstream

						gpu_tanh_backward(gradient_2, gradient_1, static_cast<Dtype*>(hc_t_rev_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // h_c_t tanh backward


						gpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(w_hc_t_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), gradient_3); // r_t * w_hc_h backward_left_child 

						gpu_sig_backward(gradient_3, gradient_3, static_cast<Dtype*>(r_t_rev_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // r_t sigmoid backward

						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[0]*/
						if (use_bias)
						{
							gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_w_rev_gradients, bias_w_rev_gradients);
						}

						weights = static_cast<Dtype*>(weights_w_rev->get_data_ptr());

						M = 1;
						N = hidden_dim;
						K = hidden_dim * 3;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)gradient_3, (int)K, &beta, (float*)hidden_gradient, (int)N);


						M = hidden_dim;
						N = hidden_dim * 3;
						K = 1;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)gradient_3, (int)N, (float*)(hidden_rev_array[i * sequence_len + j].get_data_ptr()), (int)M, &beta, (float*)weights_w_rev_gradients, (int)N);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------


						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[1]*/
						if (use_bias)
						{
							gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), gradient_3, bias_u_rev_gradients, bias_u_rev_gradients);
						}

						weights = static_cast<Dtype*>(weights_u_rev->get_data_ptr());

						M = 1;
						N = input_dim;
						K = hidden_dim * 3;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)gradient_3, (int)K, &beta, (float*)bottom_gradient, (int)N);

						M = input_dim;
						N = hidden_dim * 3;
						K = 1;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)gradient_3, (int)N, (float*)input, (int)M, &beta, (float*)weights_u_rev_gradients, (int)N);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------



						gpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_2, static_cast<Dtype*>(r_t_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient + 2 * hidden_dim); // r_t * w_hc_h backward_right_child (+ 2 * hidden_dim because this for temp_1[2])
						ZeroMemoryOnGPU(scratch_gradient, sizeof(Dtype) * 2 * hidden_dim);
						//std::cout << scratch_grad << "\n";
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[2]*/
						if (use_bias)
						{
							gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_rev_gradients, bias_w_rev_gradients);
						}

						weights = static_cast<Dtype*>(weights_w_rev->get_data_ptr());

						M = 1;
						N = hidden_dim;
						K = hidden_dim * 3;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)hidden_gradient, (int)N);


						M = hidden_dim;
						N = hidden_dim * 3;
						K = 1;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)(hidden_rev_array[i * sequence_len + j].get_data_ptr()), (int)M, &beta, (float*)weights_w_rev_gradients, (int)N);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------

						GPUToGPUCopy(scratch_gradient + 2 * hidden_dim, gradient_2, sizeof(Dtype) * hidden_dim); // (+ 2 * hidden_dim because this is for temp_0[2])
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[3]*/
						if (use_bias)
						{
							gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_rev_gradients, bias_u_rev_gradients);
						}

						weights = static_cast<Dtype*>(weights_u_rev->get_data_ptr());

						M = 1;
						N = input_dim;
						K = hidden_dim * 3;
						
						status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)bottom_gradient, (int)N);


						M = input_dim;
						N = hidden_dim * 3;
						K = 1;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)input, (int)M, &beta, (float*)weights_u_rev_gradients, (int)N);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------

						gpu_mul(hidden_dim, static_cast<Dtype>(1), gradient_0, static_cast<Dtype*>(tmp_5_rev_array[i * sequence_len + j].get_data_ptr()), static_cast<Dtype>(0), scratch_gradient); // tmp5 * zt backward_right_child

						gpu_sig_backward(scratch_gradient + hidden_dim, scratch_gradient, static_cast<Dtype*>(z_t_rev_array[i * sequence_len + j].get_data_ptr()), hidden_dim); // + hidden_dim for correct placement in temp_1[1]
						ZeroMemoryOnGPU(scratch_gradient + 2 * hidden_dim, sizeof(Dtype) * hidden_dim); // clear residual stuff
						ZeroMemoryOnGPU(scratch_gradient, sizeof(Dtype) * hidden_dim); // clear residual stuff

						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[4]*/
						if (use_bias)
						{
							gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_w_rev_gradients, bias_w_rev_gradients);
						}

						weights = static_cast<Dtype*>(weights_w_rev->get_data_ptr());

						M = 1;
						N = hidden_dim;
						K = hidden_dim * 3;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)hidden_gradient, (int)N);

						M = hidden_dim;
						N = hidden_dim * 3;
						K = 1;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)(hidden_rev_array[i * sequence_len + j].get_data_ptr()), (int)M, &beta, (float*)weights_w_rev_gradients, (int)N);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------


						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
						/*[5]*/
						if (use_bias)
						{
							gpu_axpy(hidden_dim * 3, static_cast<Dtype>(1), scratch_gradient, bias_u_rev_gradients, bias_u_rev_gradients);
						}

						weights = static_cast<Dtype*>(weights_u_rev->get_data_ptr());

						M = 1;
						N = input_dim;
						K = hidden_dim * 3;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (float*)weights, (int)K, (float*)scratch_gradient, (int)K, &beta, (float*)bottom_gradient, (int)N);


						M = input_dim;
						N = hidden_dim * 3;
						K = 1;

						status = cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha, (float*)scratch_gradient, (int)N, (float*)input, (int)M, &beta, (float*)weights_u_rev_gradients, (int)N);
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------
					}
				}
			}
#else
LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}
}


#ifdef USE_CUDA
template<typename Dtype>
void gru_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	cudnnHandle_t cudnnHandle;
	lten::GRU_CUDNN* gru_cudnn;
	int sequence_len;

	gru_cudnn = static_cast<lten::GRU_CUDNN*>(parent_ptr->misc_ptr1_);


	cudnnHandle = lten::CUDA_globlas::singleton()->get_cudnn_handle(0);


	sequence_len = static_cast<int>(parent_ptr->misc1_);
	

	cudnnErrCheck(cudnnRNNBackwardData(cudnnHandle,
		*(gru_cudnn->get_rnnDesc()),
		sequence_len,
		gru_cudnn->get_yDesc(),
		parent_ptr->get_data_ptr(), //y
		gru_cudnn->get_dyDesc(),
		top_gradient_ptr->GetDataPtr(), //dy
		*(gru_cudnn->get_hyDesc()),
		nullptr,
		*(gru_cudnn->get_dcyDesc()),
		nullptr,
		*(gru_cudnn->get_wDesc()),
		gru_cudnn->get_w(),
		*(gru_cudnn->get_hxDesc()),
		nullptr,
		*(gru_cudnn->get_cxDesc()),
		nullptr,
		gru_cudnn->get_dxDesc(),
		bottom_gradient_ptr->GetDataPtr(),
		*(gru_cudnn->get_dhxDesc()),
		nullptr,
		*(gru_cudnn->get_dcxDesc()),
		nullptr,
		gru_cudnn->get_workspace(),
		gru_cudnn->get_workSize(),
		gru_cudnn->get_reserveSpace(),
		gru_cudnn->get_reserveSize()));



	cudnnErrCheck(cudnnRNNBackwardWeights(cudnnHandle,
		*(gru_cudnn->get_rnnDesc()),
		sequence_len,
		gru_cudnn->get_xDesc(), // xDesc,
		children_ptr_array[0]->get_data_ptr(), //x
		*(gru_cudnn->get_hxDesc()),
		nullptr, //hx
		gru_cudnn->get_yDesc(),
		parent_ptr->get_data_ptr(), //y
		gru_cudnn->get_workspace(),
		gru_cudnn->get_workSize(),
		(*gru_cudnn->get_dwDesc()),
		gru_cudnn->get_dw(),
		gru_cudnn->get_reserveSpace(),
		gru_cudnn->get_reserveSize()));
	
}
#endif

template<typename Dtype>
void transpose_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	int dim1;
	int dim2;

	lten::device device_type;

	device_type = parent_ptr->get_device();

	dim1 = static_cast<int>(parent_ptr->misc1_);
	dim2 = static_cast<int>(parent_ptr->misc2_);

	assert(child_index == 0);
	assert(top_gradient_ptr->GetNumels() == bottom_gradient_ptr->GetNumels());


	if (lten::CPU == device_type)
	{
		MultiDimArray<Dtype> top_transpose;

		top_transpose = top_gradient_ptr->transpose(dim2, dim1);

		memcpy(bottom_gradient_ptr->GetDataPtr(), top_transpose.GetDataPtr(), sizeof(Dtype) * top_transpose.GetNumels());
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			const uint64_t* strides;
			const uint64_t* strides_transp;

			strides = top_gradient_ptr->GetStrides();
			strides_transp = bottom_gradient_ptr->GetStrides();

			
			gpu_transpose(static_cast<Dtype*>(top_gradient_ptr->GetDataPtr()), static_cast<Dtype*>(bottom_gradient_ptr->GetDataPtr()), dim1, dim2,
				(int)strides[dim1], (int)strides[dim1 - 1], (int)strides[dim2], (int)strides[dim2 - 1],
				(int)strides_transp[dim1], (int)strides_transp[dim1 - 1], (int)strides_transp[dim2], (int)strides_transp[dim2 - 1], top_gradient_ptr->GetNumels());


#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

}


template<typename Dtype>
void nll_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	MultiDimArray<Dtype>* operand1_md_array;
	MultiDimArray<Dtype>* operand2_md_array;
	Dtype scale;
	lten::device device_type;
	uint64_t N;


	scale = static_cast<Dtype>(-1) / static_cast<Dtype>(children_ptr_array[0]->get_sizes()[0]);

	operand1_md_array = children_ptr_array[0]->get_mdarray();
	operand2_md_array = children_ptr_array[1]->get_mdarray();

	device_type = parent_ptr->get_device();

	N = bottom_gradient_ptr->GetNumels();

	if (lten::CPU == device_type)
	{
		if (top_gradient_ptr)
		{
			scale *= static_cast<Dtype>(top_gradient_ptr->GetDataPtr()[0]);
		}

		if (child_index == 0) // operand 1
		{
			Dtype* op2_data = operand2_md_array->GetDataPtr();
			Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();

			cpu_mul(N, static_cast<Dtype>(scale), op2_data, bottom_data);
		}
		else
		{
			Dtype* op1_data = operand1_md_array->GetDataPtr();
			Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();

			cpu_mul(N, static_cast<Dtype>(scale), op1_data, bottom_data);
		}
	}
	else
	{
		if (lten::GPU == device_type)
		{
#ifdef USE_CUDA
			if (top_gradient_ptr)
			{
				scale *= static_cast<Dtype>(top_gradient_ptr->GetDataPtr()[0]);
			}

			if (child_index == 0) // operand 1
			{
				Dtype* op2_data = operand2_md_array->GetDataPtr();
				Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();
				Dtype* top_data = nullptr;

				gpu_mul(N, static_cast<Dtype>(scale), op2_data, bottom_data);
			}
			else
			{
				Dtype* op1_data = operand1_md_array->GetDataPtr();
				Dtype* bottom_data = bottom_gradient_ptr->GetDataPtr();

				gpu_mul(N, static_cast<Dtype>(scale), op1_data, bottom_data);
			}
#else
			LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
		}
		else
		{
			LTEN_ERR("Invalid tesor data type");
		}
	}

}

#ifdef USE_CUDA
template<typename Dtype>
void bn_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	cudnnHandle_t cudnnHandle;
	lten::BatchNorm_CUDNN* bn_cudnn;
	float alpha;
	float beta;
	float alpha_wts;
	float beta_wts;

	bn_cudnn = static_cast<lten::BatchNorm_CUDNN*>(parent_ptr->misc_ptr1_);


	cudnnHandle = lten::CUDA_globlas::singleton()->get_cudnn_handle(0);

	alpha = 1.0f;
	beta = 0.0f;
	alpha_wts = 1.0f;
	beta_wts = 1.0f;

	cudnnBatchNormalizationBackward(cudnnHandle, bn_cudnn->get_mode(), &alpha, &beta, &alpha_wts, &beta_wts,
		bn_cudnn->get_inputDesc(),
		children_ptr_array[0]->get_data_ptr(), //x
		bn_cudnn->get_inputDesc(),
		top_gradient_ptr->GetDataPtr(), //dy
		bn_cudnn->get_inputDesc(),
		bottom_gradient_ptr->GetDataPtr(), // dx
		bn_cudnn->get_scale_bias_Desc(),
		bn_cudnn->get_weights()->get_data_ptr(),
		bn_cudnn->get_weights()->get_grad_ptr(),
		bn_cudnn->get_bias()->get_grad_ptr(),
		bn_cudnn->get_epsilon(),
		nullptr,
		nullptr);

}
#endif

#ifdef USE_CUDA
template<typename Dtype>
void pooling_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr)
{
	cudnnHandle_t cudnnHandle;
	lten::pooling_CUDNN* pool_cudnn;
	float alpha;
	float beta;

	pool_cudnn = static_cast<lten::pooling_CUDNN*>(parent_ptr->misc_ptr1_);

	cudnnHandle = lten::CUDA_globlas::singleton()->get_cudnn_handle(0);

	alpha = 1.0f;
	beta = 0.0f;

	cudnnErrCheck(cudnnPoolingBackward(cudnnHandle,
		pool_cudnn->get_poolingDesc(),
		&alpha,
		pool_cudnn->get_outputDesc(),
		parent_ptr->get_data_ptr(), //y
		pool_cudnn->get_outputDesc(),
		top_gradient_ptr->GetDataPtr(), // dy
		pool_cudnn->get_inputDesc(),
		children_ptr_array[0]->get_data_ptr(), //x
		&beta,
		pool_cudnn->get_inputDesc(),
		bottom_gradient_ptr->GetDataPtr())); // dx


}
#endif

template void matmul_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void sub_array_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void exp_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void add_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void sub_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void mul_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void div_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void max_backward1(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void sum_backward1(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void max_backward2(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void sum_backward2(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void squeeze_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void unsqueeze_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void reshape_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void log_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void sig_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void tanh_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void scalar_mul_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void conv2_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void conv2_cudnn_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void fc_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void relu_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void dropout_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void softmax_cudnn_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void gru_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void gru_cudnn_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void transpose_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void nll_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void bn_cudnn_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);
template void pooling_cudnn_backward(MultiDimArray<float>* bottom_gradient_ptr, MultiDimArray<float>* top_gradient_ptr, lten::TensorImpl<float>** children_ptr_array, int child_index, lten::TensorImpl<float>* parent_ptr);


template void matmul_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void sub_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void exp_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void add_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void sub_array_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void mul_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void div_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void max_backward1(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void sum_backward1(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void max_backward2(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void sum_backward2(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void squeeze_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void unsqueeze_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void reshape_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void log_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void sig_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void tanh_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void scalar_mul_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void conv2_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void conv2_cudnn_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void fc_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void relu_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void dropout_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void softmax_cudnn_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void gru_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void gru_cudnn_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void transpose_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void nll_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void bn_cudnn_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);
template void pooling_cudnn_backward(MultiDimArray<int>* bottom_gradient_ptr, MultiDimArray<int>* top_gradient_ptr, lten::TensorImpl<int>** children_ptr_array, int child_index, lten::TensorImpl<int>* parent_ptr);


template void matmul_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void sub_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void exp_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void add_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void sub_array_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void mul_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void div_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void max_backward1(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void sum_backward1(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void max_backward2(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void sum_backward2(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void squeeze_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void unsqueeze_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void reshape_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void log_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void sig_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void tanh_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void scalar_mul_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void conv2_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void conv2_cudnn_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void fc_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void relu_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void dropout_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void softmax_cudnn_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void gru_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void gru_cudnn_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void transpose_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void nll_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void bn_cudnn_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
template void pooling_cudnn_backward(MultiDimArray<uint8_t>* bottom_gradient_ptr, MultiDimArray<uint8_t>* top_gradient_ptr, lten::TensorImpl<uint8_t>** children_ptr_array, int child_index, lten::TensorImpl<uint8_t>* parent_ptr);
