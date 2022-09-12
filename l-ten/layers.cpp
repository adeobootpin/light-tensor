#include <random>
#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "utils.h"


namespace lten {
	Tensor relu(Tensor& input)
	{
		uint64_t len;
		uint64_t i;
		TensorOps options;
		float* src;
		float* dst;

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(input.get_sizes(), input.get_ndims(), &options);

		len = input.get_numels();
		src = static_cast<float*>(input.get_data_ptr());
		dst = static_cast<float*>(resultImpl->get_data_ptr());

		if (CPU == options.device_type)
		{
			for (i = 0; i < len; i++)
			{
				if (src[i] > 0)
				{
					dst[i] = src[i];
				}
				else
				{
					dst[i] = 0;
				}
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_relu(dst, src, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(relu_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);
	}


	Tensor softmax(Tensor& input, int dim)
	{
		Tensor exps;
		Tensor max;
		Tensor input_minus_max;
		Tensor sum;

		max = input.max(dim);
		max = max.squeeze(0);
		max = max.unsqueeze(dim);

		input_minus_max = input - max;

		exps = input_minus_max.exp();

		sum = exps.sum(dim);
		sum = sum.squeeze(0);
		sum = sum.unsqueeze(dim);

		return exps.div(sum);
	}


	Tensor log_softmax(Tensor& input, int dim)
	{
		Tensor exps;
		Tensor max;
		Tensor input_minus_max;
		Tensor sum;

		max = input.max(dim);
		max = max.squeeze(0);
		max = max.unsqueeze(dim);

		input_minus_max = input - max;

		exps = input_minus_max.exp();

		sum = exps.sum(dim);

		sum = sum.squeeze(0);
		sum = sum.unsqueeze(dim);

		return exps.div(sum).log();
	}


#ifdef USE_CUDA
	bool softmax_CUDNN::init()
	{
		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));

		return true;
	}

	Tensor softmax_CUDNN::forward(Tensor& input)
	{
		int ndims;
		TensorOps options;
		int n, c, h, w;
		float alpha;
		float beta;
		const uint64_t* dims;
		cudnnHandle_t cudnnHandle;

		dims = input.get_sizes();

		ndims = input.get_ndims();
		
		if (ndims != 4)
		{
			LTEN_ERR("softmax_CUDNN requires tensors with exactly 4 dimensions (NCHW)");
		}
		
		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("softmax_CUDNN only supports the GPU device type");
		}

		n = static_cast<int>(dims[0]);
		c = static_cast<int>(dims[1]);
		h = static_cast<int>(dims[2]);
		w = static_cast<int>(dims[3]);
		
		const int nDims = 4;
		int dimA[nDims] = { n,c,h,w };
		int strideA[nDims] = { c*h*w, h*w, w, 1 };


		cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, 4, dimA, strideA));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(outputDesc_, CUDNN_DATA_FLOAT, 4, dimA, strideA));

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);


		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(dims, ndims, &options);

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);
		alpha = 1.0f;
		beta = 0;

		cudnnErrCheck(cudnnSoftmaxForward(cudnnHandle, algo_, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			inputDesc_,
			input.get_data_ptr(),
			&beta,
			outputDesc_,
			resultImpl->get_data_ptr()));



		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(softmax_cudnn_backward);
		resultImpl->set_autograd(true);
		

		return Tensor(result);

	}
#endif

	Tensor mse_loss(Tensor& input, Tensor& target)
	{
		Tensor diff;
		uint64_t numels;

		numels = input.get_numels();

		diff = input - target;
		diff = (diff * diff).sum();
		return (1.0f / numels) * diff;
	}


	// like libtorch, assumes input is result of log_softmax
	Tensor nll_loss(Tensor& input, Tensor& target)
	{
		uint64_t dims[1];
		uint64_t len;
		uint64_t i;
		TensorOps options;
		float* input_data;
		float* target_data;
		float* loss;
		float val;

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		dims[0] = 1;
		resultImpl->allocate(dims, 1, &options);

		len = input.get_numels();
		input_data = static_cast<float*>(input.get_data_ptr());
		target_data = static_cast<float*>(target.get_data_ptr());
		loss = static_cast<float*>(resultImpl->get_data_ptr());

		if (CPU == options.device_type)
		{
			val = 0;
			for (i = 0; i < len; i++)
			{
				val += input_data[i] * target_data[i];
			}

			*loss = val * (-1.0f / input.get_sizes()[0]);

		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_nll(input_data, target_data, loss, len, input.get_sizes()[0]);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(target.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(nll_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);

	}


	bool Pseudo_Einsum_1::init()
	{
		return true;
	}

	Tensor Pseudo_Einsum_1::forward(Tensor& A, Tensor& B)
	{
		//----------------------------------------------------------------------
		// hack implementation of torch::einsum("bythwc, hkc->bythwk", { A, B })
		// -this is just A.matmul(B.transpose())
		// however, permutations are required to use cublasSgemmStridedBatched
		//----------------------------------------------------------------------

		TensorImpl<float>* resultImpl;
		uint64_t result_dims[MAX_DIMS] = { 2, 1, 8, 56, 56, 7 };
		uint64_t permuted_result_dims[MAX_DIMS] = { 56, 2, 8, 56, 1, 1, 7 };
		uint32_t permutations_a[MAX_DIMS] = { 3, 0, 1, 2, 4, 5, 6 };
		uint32_t permutations_c[MAX_DIMS] = { 1, 2, 0, 3, 4, 5, 6 };
		uint64_t permuted_dims_a[MAX_DIMS];
		uint64_t permuted_strides_a[MAX_DIMS];
		uint64_t unsqeezed_dims_a[MAX_DIMS];
		const uint64_t* dims_a;
		int ndims;
		TensorOps options;
		int i;
		int index;
		MultiDimArray<float> unsqueezed_a;

		
		ndims = A.get_ndims();
		if (ndims != 6)
		{
			LTEN_ERR("Pseudo_Einsum_1 expects A to have 6 dimensions");
		}

		//
		//A = A.unsqueeze(5);
		//
		dims_a = A.get_sizes();
		ndims++;
		index = 0;
		for (i = 0; i < ndims; i++) 
		{
			if (i == 5)
			{
				unsqeezed_dims_a[i] = 1;
			}
			else
			{
				unsqeezed_dims_a[i] = dims_a[index++];
			}		
		}
		unsqueezed_a.Allocate(unsqeezed_dims_a, ndims, (float*)A.get_data_ptr(), false);



		if (!permuted_a_buffer_ || A.get_numels() != numels_a_)
		{
			numels_a_ = A.get_numels();
			FreeMemoryOnGPU(permuted_a_buffer_);
			AllocateMemoryOnGPU(&permuted_a_buffer_, sizeof(float) * numels_a_, false);

			FreeMemoryOnGPU(scratch_a_buffer_);
			AllocateMemoryOnGPU(&scratch_a_buffer_, sizeof(float) * numels_a_, false);	
		}


		options.data_type = A.get_data_type();
		options.device_index = A.get_device_index();
		options.device_type = A.get_device();

		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);
		resultImpl->allocate(result_dims, ndims - 1, &options);

		if (!scratch_c_buffer_ || resultImpl->get_numels() != numels_c_)
		{
			numels_c_ = resultImpl->get_numels();
			FreeMemoryOnGPU(scratch_c_buffer_);
			AllocateMemoryOnGPU(&scratch_c_buffer_, sizeof(float) * numels_c_, false);
		}

		GetPermutationStridesAndeDims(unsqueezed_a.GetSizes(), permuted_dims_a, permuted_strides_a, permutations_a, ndims);

		gpu_permute((float*)permuted_a_buffer_, unsqueezed_a.GetDataPtr(), ndims, unsqueezed_a.GetNumels(), permuted_strides_a, unsqueezed_a.GetStrides(), permutations_a);


		float alpha = 1.0f;
		float beta = 0.0f;

		cublasStatus_t status;
		cublasHandle_t hCuBlas;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(0);

		status = cublasSgemmStridedBatched(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, 7, 896, 96, &alpha, (float*)B.get_data_ptr(), 96, 672, (float*)permuted_a_buffer_, 96, 86016, &beta, (float*)scratch_c_buffer_, 7, 6272, 56);


		MultiDimArray<float> temp;
		temp.Allocate({ 56, 2, 8, 56, 1, 1, 7 }, (float*)scratch_c_buffer_, false);
		uint64_t permuted_strides_c[MAX_DIMS];
		
		GetPermutationStridesAndeDims(temp.GetSizes(), nullptr, permuted_strides_c, permutations_c, ndims);

		gpu_permute((float*)resultImpl->get_data_ptr(), (float*)scratch_c_buffer_, ndims, resultImpl->get_numels(), permuted_strides_c, temp.GetStrides(), permutations_c);


		if (is_training_ || static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())->autograd_on() || static_cast<TensorImpl<float>*>(B.get_smart_ptr().get_real_object())->autograd_on())
		{
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())));
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(B.get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(pseudo_einsum1_backward);
			resultImpl->set_autograd(true);
		}

		return Tensor(result);
	}


	Tensor Pseudo_Einsum_2::forward(Tensor& A, Tensor& B)
	{
		//--------------------------------------------------------------------------------
		// hack implementation of torch::einsum("bythwc, wkc->bythwk", { A, B })
		// -this is A.unsqeeze(ndims-2).matmul(B.unsqueeze(0).transpose(ndims-2, ndims-1))
		// e.g. A = 2, 1, 8, 56, 56, 96
		//      B = 1, 1, 1, 56,  7, 96
		// reshaped to:
		//      A = 2, 1, 8, 56, 56, 1, 96
		//      B = 1, 1, 1, 1,  56, 7, 96
		// then
		//      C = A.matmul(B.transpose(5,6))
		//--------------------------------------------------------------------------------
		int ndims;
		uint64_t required_size;
		uint64_t numels;
		TensorOps options;
		TensorImpl<float>* resultImpl;
		TensorImpl<float> scratch;
		uint64_t scratch_dims[MAX_DIMS] = { 56, 2, 1, 8, 56, 7 }; // TODO set this up dynamically
		uint64_t result_dims[MAX_DIMS];
		uint32_t result_permutation[] = { 1, 2, 3, 4, 0, 5 }; // TODO set this up dynamically (move 0 to desired location and slide rest left)
		int i;

		options.data_type = A.get_data_type();
		options.device_index = A.get_device_index();
		options.device_type = A.get_device();


		ndims = A.get_ndims();

		numels = 1;
		for (i = 0; i < ndims; i++)
		{
			numels *= scratch_dims[i];
		}

		if (is_training_ || static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())->autograd_on())
		{
			required_size = std::max(numels, A.get_numels()); // need size A for backward proc
		}
		else
		{
			required_size = numels;
		}
		

		if (!scratch_buffer_ || required_size > scratch_buffer_size_) 
		{
			FreeMemoryOnGPU(scratch_buffer_);
			AllocateMemoryOnGPU(&scratch_buffer_, sizeof(float) * required_size, false);
			scratch_buffer_size_ = required_size;
		}

		scratch.allocate_from_buffer(scratch_dims, ndims, scratch_buffer_, false);

		float alpha = 1.0f;
		float beta = 0.0f;

		cublasStatus_t status;
		cublasHandle_t hCuBlas;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(0);

		status = cublasSgemmStridedBatched(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, 7, 896, 96, &alpha, (float*)B.get_data_ptr(), 96, 672, (float*)A.get_data_ptr(), 5376, 96, &beta, (float*)scratch_buffer_, 7, 6272, 56);

		for (i = 0; i < ndims; i++)
		{
			result_dims[i] = scratch_dims[result_permutation[i]];
		}

		
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);
		resultImpl->allocate(result_dims, ndims, &options);

		gpu_permute((float*)resultImpl->get_data_ptr(), (float*)scratch_buffer_, ndims, resultImpl->get_numels(), resultImpl->get_strides(), scratch.get_strides(), result_permutation);


		if (is_training_ || static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())->autograd_on() || static_cast<TensorImpl<float>*>(B.get_smart_ptr().get_real_object())->autograd_on())
		{
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())));
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(B.get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(pseudo_einsum2_backward);
			resultImpl->set_autograd(true);
		}

		return Tensor(result);
		
		/*
		//----------------------------------------------------------------------
		// hack implementation of torch::einsum("bythwc, wkc->bythwk", { A, B })
		// HACKHACK: only works for these dims!!!
		//----------------------------------------------------------------------
		uint64_t result_dims[MAX_DIMS];
		uint64_t scratch_dims[MAX_DIMS] = { 56, 2, 8, 56, 1, 1, 7 };
		uint32_t permutation[MAX_DIMS] = { 1, 5, 2, 3, 0, 6, 4 };
		TensorOps options;
		TensorImpl<float> scratch;
		TensorImpl<float>* resultImpl;
		int ndims;
		int i;

		options.data_type = A.get_data_type();
		options.device_index = A.get_device_index();
		options.device_type = A.get_device();

		if (!scratch_buffer_)
		{
			AllocateMemoryOnGPU(&scratch_buffer_, sizeof(float) * 56 * 2 * 8 * 56 * 1 * 1 * 7, false);
		}

		ndims = 7;
		scratch.allocate_from_buffer(scratch_dims, ndims, scratch_buffer_, false);


		float alpha = 1.0f;
		float beta = 0.0f;

		int lda;
		int ldb;
		int ldc;

		cublasStatus_t status;
		cublasHandle_t hCuBlas;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(options.device_index);

		status = cublasSgemmStridedBatched(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, 7, 896, 96, &alpha, (float*)B.get_data_ptr(), 96, 672, (float*)A.get_data_ptr(), 5376, 96, &beta, (float*)scratch_buffer_, 7, 6272, 56);

		//-----------------------------------------------------------------
		// perform permutation
		//-----------------------------------------------------------------
		for (i = 0; i < ndims; i++)
		{
			result_dims[i] = scratch_dims[permutation[i]];
		}

		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);
		resultImpl->allocate(result_dims, ndims, &options);


		gpu_permute((float*)resultImpl->get_data_ptr(), (float*)scratch_buffer_, ndims, resultImpl->get_numels(), resultImpl->get_strides(), scratch.get_strides(), permutation);


		//
		//HACKHACK we know last dim is 1 so get rid of it
		//
		resultImpl->get_mdarray()->SetMemoryOwnership(false); // prevent deletion of data pointer during following call to allocate_from_buffer
		resultImpl->allocate_from_buffer(result_dims, ndims - 1, resultImpl->get_data_ptr(), true, &options); 


		if (is_training_ || static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())->autograd_on() || static_cast<TensorImpl<float>*>(B.get_smart_ptr().get_real_object())->autograd_on())
		{
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(A.get_smart_ptr().get_real_object())));
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(B.get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(pseudo_einsum2_backward);
			resultImpl->set_autograd(true);
		}


		return Tensor(result);
		*/
	}
} // namespace lten