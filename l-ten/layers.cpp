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
		// -this just A.matmul(B.transpose())
		//----------------------------------------------------------------------
		TensorOps options;
		TensorImpl<float>* resultImpl;
		cublasStatus_t status;
		cublasHandle_t hCuBlas;
		int ndims;
		int numels;
		int num_batches;
		uint64_t result_dims[MAX_DIMS];
		bool broadcast_required;
		int i;
		MultiDimArray<float>* A_md_array;


		ndims = A.get_ndims();
		numels = A.get_numels();

		A_md_array = A.get_mdarray<float>();

		broadcast_required = A_md_array->check_broadcast_required(B.get_sizes(), result_dims, true);

		options.data_type = A.get_data_type();
		options.device_index = A.get_device_index();
		options.device_type = A.get_device();
				
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);
		resultImpl->allocate(result_dims, ndims, &options);

		num_batches = resultImpl->get_numels() / (result_dims[ndims - 1] * result_dims[ndims - 2]);

		if (ndims != ndims_ || numels != numels_) // fast (but weak) test to ensure nothing has changed
		{
			FreeMemoryOnGPU(pa_.buffer);
			FreeMemoryOnGPU(oa_.buffer);

			OFFSET_ARRAYS oa_cpu;

			AllocateMemoryOnGPU(&pa_.buffer, 3 * sizeof(float*) * num_batches, false);
			pa_.a_array = (void**)pa_.buffer;
			pa_.b_array = (void**)pa_.buffer + num_batches;
			pa_.c_array = (void**)pa_.buffer + 2 * num_batches;


			AllocateMemoryOnGPU(&oa_.buffer, 3 * sizeof(uint32_t*) * num_batches, false);
			oa_.a_array = (uint32_t*)oa_.buffer;
			oa_.b_array = (uint32_t*)oa_.buffer + num_batches;
			oa_.c_array = (uint32_t*)oa_.buffer + 2 * num_batches;

			oa_cpu.buffer = new uint32_t[3 * num_batches];
			oa_cpu.a_array = (uint32_t*)oa_cpu.buffer;
			oa_cpu.b_array = (uint32_t*)oa_cpu.buffer + num_batches;
			oa_cpu.c_array = (uint32_t*)oa_cpu.buffer + 2 * num_batches;

			MultiDimArray<float>* B_md_array = B.get_mdarray<float>();
			MultiDimArray<float>* C_md_array = resultImpl->get_mdarray();

			float* A_base = (float*)A.get_data_ptr();
			float* B_base = (float*)B.get_data_ptr();
			float* C_base = (float*)result->get_data_ptr();

			int index = 0;
			md_array_dim_iterator it(result_dims, ndims - 2);
			for (auto higher_indices : it)
			{
				float* result_data = C_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				float* lhs_data = A_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);
				float* rhs_data = B_md_array->GetDataPtr(higher_indices, ndims - 2, broadcast_required);

				oa_cpu.a_array[index] = lhs_data - A_base;
				oa_cpu.b_array[index] = rhs_data - B_base;
				oa_cpu.c_array[index] = result_data - C_base;

				index++;
			}
			
			CopyDataToGPU(oa_.buffer, oa_cpu.buffer, 3 * sizeof(uint32_t) * num_batches);
		}

		set_addresses((float*)A.get_data_ptr(), (float*)B.get_data_ptr(), (float*)resultImpl->get_data_ptr(), &pa_, &oa_, num_batches);

		ndims_ = ndims;
		numels_ = numels;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(0);

		float alpha = 1.0f;
		float beta = 0.0f;

		uint64_t M;
		uint64_t N;
		uint64_t K;

		int lda;
		int ldb;
		int ldc;

		M = result_dims[ndims_ - 2];
		N = result_dims[ndims_ - 1];
		K = A.get_sizes()[ndims_ - 1];

		lda = static_cast<int>(K);
		ldb = static_cast<int>(K);
		ldc = static_cast<int>(N);


		status = cublasSgemmBatched(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
			(float**)pa_.b_array, lda, (float**)pa_.a_array, ldb, &beta, (float**)pa_.c_array, ldc, num_batches);


		return Tensor(result);
	}


	Tensor Pseudo_Einsum_2::forward(Tensor& A, Tensor& B)
	{
		//----------------------------------------------------------------------
		// hack implementation of torch::einsum("bythwc, wkc->bythwk", { A, B })
		//----------------------------------------------------------------------
		uint64_t result_dims[MAX_DIMS] = { 56, 2, 8, 56, 1, 1, 7 };
		uint32_t permutation[MAX_DIMS] = { 1, 5, 2, 3, 0, 6, 4 };
		TensorOps options;
		TensorImpl<float>* resultImpl;
		int ndims;

		options.data_type = A.get_data_type();
		options.device_index = A.get_device_index();
		options.device_type = A.get_device();

		ndims = 7;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);
		resultImpl->allocate(result_dims, ndims, &options);


		float alpha = 1.0f;
		float beta = 0.0f;

		int lda;
		int ldb;
		int ldc;

		cublasStatus_t status;
		cublasHandle_t hCuBlas;

		hCuBlas = lten::CUDA_globlas::singleton()->get_cublas_handle(0);

		status = cublasSgemmStridedBatched(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, 7, 896, 96, &alpha, (float*)B.get_data_ptr(), 96, 672, (float*)A.get_data_ptr(), 5376, 96, &beta, (float*)resultImpl->get_data_ptr(), 7, 6272, 56);


		return Tensor(result);
	}
} // namespace lten