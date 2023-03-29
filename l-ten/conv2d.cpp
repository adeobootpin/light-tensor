#include <random>
#include <iostream>
#include "lten.h"
#include "im_col.h"
#include "utils.h"


namespace lten {
	bool Conv2d::init()
	{
		float* raw_data_ptr;
		uint64_t numels;
		uint64_t i;
		TensorOps options;

		options.data_type = FLOAT32;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ (uint64_t)channels_out_ , (uint64_t)channels_in_,  (uint64_t)(kernel_h_ * kernel_w_) }, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ 1, channels_out_, 1, 1 }, &options);
			bias_ptr_->set_autograd(true);
		}

		std::default_random_engine generator;
		float k = 1.0f / (channels_in_ * (kernel_h_ * kernel_w_));
		std::uniform_real_distribution<float> distribution(-sqrtf(k), sqrtf(k));

		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		numels = weight_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		if (use_bias_)
		{
			raw_data_ptr = (float*)bias_ptr_->get_data_ptr();
			numels = bias_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
			}
		}

		return true;
	}

	Tensor Conv2d::forward(Tensor& input)
	{
		Tensor conv;
		uint64_t height_in;
		uint64_t width_in;
		uint64_t batches_in;
		uint64_t height_out;
		uint64_t width_out;
		int64_t stride_in;
		int64_t stride_out;
		uint64_t M, N, K;
		int i;
		const uint64_t* input_dims_ptr;
		dtype data_type;
		TensorOps options;
		MultiDimArray<float> col_buffer_md_array;
		MultiDimArray<float> weights_md_array;

		data_type = input.get_smart_ptr()->get_data_type();
		if (data_type != FLOAT32)
		{
			LTEN_ERR("Conv2d is only supported for FLOAT32 tensors");
		}

		if (input.get_device() != weight_ptr_->get_device() ||
			input.get_device_index() != weight_ptr_->get_device_index() ||
			input.get_data_type() != weight_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}


		if (input.get_ndims() != 4) // only NCHW supported for 2d conv
		{
			LTEN_ERR("Conv2d requires tensors with exactly 4 dimensions (NCHW)");
		}

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		input_dims_ptr = input.get_sizes();
		batches_in = input_dims_ptr[0];
		height_in = input_dims_ptr[2];
		width_in = input_dims_ptr[3];


		height_out = (height_in + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
		width_out = (width_in + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate({ batches_in, channels_out_, height_out, width_out }, &options);


		stride_in = channels_in_ * height_in * width_in;
		stride_out = channels_out_ * height_out * width_out;

		if (!col_buffer_ptr_)
		{
			col_buffer_ptr_ = new Tensor;
			*col_buffer_ptr_ = AllocateTensor({ 1, 1, channels_in_ * kernel_h_ * kernel_w_, height_out * width_out }, &options); // replace initializer list here for speed
		}
		else
		{
			if (col_buffer_ptr_->get_numels() < channels_in_ * kernel_h_ * kernel_w_ * height_out * width_out)
			{
				*col_buffer_ptr_ = AllocateTensor({ 1, 1, channels_in_ * kernel_h_ * kernel_w_, height_out * width_out }, &options); // replace initializer list here for speed
			}
		}


		weights_md_array.Allocate({ 1, 1, channels_out_, channels_in_ * kernel_h_ * kernel_w_ }, (float*)weight_ptr_->get_data_ptr(), false);
		col_buffer_md_array.Allocate({ 1, 1, channels_in_ * kernel_h_ * kernel_w_, height_out * width_out }, (float*)col_buffer_ptr_->get_data_ptr(), false);

		M = channels_out_;
		N = height_out * width_out;
		K = channels_in_ * kernel_h_ * kernel_w_;

		if (CPU == options.device_type)
		{
			for (i = 0; i < batches_in; i++)
			{
				im2col_cpu((float*)input.get_data_ptr() + i * stride_in, channels_in_, height_in, width_in, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, (float*)col_buffer_ptr_->get_data_ptr());

				cpu_gemm(false, false, M, N, K, static_cast<float>(1), weights_md_array.GetDataPtr(), col_buffer_md_array.GetDataPtr(), static_cast<float>(0), resultImpl->get_mdarray()->GetDataPtr() + i * stride_out);
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				float alpha;
				float beta;
				int lda;
				int ldb;
				int ldc;

				cublasHandle_t hCuBlas;

				hCuBlas = CUDA_globlas::singleton()->get_cublas_handle(options.device_index);

				alpha = 1.0f;
				beta = 0.0f;
				lda = static_cast<int>(N);
				ldb = static_cast<int>(K);
				ldc = static_cast<int>(N);

				for (i = 0; i < batches_in; i++)
				{
					im2col_gpu((float*)input.get_data_ptr() + i * stride_in, (int)channels_in_, (int)height_in, (int)width_in, (int)kernel_h_, (int)kernel_w_, (int)pad_h_, (int)pad_w_, (int)stride_h_, (int)stride_w_, 1, 1, (float*)col_buffer_ptr_->get_data_ptr());
					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha,
						col_buffer_md_array.GetDataPtr(), lda, weights_md_array.GetDataPtr(), ldb, &beta, resultImpl->get_mdarray()->GetDataPtr() + i * stride_out, ldc));
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


		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(weight_ptr_->get_smart_ptr().get_real_object())));
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(conv2_backward);
		resultImpl->set_autograd(true);

		if (use_bias_)
		{
			return (Tensor(result) + *bias_ptr_);
		}
		else
		{
			return Tensor(result);
		}
	}

	void Conv2d::clear_gradients()
	{
		if (weight_ptr_)
		{
			weight_ptr_->clear_gradients();
		}
		if (bias_ptr_)
		{
			bias_ptr_->clear_gradients();
		}
	}

	std::vector<Tensor*> Conv2d::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}

	void Conv2d::to(device target_device, int target_device_index)
	{
		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(target_device, target_device_index);
		}

		if (bias_ptr_)
		{
			*bias_ptr_ = bias_ptr_->to(target_device, target_device_index);
		}
	}

	//------------------------------------------------------------------------------------------ quantized conv2d------------------------------------
	bool Conv2d_q::init()
	{
		TensorOps options;

		workspace_ = new Tensor;

		options.data_type = UINT8;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ (uint64_t)channels_out_ , (uint64_t)channels_in_,  (uint64_t)(kernel_h_ * kernel_w_) }, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			options.data_type = INT32;
			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ 1, channels_out_, 1, 1 }, &options);
			bias_ptr_->set_autograd(true);
		}

		//
		// no weight initialization here (assume weights will be initialized externally by quantization setup)
		//

		return true;
	}

	Tensor Conv2d_q::forward(Tensor& input)
	{
		Tensor conv;
		uint64_t height_in;
		uint64_t width_in;
		uint64_t batches_in;
		uint64_t height_out;
		uint64_t width_out;
		int64_t stride_in;
		int64_t stride_out;
		uint64_t M, N, K;
		int i;
		const uint64_t* input_dims_ptr;
		dtype data_type;
		TensorOps options;
		MultiDimArray<uint8_t> col_buffer_md_array;
		MultiDimArray<uint8_t> weights_md_array;
		QuantizationParams qparams[3];

		data_type = input.get_smart_ptr()->get_data_type();
		if (data_type != UINT8)
		{
			LTEN_ERR("Conv2d_q is only supported for UINT8 tensors");
		}

		if (input.get_device() != weight_ptr_->get_device() ||
			input.get_device_index() != weight_ptr_->get_device_index() ||
			input.get_data_type() != weight_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}


		if (input.get_ndims() != 4) // only NCHW supported for 2d conv
		{
			LTEN_ERR("Conv2d_q requires tensors with exactly 4 dimensions (NCHW)");
		}

		TensorImpl<uint8_t>* resultImpl;
		resultImpl = new TensorImpl<uint8_t>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		input_dims_ptr = input.get_sizes();
		batches_in = input_dims_ptr[0];
		height_in = input_dims_ptr[2];
		width_in = input_dims_ptr[3];


		height_out = (height_in + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
		width_out = (width_in + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate({ batches_in, channels_out_, height_out, width_out }, &options);


		stride_in = channels_in_ * height_in * width_in;
		stride_out = channels_out_ * height_out * width_out;

		if (!col_buffer_ptr_)
		{
			col_buffer_ptr_ = new Tensor;
			*col_buffer_ptr_ = AllocateTensor({ 1, 1, channels_in_ * kernel_h_ * kernel_w_, height_out * width_out }, &options); // replace initializer list here for speed
		}
		else
		{
			if (col_buffer_ptr_->get_numels() < channels_in_ * kernel_h_ * kernel_w_ * height_out * width_out)
			{
				*col_buffer_ptr_ = AllocateTensor({ 1, 1, channels_in_ * kernel_h_ * kernel_w_, height_out * width_out }, &options); // replace initializer list here for speed
			}
		}


		weights_md_array.Allocate({ 1, 1, channels_out_, channels_in_ * kernel_h_ * kernel_w_ }, (uint8_t*)weight_ptr_->get_data_ptr(), false);
		col_buffer_md_array.Allocate({ 1, 1, channels_in_ * kernel_h_ * kernel_w_, height_out * width_out }, (uint8_t*)col_buffer_ptr_->get_data_ptr(), false);

		M = channels_out_;
		N = height_out * width_out;
		K = channels_in_ * kernel_h_ * kernel_w_;

		qparams[1] = qparams_in_;
		qparams[0] = qparams_wt_;
		qparams[2] = qparams_out_;

		if (N > max_input_size_)
		{
			options.data_type = INT32;
			*workspace_ = AllocateTensor({ M + N }, &options);
			max_input_size_ = N;
		}

		if (CPU == options.device_type)
		{
			for (i = 0; i < batches_in; i++)
			{
				im2col_cpu((uint8_t*)input.get_data_ptr() + i * stride_in, channels_in_, height_in, width_in, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, (uint8_t*)col_buffer_ptr_->get_data_ptr());

				if (bias_ptr_)
				{
					quantized_matmul(false, false, M, N, K, 1, weights_md_array.GetDataPtr(), col_buffer_md_array.GetDataPtr(), 0, resultImpl->get_mdarray()->GetDataPtr() + i * stride_out, qparams, static_cast<int*>(bias_ptr_->get_data_ptr()), static_cast<int*>(workspace_->get_data_ptr()));
				}
				else
				{
					quantized_matmul(false, false, M, N, K, 1, weights_md_array.GetDataPtr(), col_buffer_md_array.GetDataPtr(), 0, resultImpl->get_mdarray()->GetDataPtr() + i * stride_out, qparams, nullptr, static_cast<int*>(workspace_->get_data_ptr()));
				}
			}
		}
		else
		{
			LTEN_ERR("Conv2d_q only supports the CPU device type");  // only CPU for now
		}


		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<uint8_t>*>(weight_ptr_->get_smart_ptr().get_real_object())));
		resultImpl->add_child(*(static_cast<TensorImpl<uint8_t>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(conv2_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);
	}

	void Conv2d_q::clear_gradients()
	{
		if (weight_ptr_)
		{
			weight_ptr_->clear_gradients();
		}
		if (bias_ptr_)
		{
			bias_ptr_->clear_gradients();
		}
	}

	std::vector<Tensor*> Conv2d_q::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}


	void Conv2d_q::set_qparams_params(QuantizationParams* qparams, int count)
	{
		if (qparams && count)
		{
			qparams_wt_ = qparams[0];
		}
	}



	//------------------------------------------------------------------------------------------ CUDNN conv2d----------------------------------------
#ifdef USE_CUDA
	bool conv2d_CUDNN::init()
	{
		TensorOps options;
		float* raw_data_ptr;
		uint64_t numels;
		uint64_t i;
		cudnnHandle_t cudnnHandle;
		int n, c, h, w;

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&biasDesc_));
		cudnnErrCheck(cudnnCreateFilterDescriptor(&wtDesc_));
		cudnnErrCheck(cudnnCreateConvolutionDescriptor(&convDesc_));


		cudnnErrCheck(cudnnSetConvolution2dDescriptor(convDesc_, static_cast<int>(pad_h_), static_cast<int>(pad_w_), static_cast<int>(stride_h_), static_cast<int>(stride_w_), 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
		cudnnErrCheck(cudnnSetFilter4dDescriptor(wtDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, static_cast<int>(channels_out_), static_cast<int>(channels_in_), static_cast<int>(kernel_h_), static_cast<int>(kernel_w_)));
		cudnnErrCheck(cudnnSetTensor4dDescriptor(biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, static_cast<int>(channels_out_), 1, 1));


		cudnnErrCheck(cudnnSetTensor4dDescriptor(inputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(batch_size_), static_cast<int>(channels_in_), static_cast<int>(height_in_), static_cast<int>(width_in_)));
		cudnnErrCheck(cudnnGetConvolution2dForwardOutputDim(convDesc_, inputDesc_, wtDesc_, &n, &c, &h, &w));
		cudnnErrCheck(cudnnSetTensor4dDescriptor(outputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
		cudnnErrCheck(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
		cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, algo_, &workspace_size_));
		AllocateMemoryOnGPU(&workspace_, workspace_size_, false);


		cudnnErrCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc_, outputDesc_, convDesc_, wtDesc_, bwf_algo_, &bwf_workspace_size_));
		AllocateMemoryOnGPU(&bwf_workspace_, bwf_workspace_size_, false);


		cudnnErrCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wtDesc_, outputDesc_, convDesc_, inputDesc_, bwd_algo_, &bwd_workspace_size_));
		AllocateMemoryOnGPU(&bwd_workspace_, bwd_workspace_size_, false);


		output_dims_[0] = n;
		output_dims_[1] = c;
		output_dims_[2] = h;
		output_dims_[3] = w;

		options.data_type = FLOAT32;
		options.alloc_gradient_buffer = true;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ (uint64_t)channels_out_ , (uint64_t)channels_in_,  (uint64_t)kernel_h_,  (uint64_t)kernel_w_ }, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ 1, channels_out_, 1, 1 }, &options);
			bias_ptr_->set_autograd(true);
		}

		std::random_device generator;
		//std::default_random_engine generator;
		float k = 1.0f / (channels_in_ * (kernel_h_ + kernel_w_));
		std::uniform_real_distribution<float> distribution(-sqrtf(k), sqrtf(k));

		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		numels = weight_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		if (use_bias_)
		{
			raw_data_ptr = (float*)bias_ptr_->get_data_ptr();
			numels = bias_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
			}
		}


		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(GPU, 0);
		}

		if (bias_ptr_)
		{
			*bias_ptr_ = bias_ptr_->to(GPU, 0);
		}


		return true;
	}


	Tensor conv2d_CUDNN::forward(Tensor& input)
	{
		const uint64_t* dims;
		TensorOps options;
		int n, c, h, w;
		int ndims;
		float alpha;
		float beta;
		cudnnStatus_t status;
		cudnnHandle_t cudnnHandle;


		dims = input.get_sizes();

		ndims = input.get_ndims();

		if (ndims != 4)
		{
			LTEN_ERR("conv2d_CUDNN requires tensors with exactly 4 dimensions (NCHW)");
		}

		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("Invalid tensor device type");
		}

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		if (channels_in_ != dims[1]) // this means reinitializing weights, a no-no
		{
			LTEN_ERR("Dimension 1 must be equal to the number of input channels");
		}

		if (batch_size_ != dims[0] || height_in_ != dims[2] || width_in_ != dims[3])
		{
			LTEN_CUDNN_CHECK_2(cudnnSetTensor4dDescriptor(inputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]), static_cast<int>(dims[3])));
			LTEN_CUDNN_CHECK_2(cudnnGetConvolution2dForwardOutputDim(convDesc_, inputDesc_, wtDesc_, &n, &c, &h, &w));
			LTEN_CUDNN_CHECK_2(cudnnSetTensor4dDescriptor(outputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, algo_, &workspace_size_));
			FreeMemoryOnGPU(workspace_);
			AllocateMemoryOnGPU(&workspace_, workspace_size_, false);


			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc_, outputDesc_, convDesc_, wtDesc_, bwf_algo_, &bwf_workspace_size_));
			FreeMemoryOnGPU(bwf_workspace_);
			AllocateMemoryOnGPU(&bwf_workspace_, bwf_workspace_size_, false);


			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wtDesc_, outputDesc_, convDesc_, inputDesc_, bwd_algo_, &bwd_workspace_size_));
			FreeMemoryOnGPU(bwd_workspace_);
			AllocateMemoryOnGPU(&bwd_workspace_, bwd_workspace_size_, false);

			batch_size_ = dims[0];
			channels_in_ = dims[1];
			height_in_ = dims[2];
			width_in_ = dims[3];

			output_dims_[0] = n;
			output_dims_[1] = c;
			output_dims_[2] = h;
			output_dims_[3] = w;
		}


		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(output_dims_, ndims, &options);

		alpha = 1.0f;
		beta = 0;

		LTEN_CUDNN_CHECK_2(cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc_, input.get_data_ptr(), wtDesc_, weight_ptr_->get_data_ptr(), convDesc_, algo_, workspace_, workspace_size_, &beta, outputDesc_, result->get_data_ptr()));
		if (use_bias_)
		{
			beta = 1.0f;
			LTEN_CUDNN_CHECK_2(cudnnAddTensor(cudnnHandle, &alpha, biasDesc_, bias_ptr_->get_data_ptr(), &beta, outputDesc_, result->get_data_ptr()));
		}


		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(conv2_cudnn_backward);
		resultImpl->set_autograd(true);


		return Tensor(result);
	}


	std::vector<Tensor*> conv2d_CUDNN::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}


	void conv2d_CUDNN::clear_gradients()
	{
		if (weight_ptr_)
		{
			weight_ptr_->clear_gradients();
		}
		if (bias_ptr_)
		{
			bias_ptr_->clear_gradients();
		}
	}


	//------------------------------------------------------
	bool conv3d_CUDNN::init()
	{
		TensorOps options;
		float* raw_data_ptr;
		uint64_t numels;
		int i;
		cudnnHandle_t cudnnHandle;
		int dimsA[conv_dims + 2];
		int strideA[conv_dims + 2];
		int ouputDimsA[conv_dims + 2];
		int outputStrideA[conv_dims + 2];
		int padA[conv_dims];
		int StrideA[conv_dims];
		int dilationA[conv_dims];
		int filterDimsA[conv_dims + 2];
		int biasDimsA[conv_dims + 2];
		int biasStrideA[conv_dims + 2];


		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&biasDesc_));
		cudnnErrCheck(cudnnCreateFilterDescriptor(&wtDesc_));
		cudnnErrCheck(cudnnCreateConvolutionDescriptor(&convDesc_));


		padA[0] = pad_c_;
		padA[1] = pad_h_;
		padA[2] = pad_w_;

		StrideA[0] = stride_c_;
		StrideA[1] = stride_h_;
		StrideA[2] = stride_w_;

		dilationA[0] = 1;
		dilationA[1] = 1;
		dilationA[2] = 1;


		filterDimsA[0] = channels_out_;
		filterDimsA[1] = channels_in_;
		filterDimsA[2] = kernel_c_;
		filterDimsA[3] = kernel_h_;
		filterDimsA[4] = kernel_w_;


		dimsA[0] = batch_size_;
		dimsA[1] = channels_in_;
		dimsA[2] = depth_in_;
		dimsA[3] = height_in_;
		dimsA[4] = width_in_;

		GetStrides(dimsA, strideA, conv_dims + 2);


		biasDimsA[0] = 1;
		biasDimsA[1] = channels_out_;
		biasDimsA[2] = 1;
		biasDimsA[3] = 1;
		biasDimsA[4] = 1;

		GetStrides(biasDimsA, biasStrideA, conv_dims + 2);


		cudnnErrCheck(cudnnSetConvolutionNdDescriptor(convDesc_, 3, padA, StrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
		cudnnErrCheck(cudnnSetFilterNdDescriptor(wtDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv_dims + 2, filterDimsA));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(biasDesc_, CUDNN_DATA_FLOAT, conv_dims + 2, biasDimsA, biasStrideA));

		cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, conv_dims + 2, dimsA, strideA));
		cudnnErrCheck(cudnnGetConvolutionNdForwardOutputDim(convDesc_, inputDesc_, wtDesc_, conv_dims + 2, ouputDimsA));
		GetStrides(ouputDimsA, outputStrideA, conv_dims + 2);
		cudnnErrCheck(cudnnSetTensorNdDescriptor(outputDesc_, CUDNN_DATA_FLOAT, conv_dims + 2, ouputDimsA, outputStrideA));

		cudnnErrCheck(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
		cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, algo_, &workspace_size_));
		AllocateMemoryOnGPU(&workspace_, workspace_size_, false);


		cudnnErrCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc_, outputDesc_, convDesc_, wtDesc_, bwf_algo_, &bwf_workspace_size_));
		AllocateMemoryOnGPU(&bwf_workspace_, bwf_workspace_size_, false);


		cudnnErrCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wtDesc_, outputDesc_, convDesc_, inputDesc_, bwd_algo_, &bwd_workspace_size_));
		AllocateMemoryOnGPU(&bwd_workspace_, bwd_workspace_size_, false);


		for (i = 0; i < conv_dims + 2; i++)
		{
			output_dims_[i] = ouputDimsA[i];
		}


		options.data_type = FLOAT32;
		options.alloc_gradient_buffer = true;


		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ (uint64_t)channels_out_ , (uint64_t)channels_in_,  (uint64_t)kernel_c_,  (uint64_t)kernel_h_,  (uint64_t)kernel_w_ }, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ 1, channels_out_, 1, 1 }, &options);
			bias_ptr_->set_autograd(true);
		}

		std::random_device generator;
		//std::default_random_engine generator;
		float k = 1.0f / (channels_in_ * (kernel_h_ + kernel_w_ + kernel_c_));
		std::uniform_real_distribution<float> distribution(-sqrtf(k), sqrtf(k));

		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		numels = weight_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		if (use_bias_)
		{
			raw_data_ptr = (float*)bias_ptr_->get_data_ptr();
			numels = bias_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
			}
		}


		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(GPU, 0);
		}

		if (bias_ptr_)
		{
			*bias_ptr_ = bias_ptr_->to(GPU, 0);
		}


		return true;
	}

	conv3d_CUDNN::~conv3d_CUDNN()
	{
		cudnnErrCheck(cudnnDestroyTensorDescriptor(inputDesc_));
		cudnnErrCheck(cudnnDestroyTensorDescriptor(outputDesc_));
		cudnnErrCheck(cudnnDestroyTensorDescriptor(biasDesc_));
		cudnnErrCheck(cudnnDestroyFilterDescriptor(wtDesc_));
		cudnnErrCheck(cudnnDestroyConvolutionDescriptor(convDesc_));

		FreeMemoryOnGPU(workspace_);
		FreeMemoryOnGPU(bwf_workspace_);
		FreeMemoryOnGPU(bwd_workspace_);

		delete weight_ptr_;
		delete bias_ptr_;
	}

	Tensor conv3d_CUDNN::forward(Tensor& input)
	{
		const uint64_t* dims;
		TensorOps options;
		int ndims;
		float alpha;
		float beta;
		cudnnStatus_t status;
		cudnnHandle_t cudnnHandle;


		dims = input.get_sizes();

		ndims = input.get_ndims();

		if (ndims != 5)
		{
			LTEN_ERR("conv2d_CUDNN requires tensors with exactly 5 dimensions (NCDHW)");
		}

		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("Invalid tensor device type");
		}

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		if (channels_in_ != dims[1]) // this means reinitializing weights, a no-no
		{
			LTEN_ERR("Dimension 1 must be equal to the number of input channels");
		}

		/*
		if (batch_size_ != dims[0] || height_in_ != dims[2] || width_in_ != dims[3])
		{
			LTEN_CUDNN_CHECK_2(cudnnSetTensor4dDescriptor(inputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]), static_cast<int>(dims[3])));
			LTEN_CUDNN_CHECK_2(cudnnGetConvolution2dForwardOutputDim(convDesc_, inputDesc_, wtDesc_, &n, &c, &h, &w));
			LTEN_CUDNN_CHECK_2(cudnnSetTensor4dDescriptor(outputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, algo_, &workspace_size_));
			FreeMemoryOnGPU(workspace_);
			AllocateMemoryOnGPU(&workspace_, workspace_size_, false);


			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc_, outputDesc_, convDesc_, wtDesc_, bwf_algo_, &bwf_workspace_size_));
			FreeMemoryOnGPU(bwf_workspace_);
			AllocateMemoryOnGPU(&bwf_workspace_, bwf_workspace_size_, false);


			LTEN_CUDNN_CHECK_2(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wtDesc_, outputDesc_, convDesc_, inputDesc_, bwd_algo_, &bwd_workspace_size_));
			FreeMemoryOnGPU(bwd_workspace_);
			AllocateMemoryOnGPU(&bwd_workspace_, bwd_workspace_size_, false);

			batch_size_ = dims[0];
			channels_in_ = dims[1];
			height_in_ = dims[2];
			width_in_ = dims[3];

			output_dims_[0] = n;
			output_dims_[1] = c;
			output_dims_[2] = h;
			output_dims_[3] = w;
		}
		*/

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(output_dims_, ndims, &options);

		alpha = 1.0f;
		beta = 0;

		LTEN_CUDNN_CHECK_2(cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc_, input.get_data_ptr(), wtDesc_, weight_ptr_->get_data_ptr(), convDesc_, algo_, workspace_, workspace_size_, &beta, outputDesc_, result->get_data_ptr()));
		if (use_bias_)
		{
			beta = 1.0f;
			LTEN_CUDNN_CHECK_2(cudnnAddTensor(cudnnHandle, &alpha, biasDesc_, bias_ptr_->get_data_ptr(), &beta, outputDesc_, result->get_data_ptr()));
		}


		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(conv3_cudnn_backward);
		resultImpl->set_autograd(true);


		return Tensor(result);
	}

	std::vector<Tensor*> conv3d_CUDNN::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}


	void conv3d_CUDNN::clear_gradients()
	{
		if (weight_ptr_)
		{
			weight_ptr_->clear_gradients();
		}
		if (bias_ptr_)
		{
			bias_ptr_->clear_gradients();
		}
	}




	bool conv_CUDNN::init()
	{
		TensorOps options;
		float* raw_data_ptr;
		uint64_t numels;
		uint32_t i;
		cudnnHandle_t cudnnHandle;
		int dimsA[MAX_DIMS + 2];
		int ouputDimsA[MAX_DIMS + 2];
		int outputStrideA[MAX_DIMS + 2];
		int strideA[MAX_DIMS + 2];
		int dilationA[MAX_DIMS];
		int filterDimsA[MAX_DIMS + 2];
		int biasDimsA[MAX_DIMS + 2];
		int biasStrideA[MAX_DIMS + 2];


		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&biasDesc_));
		cudnnErrCheck(cudnnCreateFilterDescriptor(&wtDesc_));
		cudnnErrCheck(cudnnCreateConvolutionDescriptor(&convDesc_));


		dilationA[0] = 1;
		dilationA[1] = 1;
		dilationA[2] = 1;


		filterDimsA[0] = channels_out_;
		filterDimsA[1] = channels_in_ / groupCount_;
		for (i = 0; i < ndims_; i++)
		{
			filterDimsA[i + 2] = kernel_[i];
		}

		dimsA[0] = batch_size_;
		dimsA[1] = channels_in_;
		for (i = 0; i < ndims_; i++)
		{
			dimsA[i + 2] = dims_[i];
		}
		GetStrides(dimsA, strideA, ndims_ + 2);


		biasDimsA[0] = 1;
		biasDimsA[1] = channels_out_;
		biasDimsA[2] = 1;
		biasDimsA[3] = 1;
		biasDimsA[4] = 1;

		GetStrides(biasDimsA, biasStrideA, ndims_ + 2);

		if (ndims_ == 3) // not sure why but ndims_ = 3 is very slow with *ALGO_1
		{
			bwf_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
			bwd_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		}
		else
		{
			bwf_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
			bwd_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
		}

		cudnnErrCheck(cudnnSetConvolutionNdDescriptor(convDesc_, ndims_, padding_, stride_, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
		cudnnErrCheck(cudnnSetConvolutionGroupCount(convDesc_, groupCount_));
		cudnnErrCheck(cudnnSetFilterNdDescriptor(wtDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, ndims_ + 2, filterDimsA));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(biasDesc_, CUDNN_DATA_FLOAT, ndims_ + 2, biasDimsA, biasStrideA));

		cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, ndims_ + 2, dimsA, strideA));
		cudnnErrCheck(cudnnGetConvolutionNdForwardOutputDim(convDesc_, inputDesc_, wtDesc_, ndims_ + 2, ouputDimsA));
		GetStrides(ouputDimsA, outputStrideA, ndims_ + 2);
		cudnnErrCheck(cudnnSetTensorNdDescriptor(outputDesc_, CUDNN_DATA_FLOAT, ndims_ + 2, ouputDimsA, outputStrideA));

		cudnnErrCheck(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
		cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, algo_, &workspace_size_));
		AllocateMemoryOnGPU(&workspace_, workspace_size_, false);


		cudnnErrCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc_, outputDesc_, convDesc_, wtDesc_, bwf_algo_, &bwf_workspace_size_));
		AllocateMemoryOnGPU(&bwf_workspace_, bwf_workspace_size_, false);


		cudnnErrCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wtDesc_, outputDesc_, convDesc_, inputDesc_, bwd_algo_, &bwd_workspace_size_));
		AllocateMemoryOnGPU(&bwd_workspace_, bwd_workspace_size_, false);

		for (i = 0; i < ndims_ + 2; i++)
		{
			output_dims_[i] = ouputDimsA[i];
		}


		options.data_type = FLOAT32;
		options.alloc_gradient_buffer = true;


		uint64_t filterDims[MAX_DIMS];
		for (i = 0; i < ndims_ + 2; i++)
		{
			filterDims[i] = static_cast<uint64_t>(filterDimsA[i]);
		}

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor(filterDims, ndims_ + 2, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			uint64_t  biasDims[MAX_DIMS];
			for (i = 0; i < ndims_ + 2; i++)
			{
				biasDims[i] = static_cast<uint64_t>(biasDimsA[i]);
			}

			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor(biasDims, ndims_ + 2, &options);
			bias_ptr_->set_autograd(true);
		}
		else
		{
			bias_ptr_ = nullptr;
		}


		std::random_device generator;
		//std::default_random_engine generator;
		float k = 1.0f / (channels_in_ * (weight_ptr_->get_numels() / channels_out_));
		std::uniform_real_distribution<float> distribution(-sqrtf(k), sqrtf(k));

		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		numels = weight_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}

		if (use_bias_)
		{
			raw_data_ptr = (float*)bias_ptr_->get_data_ptr();
			numels = bias_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
			}
		}

		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(GPU, 0);
		}

		if (bias_ptr_)
		{
			*bias_ptr_ = bias_ptr_->to(GPU, 0);
		}


		return true;
	}

	conv_CUDNN::~conv_CUDNN()
	{
		cudnnErrCheck(cudnnDestroyTensorDescriptor(inputDesc_));
		cudnnErrCheck(cudnnDestroyTensorDescriptor(outputDesc_));
		cudnnErrCheck(cudnnDestroyTensorDescriptor(biasDesc_));
		cudnnErrCheck(cudnnDestroyFilterDescriptor(wtDesc_));
		cudnnErrCheck(cudnnDestroyConvolutionDescriptor(convDesc_));

		FreeMemoryOnGPU(workspace_);
		FreeMemoryOnGPU(bwf_workspace_);
		FreeMemoryOnGPU(bwd_workspace_);

		delete dims_;
		delete kernel_;
		delete padding_;
		delete stride_;
		delete output_dims_;

		delete weight_ptr_;
		delete bias_ptr_;
	}


	Tensor conv_CUDNN::forward(Tensor& input)
	{
		const uint64_t* dims;
		TensorOps options;
		int ndims;
		float alpha;
		float beta;
		cudnnStatus_t status;
		cudnnHandle_t cudnnHandle;


		dims = input.get_sizes();

		ndims = input.get_ndims();

		if (ndims != ndims_ + 2)
		{
			LTEN_ERR("Incompatible number of input dimensions");
		}

		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("Invalid tensor device type");
		}

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		if (channels_in_ != dims[1]) // this means reinitializing weights, a no-no
		{
			LTEN_ERR("Dimension 1 must be equal to the number of input channels");
		}

		if (batch_size_ != dims[0] || dims_[0] != dims[2] || dims_[1] != dims[3] || dims_[2] != dims[4]) // TODO: BUGBUG: this check will only work for 3d (i.e. ndims==5), fix for other dims
		{
			int dimsA[MAX_DIMS + 2];
			int strideA[MAX_DIMS + 2];
			int ouputDimsA[MAX_DIMS + 2];
			int outputStrideA[MAX_DIMS + 2];
			int i;

			for (i = 2; i < ndims; i++)
			{
				dims_[i - 2] = static_cast<int>(dims[i]);
			}
			batch_size_ = static_cast<uint32_t>(dims[0]);

			dimsA[0] = batch_size_;
			dimsA[1] = channels_in_;
			for (i = 0; i < (int)ndims_; i++)
			{
				dimsA[i + 2] = dims_[i];
			}
			GetStrides(dimsA, strideA, ndims_ + 2);

			cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, ndims_ + 2, dimsA, strideA));
			cudnnErrCheck(cudnnGetConvolutionNdForwardOutputDim(convDesc_, inputDesc_, wtDesc_, ndims_ + 2, ouputDimsA));
			GetStrides(ouputDimsA, outputStrideA, ndims_ + 2);
			cudnnErrCheck(cudnnSetTensorNdDescriptor(outputDesc_, CUDNN_DATA_FLOAT, ndims_ + 2, ouputDimsA, outputStrideA));


			cudnnErrCheck(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
			cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc_, wtDesc_, convDesc_, outputDesc_, algo_, &workspace_size_));
			FreeMemoryOnGPU(workspace_);
			AllocateMemoryOnGPU(&workspace_, workspace_size_, false);


			cudnnErrCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc_, outputDesc_, convDesc_, wtDesc_, bwf_algo_, &bwf_workspace_size_));
			FreeMemoryOnGPU(bwf_workspace_);
			AllocateMemoryOnGPU(&bwf_workspace_, bwf_workspace_size_, false);


			cudnnErrCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wtDesc_, outputDesc_, convDesc_, inputDesc_, bwd_algo_, &bwd_workspace_size_));
			FreeMemoryOnGPU(bwd_workspace_);
			AllocateMemoryOnGPU(&bwd_workspace_, bwd_workspace_size_, false);

			for (i = 0; i < ndims; i++)
			{
				output_dims_[i] = ouputDimsA[i];
			}

		}

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(output_dims_, ndims, &options);

		alpha = 1.0f;
		beta = 0;

		LTEN_CUDNN_CHECK_2(cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc_, input.get_data_ptr(), wtDesc_, weight_ptr_->get_data_ptr(), convDesc_, algo_, workspace_, workspace_size_, &beta, outputDesc_, result->get_data_ptr()));
		if (use_bias_)
		{
			beta = 1.0f;
			LTEN_CUDNN_CHECK_2(cudnnAddTensor(cudnnHandle, &alpha, biasDesc_, bias_ptr_->get_data_ptr(), &beta, outputDesc_, result->get_data_ptr()));
		}


		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(conv_cudnn_backward);
		resultImpl->set_autograd(true);


		return Tensor(result);
	}

	std::vector<Tensor*> conv_CUDNN::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}


	void conv_CUDNN::clear_gradients()
	{
		if (weight_ptr_)
		{
			weight_ptr_->clear_gradients();
		}
		if (bias_ptr_)
		{
			bias_ptr_->clear_gradients();
		}
	}

#endif


} // namespace lten

