#include <random>
#include <iostream>
#include "lten.h"
#include "utils.h"


namespace lten {
	bool FullyConnected::init()
	{
		float* raw_data_ptr;
		uint64_t numels;
		uint64_t i;
		TensorOps options;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ 1, 1, output_features_, input_features_ }, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			options.alloc_gradient_buffer = true;
			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ output_features_ }, &options);
			bias_ptr_->set_autograd(true);

			bias_multiplier_ = new Tensor;
		}

		std::random_device generator;
		//std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-sqrtf(1.0f / input_features_), sqrtf(1.0f / input_features_));


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


	Tensor FullyConnected::forward(Tensor& input)
	{
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		uint64_t M;
		int ndims;
		int i;

		ndims = input.get_ndims();
		if (ndims < 3)
		{
			LTEN_ERR("FullyConnected requires tensors with at least 3 dimensions");
		}

		if (input.get_device() != weight_ptr_->get_device() ||
			input.get_device_index() != weight_ptr_->get_device_index() ||
			input.get_data_type() != weight_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}

		memcpy(dims, input.get_sizes(), sizeof(uint64_t) * ndims);

		if (dims[ndims - 1] != input_features_)
		{
			LTEN_ERR("Last dimension must be equal to the number of input features");
		}

		dims[ndims - 1] = output_features_;


		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		
		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(dims, ndims, &options);

		M = 1;
		for (i = 0; i < ndims - 1; i++) // fold in all batches so that only one gemm call is needed
		{
			M *= dims[i];
		}

		float* weights = static_cast<float*>(weight_ptr_->get_data_ptr());

		if (options.device_type == CPU)
		{
			float* inputs = input.get_mdarray<float>()->GetDataPtr();
			float* results = resultImpl->get_mdarray()->GetDataPtr();

			cpu_gemm(false, true, M, output_features_, input_features_, 1.0f, inputs, weights, 0.0f, results);

			if (use_bias_)
			{
				if (M > max_bias_multiplier_size_)
				{
					*bias_multiplier_ = AllocateTensor({ M }, &options);
					FillBuffer<float>(static_cast<float*>(bias_multiplier_->get_data_ptr()), M, 1.0f);
					max_bias_multiplier_size_ = M;
				}
				cpu_gemm(false, false, M, output_features_, 1, 1.0f, static_cast<float*>(bias_multiplier_->get_data_ptr()), static_cast<float*>(bias_ptr_->get_data_ptr()), 1.0f, results);
			}
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				cublasHandle_t hCuBlas;

				float alpha;
				float beta;
				int lda;
				int ldb;
				int ldc;

				hCuBlas = CUDA_globlas::singleton()->get_cublas_handle(options.device_index);

				alpha = 1.0f;
				beta = 0.0f;
				lda = static_cast<int>(input_features_);
				ldb = static_cast<int>(input_features_);
				ldc = static_cast<int>(output_features_);

				float* inputs = input.get_mdarray<float>()->GetDataPtr();
				float* results = (float*)resultImpl->get_mdarray()->GetDataPtr();

				LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(output_features_), static_cast<int>(M), static_cast<int>(input_features_), &alpha, (float*)weights, lda, inputs, ldb, &beta, results, ldc));

				if (use_bias_)
				{
					if (M > max_bias_multiplier_size_)
					{
						*bias_multiplier_ = AllocateTensor({ M }, &options);
						gpu_fill(static_cast<float*>(bias_multiplier_->get_data_ptr()), M, 1.0f);
						max_bias_multiplier_size_ = M;
					}

					beta = 1.0f;
					lda = static_cast<int>(output_features_);
					ldb = 1;
					ldc = static_cast<int>(output_features_);
					LTEN_CUBLAS_CHECK(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(output_features_), static_cast<int>(M), 1, &alpha, static_cast<float*>(bias_ptr_->get_data_ptr()), lda, static_cast<float*>(bias_multiplier_->get_data_ptr()), ldb, &beta, results, ldc));
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

		
		if (is_training_ || input.autograd_on())
		{
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(weight_ptr_->get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(fc_backward);
			resultImpl->set_autograd(true);
		}
		
		return Tensor(result);	
	}


	void FullyConnected::clear_gradients()
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


	std::vector<Tensor*> FullyConnected::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}

	
	void FullyConnected::to(device target_device, int target_device_index)
	{
		TensorOps options;

		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(target_device, target_device_index);
		}

		if (bias_ptr_)
		{
			*bias_ptr_ = bias_ptr_->to(target_device, target_device_index);
		}

		if (bias_multiplier_)
		{
			assert(use_bias_);

			options.data_type = bias_ptr_->get_data_type();
			options.device_index = target_device_index;
			options.device_type = target_device;

			delete bias_multiplier_;
			bias_multiplier_ = new Tensor;

			if (CPU == target_device)
			{
				*bias_multiplier_ = AllocateTensor({ max_bias_multiplier_size_ }, &options);
				switch (options.data_type)
				{
				case UINT8:
					FillBuffer<uint8_t>(static_cast<uint8_t*>(bias_multiplier_->get_data_ptr()), max_bias_multiplier_size_, static_cast<uint8_t>(1));
					break;
				case INT32:
					FillBuffer<int>(static_cast<int*>(bias_multiplier_->get_data_ptr()), max_bias_multiplier_size_, static_cast<int>(1));
					break;
				case FLOAT32:
					FillBuffer<float>(static_cast<float*>(bias_multiplier_->get_data_ptr()), max_bias_multiplier_size_, static_cast<float>(1));
					break;
				}
			}
			else
			{
				if (GPU == target_device)
				{
#ifdef USE_CUDA
					*bias_multiplier_ = AllocateTensor({ max_bias_multiplier_size_ }, &options);
					switch (options.data_type)
					{
					case UINT8:
						gpu_fill(static_cast<uint8_t*>(bias_multiplier_->get_data_ptr()), max_bias_multiplier_size_, static_cast<uint8_t>(1));
						break;
					case INT32:
						gpu_fill(static_cast<int*>(bias_multiplier_->get_data_ptr()), max_bias_multiplier_size_, static_cast<int>(1));
						break;
					case FLOAT32:
						gpu_fill(static_cast<float*>(bias_multiplier_->get_data_ptr()), max_bias_multiplier_size_, static_cast<float>(1));
						break;
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



	//------------------------------------------------------------------------------------------ quantized fc----------------------------------------

	bool FullyConnected_q::init()
	{
		TensorOps options;

		workspace_ = new Tensor;

		options.data_type = UINT8;
		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ 1, 1, output_features_, input_features_ }, &options);
		weight_ptr_->set_autograd(true);

		if (use_bias_)
		{
			bias_ptr_ = new Tensor;
			options.data_type = INT32; // biases are 32-bit
			*bias_ptr_ = AllocateTensor({ output_features_ }, &options);
			bias_ptr_->set_autograd(true);
		}
		

		//
		// no weight initialization here (assume weights will be initialized externally by quantization setup)
		//

		return true;
	}

	Tensor FullyConnected_q::forward(Tensor& input)
	{
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		uint64_t M;
		int ndims;
		QuantizationParams qparams[3];

		ndims = input.get_ndims();
		if (ndims < 3)
		{
			LTEN_ERR("FullyConnected_q requires tensors with at least 3 dimensions");
		}

		if (input.get_device() != weight_ptr_->get_device() ||
			input.get_device_index() != weight_ptr_->get_device_index() ||
			input.get_data_type() != weight_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}

		memcpy(dims, input.get_sizes(), sizeof(uint64_t) * ndims);

		if (dims[ndims - 1] != input_features_)
		{
			LTEN_ERR("Last dimension must be equal to the number of input features");
		}

		dims[ndims - 1] = output_features_;


		TensorImpl<uint8_t>* resultImpl;
		resultImpl = new TensorImpl<uint8_t>;
		intrusive_ptr<TensorImplBase> result(resultImpl);


		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(dims, ndims, &options);

		M = dims[ndims - 2];

		if (M > max_batch_size_)
		{
			options.data_type = INT32;
			*workspace_ = AllocateTensor({ M + output_features_}, &options);
			max_batch_size_ = M;
		}

		md_array_dim_iterator it(dims, ndims - 2);

		uint8_t* weights = static_cast<uint8_t*>(weight_ptr_->get_data_ptr());

		qparams[0] = qparams_in_;
		qparams[1] = qparams_wt_;
		qparams[2] = qparams_out_;

		if (options.device_type == CPU)
		{
			for (auto higher_indices : it)
			{
				uint8_t* inputs = input.get_mdarray<uint8_t>()->GetDataPtr(higher_indices, ndims - 2);
				uint8_t* results = resultImpl->get_mdarray()->GetDataPtr(higher_indices, ndims - 2);
				if (bias_ptr_)
				{
					quantized_matmul(false, true, M, output_features_, input_features_, 1, inputs, weights, 0, results, qparams, static_cast<int*>(bias_ptr_->get_data_ptr()), static_cast<int*>(workspace_->get_data_ptr()));
				}
				else
				{
					quantized_matmul(false, true, M, output_features_, input_features_, 1, inputs, weights, 0, results, qparams, nullptr, static_cast<int*>(workspace_->get_data_ptr()));
				}
			}
		}
		else
		{
			LTEN_ERR("FullyConnected_q only supports the CPU device type");  // only CPU for now
		}

		if (is_training_)
		{
			resultImpl->add_child(*(static_cast<TensorImpl<uint8_t>*>(input.get_smart_ptr().get_real_object())));
			resultImpl->add_child(*(static_cast<TensorImpl<uint8_t>*>(weight_ptr_->get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(fc_backward);
			resultImpl->set_autograd(true);
		}

		return Tensor(result);
	}


	void FullyConnected_q::clear_gradients()
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

	std::vector<Tensor*> FullyConnected_q::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		if (use_bias_)
		{
			weights.push_back(bias_ptr_);
		}

		return weights;
	}

	void FullyConnected_q::set_qparams_params(QuantizationParams* qparams, int count)
	{
		if (qparams && count)
		{
			qparams_wt_ = qparams[0];
		}
	}

} // namespace lten