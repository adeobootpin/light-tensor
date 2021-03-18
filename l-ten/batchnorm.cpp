#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "utils.h"

namespace lten {
	bool BatchNorm::init()
	{
		assert(0); // implementation not completed
		float* raw_data_ptr;
		uint64_t i;
		TensorOps options;

		options.alloc_gradient_buffer = true;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ num_features_ }, &options);
		weight_ptr_->set_autograd(true);

		bias_ptr_ = new Tensor;
		*bias_ptr_ = AllocateTensor({ num_features_ }, &options);
		bias_ptr_->set_autograd(true);


		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		for (i = 0; i < num_features_; i++)
		{
			raw_data_ptr[i] = 1.0f;  // same as pytorch
		}

		raw_data_ptr = (float*)bias_ptr_->get_data_ptr();
		for (i = 0; i < num_features_; i++)
		{
			raw_data_ptr[i] = 0; // same as pytorch
		}


		mu_ = new Tensor;
		*mu_ = AllocateTensor({ num_features_ }, &options);

		sigma_ = new Tensor;
		*sigma_ = AllocateTensor({ num_features_ }, &options);


		raw_data_ptr = (float*)mu_->get_data_ptr();
		for (i = 0; i < num_features_; i++)
		{
			raw_data_ptr[i] = 0.0f;  // same as pytorch
		}

		raw_data_ptr = (float*)sigma_->get_data_ptr();
		for (i = 0; i < num_features_; i++)
		{
			raw_data_ptr[i] = 1.0f; // same as pytorch
		}


		max_ones_vector_size_ = 100;
		ones_vector_ = new Tensor;
		*ones_vector_ = AllocateTensor({ max_ones_vector_size_ }, &options);
		FillBuffer(static_cast<float*>(ones_vector_->get_data_ptr()), max_ones_vector_size_, 1.0f);


		return true;
	}


	Tensor BatchNorm::forward(Tensor& input)
	{
		assert(0); // implementation not completed
		TensorOps options;
		const uint64_t* dims;
		uint64_t M;
		//float alpha;
		//float beta;
		//int lda;
		//int ldb;
		//int ldc;
		int ndims;
		int i;

		ndims = input.get_ndims();
		if (ndims < 3)
		{
			LTEN_ERR("BatchNorm requires tensors with at least 3 dimensions");
		}

		if (input.get_device() != weight_ptr_->get_device() ||
			input.get_device_index() != weight_ptr_->get_device_index() ||
			input.get_data_type() != weight_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}

		dims = input.get_sizes();

		if (dims[1] != num_features_)
		{
			LTEN_ERR("Dimension 1 must be equal to the number of features");
		}


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


		if (options.device_type == CPU)
		{
			//Dtype* inputs = input.get_mdarray<Dtype>()->GetDataPtr();


			//cpu_gemm(false, true, M, output_features_, input_features_, static_cast<Dtype>(1), inputs, weights, static_cast<Dtype>(0), results);
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		/*
		if (is_training_)
		{
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<Dtype>*>(input.get_impl().get())));
			resultImpl->add_child(*(static_cast<TensorImpl<Dtype>*>(weight_ptr_->get_impl().get())));
			resultImpl->set_grad_fn(fc_backward);
			resultImpl->set_autograd(true);
		}
		*/

		return Tensor(result);
	}

	void BatchNorm::clear_gradients()
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


	std::vector<Tensor*> BatchNorm::get_all_weights()
	{
		assert(0); // implementation not completed
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		weights.push_back(bias_ptr_);

		return weights;
	}


	void BatchNorm::to(device target_device, int target_device_index)
	{
		assert(0); // implementation not completed
		TensorOps options;

		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(target_device, target_device_index);
		}

		if (bias_ptr_)
		{
			*bias_ptr_ = bias_ptr_->to(target_device, target_device_index);
		}
	}

#ifdef USE_CUDA
	bool BatchNorm_CUDNN::init()
	{
		TensorOps options;

		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&scale_bias_mean_var_Desc_));
		
		options.device_type = GPU;
		options.alloc_gradient_buffer = true;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ num_features_ }, &options);
		weight_ptr_->set_autograd(true);

		bias_ptr_ = new Tensor;
		*bias_ptr_ = AllocateTensor({ num_features_ }, &options);
		bias_ptr_->set_autograd(true);

		gpu_fill((float*)weight_ptr_->get_data_ptr(), num_features_, 1.0f); // same as pytorch
		gpu_fill((float*)bias_ptr_->get_data_ptr(), num_features_, 0.0f); // same as pytorch

		mu_ = new Tensor;
		*mu_ = AllocateTensor({ num_features_ }, &options);

		sigma_ = new Tensor;
		*sigma_ = AllocateTensor({ num_features_ }, &options);

		gpu_fill((float*)mu_->get_data_ptr(), num_features_, 0.0f); // same as pytorch
		gpu_fill((float*)sigma_->get_data_ptr(), num_features_, 1.0f); // same as pytorch

		mode_ = CUDNN_BATCHNORM_SPATIAL;
		mo_ = 0.1f;
		epsilon_ = 1e-5;

		return true;
	}


	Tensor BatchNorm_CUDNN::forward(Tensor& input)
	{
		TensorOps options;
		const uint64_t* dims;
		const uint64_t* strides;
		int ndims;
		int i;
		float alpha;
		float beta;
		cudnnHandle_t cudnnHandle;


		ndims = input.get_ndims();
		if (ndims < 4)
		{
			LTEN_ERR("BatchNorm_CUDNN requires tensors with at least 4 dimensions");
		}

		if (input.get_device() != weight_ptr_->get_device() ||
			input.get_device_index() != weight_ptr_->get_device_index() ||
			input.get_data_type() != weight_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}

		dims = input.get_sizes();

		if (dims[1] != num_features_)
		{
			LTEN_ERR("Dimension 1 must be equal to the number of features");
		}

		int dimA[MAX_DIMS];
		int strideA[MAX_DIMS];
		
		strides = input.get_mdarray<float>()->GetStrides();

		for (i = 0; i < ndims; i++)
		{
			dimA[i] = static_cast<int>(dims[i]);
			strideA[i] = static_cast<int>(strides[i]);
		}

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(dims, ndims, &options);


		cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, ndims, dimA, strideA));
		cudnnErrCheck(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_Desc_, inputDesc_, mode_));

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		alpha = 1.0f;
		beta = 0.0f;

		if (is_training_)
		{
			cudnnErrCheck(cudnnBatchNormalizationForwardTraining(cudnnHandle, mode_, &alpha, &beta, inputDesc_, input.get_data_ptr(), inputDesc_, resultImpl->get_data_ptr(), scale_bias_mean_var_Desc_,
				weight_ptr_->get_data_ptr(), bias_ptr_->get_data_ptr(), mo_, mu_->get_data_ptr(), sigma_->get_data_ptr(), epsilon_, nullptr, nullptr));
		}
		else
		{
			cudnnErrCheck(cudnnBatchNormalizationForwardInference(cudnnHandle, mode_, &alpha, &beta, inputDesc_, input.get_data_ptr(), inputDesc_, resultImpl->get_data_ptr(), scale_bias_mean_var_Desc_,
				weight_ptr_->get_data_ptr(), bias_ptr_->get_data_ptr(), mu_->get_data_ptr(), sigma_->get_data_ptr(), epsilon_));
		}

		
		if (is_training_)
		{
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(bn_cudnn_backward);
			resultImpl->set_autograd(true);
		}
		
		return Tensor(result);
	}

	void BatchNorm_CUDNN::clear_gradients()
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


	std::vector<Tensor*> BatchNorm_CUDNN::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		weights.push_back(bias_ptr_);

		return weights;
	}
#endif

}