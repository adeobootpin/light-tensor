#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "utils.h"

//https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

namespace lten {
	bool LayerNorm::init()
	{
		float* raw_data_ptr;
		uint64_t i;
		TensorOps options;

		if (affine_)
		{
			options.alloc_gradient_buffer = true;

			weight_ptr_ = new Tensor;
			*weight_ptr_ = AllocateTensor({ 1, 1, 1, num_features_ }, &options);
			weight_ptr_->set_autograd(true);

			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ 1, 1, 1, num_features_ }, &options);
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
		}

		return true;
	}


	Tensor LayerNorm::forward(Tensor& input)
	{
		int i;
		TensorOps options;
		const uint64_t* dims_src;
		uint64_t dims_dst[MAX_DIMS];
		uint64_t dims_tmp[MAX_DIMS];
		int ndims_src;
		int ndims_dst;

		//----------------------------------------
		// only last dim supported as axis for now
		naxes_ = 1;
		axes_[0] = input.get_ndims() - 1;
		//----------------------------------------

		ndims_src = input.get_ndims();
		if (ndims_src < 2)
		{
			LTEN_ERR("LayerNorm requires tensors with at least 2 dimensions");
		}

		if (affine_)
		{
			if (!weight_ptr_ || 
				input.get_device() != weight_ptr_->get_device() ||
				input.get_device_index() != weight_ptr_->get_device_index() ||
				input.get_data_type() != weight_ptr_->get_data_type())
			{
				LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
			}
		}


		for (i = 0; i < naxes_; i++)
		{
			if (axes_[i] >= ndims_src)
			{
				LTEN_ERR("Invalid index");
			}
		}

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_type = input.get_device();
		options.device_index = input.get_device_index();


		dims_src = input.get_sizes();

		memcpy(dims_tmp, dims_src, sizeof(uint64_t) * ndims_src);
		for (i = 0; i < naxes_; i++)
		{
			dims_tmp[axes_[i]] = 0; // set to an invalid value
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


		sd_ = AllocateTensor(dims_src, ndims_src - 1, &options);
		ln_ = AllocateTensor(dims_src, ndims_src, &options);
		LTEN_ERR_CHECK(resultImpl->allocate(dims_src, ndims_src, &options));


		if (CPU == options.device_type)
		{
			assert(0); // TODO implement cpu version (previous version below)
			LTEN_ERR("LayerNorm: Not yet implemented: mean for cpu");
			/*
			cpu_mean((float*)input.get_data_ptr(), (float*)mu_.get_data_ptr(), mu_.get_numels(), ratio, dim_size, src_strides[axis]);
			cpu_std((float*)input.get_data_ptr(), (float*)sd_.get_data_ptr(), sd_.get_numels(), ratio, dim_size, src_strides[axis], false);

			mu_ = mu_.unsqueeze(mu_.get_ndims());

			sd_ = sd_.unsqueeze(sd_.get_ndims());

			temp1_ = input - mu_;
			ln_ = temp1_.div(sd_);

			if (affine_)
			{
				temp2_ = ln_ * (*weight_ptr_);
				temp3_ = temp2_ + (*bias_ptr_);
				temp3_.get_mdarray<float>()->SetMemoryOwnership(false);
				resultImpl->allocate_from_buffer(dims, ndims, temp3_.get_data_ptr(), true, &options);
			}
			else
			{
				ln_.get_mdarray<float>()->SetMemoryOwnership(false);
				resultImpl->allocate_from_buffer(dims, ndims, ln_.get_data_ptr(), true, &options);
			}
			*/
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				if (affine_)
				{
					if (is_training_)
					{
						gpu_layer_norm((float*)resultImpl->get_data_ptr(), (float*)input.get_data_ptr(), input.get_numels(), resultImpl->get_strides(), input.get_strides(), ndims_dst, ndims_src, input.get_sizes(), axes_, (float*)weight_ptr_->get_data_ptr(), (float*)bias_ptr_->get_data_ptr(), (float*)ln_.get_data_ptr(), (float*)sd_.get_data_ptr());
					}
					else
					{
						gpu_layer_norm((float*)resultImpl->get_data_ptr(), (float*)input.get_data_ptr(), input.get_numels(), resultImpl->get_strides(), input.get_strides(), ndims_dst, ndims_src, input.get_sizes(), axes_, (float*)weight_ptr_->get_data_ptr(), (float*)bias_ptr_->get_data_ptr(), (float*)nullptr, (float*)nullptr);
					}
				}
				else
				{
					gpu_layer_norm((float*)resultImpl->get_data_ptr(), (float*)input.get_data_ptr(), input.get_numels(), resultImpl->get_strides(), input.get_strides(), ndims_dst, ndims_src, input.get_sizes(), axes_, (float*)nullptr, (float*)nullptr, (float*)nullptr, (float*)nullptr);
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

		if (is_training_)
		{
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(layernorm_backward);
			resultImpl->set_autograd(true);
		}


		return Tensor(result);
			}

	void LayerNorm::clear_gradients()
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


	std::vector<Tensor*> LayerNorm::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		weights.push_back(bias_ptr_);

		return weights;
	}


	void LayerNorm::to(device target_device, int target_device_index)
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

}
