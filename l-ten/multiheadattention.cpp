#include <iostream>
#include "lten.h"
#include "utils.h"

namespace lten {
	bool MultiheadAttention::init()
	{
		float* raw_data_ptr;
		uint64_t numels;
		uint64_t i;
		TensorOps options;

		if (embedding_dim_ % num_heads_)
		{
			LTEN_ERR("MultiheadAttention layers require the embedding dimension to be a multiple of the number of heads");
		}

		options.alloc_gradient_buffer = true;
		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ 1, 3, embedding_dim_, embedding_dim_ }, &options);
		weight_ptr_->set_autograd(true);
		weight_ptr_->set_accumulate_gradients(true);

		if (use_bias_)
		{
			bias_ptr_ = new Tensor;
			*bias_ptr_ = AllocateTensor({ 1, 3, 1, embedding_dim_ }, &options);
			bias_ptr_->set_autograd(true);
			bias_ptr_->set_accumulate_gradients(true);

			projection_bias_ptr_ = new Tensor;
			*projection_bias_ptr_ = AllocateTensor({ 1, 1, embedding_dim_ }, &options);
			projection_bias_ptr_->set_autograd(true);
			projection_bias_ptr_->set_accumulate_gradients(true);
		
		}

		projection_ptr_ = new Tensor;
		*projection_ptr_ = AllocateTensor({ 1, embedding_dim_, embedding_dim_ }, &options);
		projection_ptr_->set_autograd(true);
		projection_ptr_->set_accumulate_gradients(true);


		std::random_device generator;
		//std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-sqrtf(1.0f / embedding_dim_), sqrtf(1.0f / embedding_dim_));


		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		numels = weight_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}


		raw_data_ptr = (float*)projection_ptr_->get_data_ptr();
		numels = projection_ptr_->get_numels();

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

			raw_data_ptr = (float*)projection_bias_ptr_->get_data_ptr();
			numels = projection_bias_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
			}
		}

		return true;
	}


	Tensor MultiheadAttention::forward(Tensor& input)
	{
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		int ndims;
		//float* result_elements;
		Tensor lin;
		Tensor q;
		Tensor k;
		Tensor v;
		Tensor x;
		Tensor attn;
		int batch_size;
		int seq_len;
		const uint64_t* sizes;
		float scale;

		ndims = input.get_ndims();
		if (ndims != 3)
		{
			LTEN_ERR("MultiheadAttention layers require tensors to have 3 dimensions");
		}

		sizes = input.get_sizes();
		if (sizes[ndims - 1] != embedding_dim_)
		{
			LTEN_ERR("MultiheadAttention layers require least significant dimension to be equal to the embedding dimension");
		}

		batch_size = (uint32_t)sizes[0];
		seq_len = (uint32_t)sizes[1];

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);


		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		memcpy(dims, input.get_sizes(), sizeof(uint64_t) * ndims);

		//resultImpl->allocate(dims, ndims, &options);
		//result_elements = static_cast<float*>(resultImpl->get_data_ptr());

		scale = 1.0f / (float)sqrt(embedding_dim_ / num_heads_);

		if (CPU == options.device_type)
		{
			input = input.unsqueeze(1);

			//------------------------------------------------
			// transposed just to match pytorch, please remove
			lin = input.matmul((*weight_ptr_).transpose(2,3));
			//------------------------------------------------
			

			if (use_bias_)
			{
				lin = lin + (*bias_ptr_);
			}
			

			lin = lin.transpose(0, 1); // traspose so that 3-way split can be done

			q = lin[0];
			k = lin[1];
			v = lin[2];
			q = q.reshape({ (uint64_t)batch_size, (uint64_t)seq_len, (uint64_t)num_heads_, (uint64_t)embedding_dim_ / num_heads_ });
			k = k.reshape({ (uint64_t)batch_size, (uint64_t)seq_len, (uint64_t)num_heads_, (uint64_t)embedding_dim_ / num_heads_ });
			v = v.reshape({ (uint64_t)batch_size, (uint64_t)seq_len, (uint64_t)num_heads_, (uint64_t)embedding_dim_ / num_heads_ });
			q = q.permute({ 0, 2, 1, 3 });
			k = k.permute({ 0, 2, 1, 3 });
			v = v.permute({ 0, 2, 1, 3 });

			attn = q.matmul(k.transpose(2, 3)) * scale;

			attn = lten::softmax(attn, -1);

			//printf("x---------------------------\n\n");
			//std::cout << q << "\n";
			//std::cout << k << "\n";
			//std::cout << v << "\n";
			//printf("y---------------------------\n\n");
			//std::cout << attn << "\n";

			x = (attn.matmul(v)).transpose(1, 2);
			x = x.reshape({ (uint64_t)batch_size, (uint64_t)seq_len, (uint64_t)embedding_dim_ });
		

			//------------------------------------------------
			// transposed just to match pytorch, please remove
			x = x.matmul((*projection_ptr_).transpose(1,2));
			if (use_bias_)
			{
				x = x + (*projection_bias_ptr_);
			}
			return x;
			//------------------------------------------------
			//std::cout << x << "\n";

			//assert(result->get_numels() == x.get_numels());
			//memcpy(result->get_data_ptr(), x.get_data_ptr(), sizeof(float) * x.get_numels());

		}
		else
		{
			if (GPU == options.device_type)
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

		//resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		//resultImpl->set_grad_fn(dropout_backward);
		//resultImpl->set_autograd(true);

		return Tensor(result);
	}

	void MultiheadAttention::clear_gradients()
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


	std::vector<Tensor*> MultiheadAttention::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);
		weights.push_back(projection_ptr_);

		if (bias_ptr_)
		{
			weights.push_back(bias_ptr_);
			weights.push_back(projection_bias_ptr_);		
		}

		
		return weights;
	}


	void MultiheadAttention::to(device target_device, int target_device_index)
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
