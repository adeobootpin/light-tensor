#include <random>
#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "utils.h"


namespace lten {
	bool Embedding::init()
	{
		float* raw_data_ptr;
		uint64_t numels;
		uint64_t i;
		TensorOps options;

		weight_ptr_ = new Tensor;
		*weight_ptr_ = AllocateTensor({ num_embeddings_, embedding_dim_ }, &options);
		weight_ptr_->set_autograd(true);


		std::random_device generator;
		std::normal_distribution<float> distribution(0, 1.0f);


		raw_data_ptr = (float*)weight_ptr_->get_data_ptr();
		numels = weight_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
		}


		return true;
	}


	Tensor Embedding::forward(Tensor& input)
	{
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		int ndims;
		dtype data_type;
		uint64_t i;
		uint64_t j;
		uint64_t k;
		uint64_t index;

		data_type = input.get_smart_ptr()->get_data_type();
		if (data_type != INT32)
		{
			LTEN_ERR("Embedding layer supports only INT32 tensors");
		}

		ndims = input.get_ndims();
		if (input.get_ndims() != 2)
		{
			LTEN_ERR("Embedding layer requires tensors with exactly 2 dimensions");
		}

		options.data_type = lten::INT32;
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		//input_indices_ = AllocateTensor(input.get_sizes(), ndims, &options);
		input_indices_ = input;


		dims[0] = input.get_sizes()[0];
		dims[1] = input.get_sizes()[1];
		dims[2] = embedding_dim_;


		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);


		options.data_type = lten::FLOAT32;
		resultImpl->allocate(dims, 3, &options);

		/*
		uint64_t coordinates[2];
		MultiDimArray<float>* dst;
		MultiDimArray<float>* src;
		MultiDimArray<int>* inp;

		dst = resultImpl->get_mdarray();
		src = weight_ptr_->get_mdarray<float>();
		inp = input.get_mdarray<int>();

		if (options.device_type == CPU)
		{
			for (i = 0; i < dims[0]; i++)
			{
				coordinates[0] = i;
				for (j = 0; j < dims[1]; j++)
				{
					coordinates[1] = j;
					index = (*inp)(coordinates, 2);
					memcpy(dst->GetDataPtr(coordinates, 2), src->GetDataPtr(&index, 1), sizeof(float) * embedding_dim_);
				}
			}
		}
		*/

		float* dst = (float*)resultImpl->get_data_ptr();
		float* src = (float*)weight_ptr_->get_data_ptr();
		int* inp = (int*)input.get_data_ptr();
		uint64_t numels = resultImpl->get_numels();
		uint64_t indices_per_batch = dims[1];

		if (options.device_type == CPU)
		{
			uint64_t rem;
			uint64_t offs;
			for (i = 0; i < numels; i++)
			{
				j = i / (indices_per_batch * embedding_dim_);
				rem = i % (indices_per_batch * embedding_dim_);
				k = rem / embedding_dim_;
				offs = rem % embedding_dim_;

				index = inp[j * indices_per_batch + k];

				dst[j * (indices_per_batch * embedding_dim_) + k * embedding_dim_ + offs] = src[index * embedding_dim_ + offs];
			}
			//memcpy(input_indices_.get_data_ptr(), input.get_data_ptr(), sizeof(int) * input_indices_.get_numels());
		}
		else
		{
			if (options.device_type == GPU)
			{
#ifdef USE_CUDA
				gpu_embedding(dst, src, inp, numels, indices_per_batch, embedding_dim_);
				//GPUToGPUCopy(input_indices_.get_data_ptr(), input.get_data_ptr(), sizeof(int) * input_indices_.get_numels());
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
			resultImpl->misc1_ = embedding_dim_;
			resultImpl->misc_ptr1_ = &input_indices_;
			resultImpl->own_misc_ptr1_ = false;
			//resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(weight_ptr_->get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(embedding_backward);
			resultImpl->set_autograd(true);
		}

		return Tensor(result);
	}


	void Embedding::clear_gradients()
	{
		if (weight_ptr_)
		{
			weight_ptr_->clear_gradients();
		}
	}


	std::vector<Tensor*> Embedding::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weight_ptr_);

		return weights;
	}


	void Embedding::to(device target_device, int target_device_index)
	{
		TensorOps options;

		if (weight_ptr_)
		{
			*weight_ptr_ = weight_ptr_->to(target_device, target_device_index);
		}
	}
} // namespace lten
