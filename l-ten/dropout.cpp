#include <random>
#include <iostream>
#include <climits>
#include "lten.h"
#include "utils.h"


namespace lten {
	bool Dropout::init()
	{
		threshold_ = static_cast<unsigned int>(UINT_MAX * probability_);

		scale_ = 1.0f / (1.0f - probability_);

		mask_ = new MultiDimArray<unsigned int>;

#ifdef USE_CUDA
		curandCreateGenerator(&cuda_generator_, CURAND_RNG_PSEUDO_DEFAULT);
#endif

		return true;
	}

	Tensor Dropout::forward(Tensor& input)
	{
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		uint64_t len;
		uint64_t i;
		int ndims;
		unsigned int* mask_elements;
		float* result_elements;


		ndims = input.get_ndims();

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);


		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		memcpy(dims, input.get_sizes(), sizeof(uint64_t) * ndims);

		mask_->Allocate(dims, ndims);

		mask_elements = mask_->GetDataPtr();
		len = mask_->GetNumels();

		resultImpl->allocate(dims, ndims, &options);

		result_elements = static_cast<float*>(resultImpl->get_data_ptr());

		if (CPU == options.device_type)
		{
			if (is_training_)
			{
				for (i = 0; i < len; i++)
				{
					mask_elements[i] = (*distribution_)(generator_);
				}

				cpu_dropout((float*)resultImpl->get_data_ptr(), (float*)input.get_data_ptr(), mask_elements, threshold_, scale_, len);
			}
			else
			{
				memcpy(resultImpl->get_data_ptr(), input.get_data_ptr(), sizeof(float) * input.get_numels());
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				if (is_training_)
				{
					curandGenerate(cuda_generator_, mask_elements, len);
					gpu_dropout((float*)resultImpl->get_data_ptr(), (float*)input.get_data_ptr(), mask_elements, threshold_, scale_, len);
				}
				else
				{
					GPUToGPUCopy(resultImpl->get_data_ptr(), input.get_data_ptr(), sizeof(float) * input.get_numels());
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
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(dropout_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);

	}



	void Dropout::to(device target_device, int target_device_index)
	{
		if (mask_)
		{
			delete mask_;
		}

		if (CPU == target_device)
		{
			mask_ = new MultiDimArray<unsigned int>;
		}
		else
		{
			if (GPU == target_device)
			{
#ifdef USE_CUDA
				mask_ = new CUDA_MultiDimArray<unsigned int>;
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
} // namespace lten