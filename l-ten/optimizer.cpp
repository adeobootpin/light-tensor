#include <cmath>
#include <string.h>
#include "tensor.h"
#include "layers.h"
#include "net.h"
#include "optimizer.h"
#include "error.h"

namespace lten {
	void SGDOptimizer::setup_optimizer()
	{
		int i;
		uint64_t numels;

		for (i = 0; i < num_params_; i++)
		{
			numels = network_params_ptr_[i].param_->get_numels();
			network_params_ptr_[i].param_data_ = new Tensor;
			*network_params_ptr_[i].param_data_ = AllocateTensor(network_params_ptr_[i].param_->get_sizes(), network_params_ptr_[i].param_->get_ndims(),0); // allocate buffer for momentum/velocity
			memset(network_params_ptr_[i].param_data_->get_data_ptr(), 0, sizeof(float) * numels);
		}

	}


	void SGDOptimizer::step()
	{
		float* weight_ptr;
		float* weight_grad_ptr;
		float* velocity_ptr;
		int i;
		uint64_t j;
		uint64_t numels;
		device device_type;

		if (!num_params_)
		{
			LTEN_ERR("No parameters have been added to the optimizer");
		}

		device_type = network_params_ptr_[0].param_->get_device(); // key off first param

		if (device_type == CPU)
		{
			for (i = 0; i < num_params_; i++)
			{
				weight_ptr = (float*)network_params_ptr_[i].param_->get_data_ptr();
				weight_grad_ptr = (float*)network_params_ptr_[i].param_->get_grad_ptr();
				numels = network_params_ptr_[i].param_->get_numels();
				velocity_ptr = (float*)network_params_ptr_[i].param_data_->get_data_ptr();


				for (j = 0; j < numels; j++)
				{
					weight_grad_ptr[j] = wd_ * weight_ptr[j] + weight_grad_ptr[j];
					velocity_ptr[j] = velocity_ptr[j] * mo_ + (1.0f - mo_) * weight_grad_ptr[j];
					weight_ptr[j] = weight_ptr[j] - (velocity_ptr[j] * lr_);
				}
			}
		}
		else
		{
			if (device_type == GPU)
			{
#ifdef USE_CUDA
				for (i = 0; i < num_params_; i++)
				{
					weight_ptr = (float*)network_params_ptr_[i].param_->get_data_ptr();
					weight_grad_ptr = (float*)network_params_ptr_[i].param_->get_grad_ptr();
					numels = network_params_ptr_[i].param_->get_numels();
					velocity_ptr = (float*)network_params_ptr_[i].param_data_->get_data_ptr();

					gpu_sgd_step(weight_ptr, weight_grad_ptr, velocity_ptr, numels, mo_, wd_, lr_);
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



	void AdamOptimizer::setup_optimizer()
	{
		int i;
		uint64_t dims[MAX_DIMS];
		uint64_t numels;

		for (i = 0; i < num_params_; i++)
		{
			numels = network_params_ptr_[i].param_->get_numels();
			network_params_ptr_[i].param_data_ = new Tensor;
			memcpy(dims, network_params_ptr_[i].param_->get_sizes(), sizeof(uint64_t) * network_params_ptr_[i].param_->get_ndims());
			dims[0] *= 3; // make room for momentum, rmsprop histories and scratch space
			*network_params_ptr_[i].param_data_ = AllocateTensor(dims, network_params_ptr_[i].param_->get_ndims(), 0); // allocate history buffer
			memset(network_params_ptr_[i].param_data_->get_data_ptr(), 0, sizeof(float) * numels * 3);
		}

	}


	void AdamOptimizer::step()
	{
		float* weight_ptr;
		float* weight_grad_ptr;
		float* v_dw;
		float* s_dw;
		float* scratch;
		float epsilon;
		uint64_t numels;
		int i;
		device device_type;
		float bias_correction1;
		float bias_correction2;


		if (!num_params_)
		{
			LTEN_ERR("No parameters have been added to the optimizer");
		}

		device_type = network_params_ptr_[0].param_->get_device(); // key off first param

		iteration_++;

		epsilon = 1.0e-8f;

		for (i = 0; i < num_params_; i++)
		{
			weight_grad_ptr = static_cast<float*>(network_params_ptr_[i].param_->get_grad_ptr());
			if (!weight_grad_ptr)
			{
				continue;
			}
			weight_ptr = static_cast<float*>(network_params_ptr_[i].param_->get_data_ptr());
			numels = network_params_ptr_[i].param_->get_numels();
			v_dw = static_cast<float*>(network_params_ptr_[i].param_data_->get_data_ptr());
			s_dw = v_dw + (numels);
			scratch = v_dw + (2 * numels);

			bias_correction1 = 1.0f - powf(beta1_, static_cast<float>(iteration_));
			bias_correction2 = 1.0f - powf(beta2_, static_cast<float>(iteration_));

			if (device_type == CPU)
			{
				cpu_axpby(numels, 1.0f - beta1_, weight_grad_ptr, beta1_, v_dw, v_dw);
				cpu_mul(numels, weight_grad_ptr, weight_grad_ptr, scratch);
				cpu_axpby(numels, 1.0f - beta2_, scratch, beta2_, s_dw, s_dw);
				cpu_mul(numels, 1.0f / bias_correction2, s_dw, scratch);
				cpu_powx(numels, scratch, 0.5f, scratch);
				cpu_add(numels, epsilon, scratch, scratch);
				cpu_div(numels, v_dw, scratch, scratch);
				cpu_axpy(numels, -(1.0f / bias_correction1) * lr_, scratch, weight_ptr, weight_ptr);
			}
			else
			{
				if (device_type == GPU)
				{
#ifdef USE_CUDA
					gpu_axpby(numels, 1.0f - beta1_, weight_grad_ptr, beta1_, v_dw, v_dw);
					gpu_mul(numels, weight_grad_ptr, weight_grad_ptr, scratch);
					gpu_axpby(numels, 1.0f - beta2_, scratch, beta2_, s_dw, s_dw);
					gpu_mul(numels, 1.0f / bias_correction2, s_dw, scratch);
					gpu_powx(numels, scratch, 0.5f, scratch);
					gpu_add(numels, epsilon, scratch, scratch);
					gpu_div(numels, v_dw, scratch, scratch);
					gpu_axpy(numels, -(1.0f / bias_correction1) * lr_, scratch, weight_ptr, weight_ptr);
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

} // namespace btpn
