#ifndef NET_H
#define NET_H

#include "utils.h"


namespace lten {
struct NetworkParms
{
	char* name_ptr;
	Tensor* param_; // trainable params
	Tensor* param_data_; // extra stuff related to params (e.g. velocity history etc.)
};

struct NetworkModules
{
	char* name_ptr;
	Module* module_;
};

class NeuralNetwork
{
public:
	NeuralNetwork() : num_params_(0), network_param_array_(nullptr), num_modules_(0), network_module_array_(nullptr) {}
	~NeuralNetwork()
	{
		int i;
		for (i = 0; i < num_params_; i++)
		{
			delete network_param_array_[i].name_ptr;
		}

		delete network_param_array_;
		delete network_module_array_;
	}

	template<typename T>
	T* register_module(const char* module_name, T&& temp )
	{
		T* network_module;
		size_t len;
		int index = 0; // TODO add index to param name (if named)

		network_module = new T(temp);
		*network_module = temp;
		network_module->init();


		for (auto& param : network_module->get_all_weights())
		{
			network_param_array_ = (NetworkParms*)BlockRealloc(network_param_array_, num_params_ * sizeof(NetworkParms), (num_params_ + 1) * sizeof(NetworkParms));
			if (module_name)
			{
				len = strlen(module_name);
				network_param_array_[num_params_].name_ptr = new char[len + 1];
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
				strcpy_s(network_param_array_[num_params_].name_ptr, len + 1, module_name);
#else
				strncpy(network_param_array_[num_params_].name_ptr, module_name, len + 1);
#endif
			}
			else
			{
				network_param_array_[num_params_].name_ptr = nullptr; // g++ does not like '\0' here
			}

			network_param_array_[num_params_].param_data_ = nullptr;
			network_param_array_[num_params_++].param_ = param;		
		}

		network_module_array_ = (NetworkModules*)BlockRealloc(network_module_array_, num_modules_ * sizeof(NetworkModules), (num_modules_ + 1) * sizeof(NetworkModules));
		network_module_array_[num_modules_].module_ = network_module;
		num_modules_++;

		return network_module;
	}

	NetworkParms* get_parameters(int* num_params_ptr)
	{ 
		if (!num_params_ptr)
		{
			return nullptr;
		}

		*num_params_ptr = num_params_; 
		return network_param_array_; 
	}

	void to(device target_device, int target_device_index = 0)
	{
		int i;

		for (i = 0; i < num_modules_; i++)
		{
			network_module_array_[i].module_->to(target_device, target_device_index);
		}

		
		for (i = 0; i < num_params_; i++)
		{
			if (network_param_array_[i].param_data_)
			{
				*network_param_array_[i].param_data_ = network_param_array_[i].param_data_->to(target_device, target_device_index);
			}
		}
		
	}

	void train(bool is_training)
	{
		int i;

		for (i = 0; i < num_modules_; i++)
		{
			network_module_array_[i].module_->train(is_training);
		}
	}

	Module* get_module(int index)
	{
		if (index >= num_modules_)
		{
			return nullptr;
		}

		return network_module_array_[index].module_;
	}


	int save_checkpoint(const char* checkpoint_path)
	{
		int i;
		FILE* stream;
		size_t bytes;
		size_t data_size;
		device device_type;
		void* data = nullptr;


		device_type = network_param_array_[0].param_->get_device(); // key off first param

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    errno_t err;
		err = fopen_s(&stream, checkpoint_path, "wb");
		if (err)
		{
			return -1;
		}
#else
		stream = fopen(checkpoint_path, "wb");
		if (stream)
		{
			return -1;
		}
#endif
		for (i = 0; i < num_params_; i++)
		{
			data_size = network_param_array_[i].param_->get_numels() * sizeof(float);
			if (CPU == device_type)
			{
				data = network_param_array_[i].param_->get_data_ptr();
			}
			else
			{
				if (GPU == device_type)
				{
#ifdef USE_CUDA
					data = new char[data_size];
					CopyDataFromGPU(data, network_param_array_[i].param_->get_data_ptr(), data_size);
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid device");
				}
			}
			bytes = fwrite(data, sizeof(char), data_size, stream);
			if (bytes != data_size)
			{
				fclose(stream);
				return -1;
			}

			
			if (GPU == device_type)
			{
				delete static_cast<char*>(data);
			}

		}

		fclose(stream);
		return 0;
	}

	int load_checkpoint(const char* checkpoint_path)
	{
		int i;
		FILE* stream;
		size_t bytes;
		size_t data_size;
		device device_type;
		void* data = nullptr;


		device_type = network_param_array_[0].param_->get_device(); // key off first param

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    errno_t err;
		err = fopen_s(&stream, checkpoint_path, "rb");
		if (err)
		{
			return -1;
		}
#else
		stream = fopen(checkpoint_path, "rb");
		if (stream)
		{
			return -1;
		}
#endif
		for (i = 0; i < num_params_; i++)
		{
			data_size = network_param_array_[i].param_->get_numels() * sizeof(float);
			if (CPU == device_type)
			{
				data = network_param_array_[i].param_->get_data_ptr();
			}
			else
			{
				if (GPU == device_type)
				{
#ifdef USE_CUDA
					data = new char[data_size];
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid device");
				}
			}

			bytes = fread(data, sizeof(char), data_size, stream);
			if (bytes != data_size)
			{
				fclose(stream);
				return -1;
			}

			if (GPU == device_type)
			{
#ifdef USE_CUDA
				CopyDataToGPU(network_param_array_[i].param_->get_data_ptr(), data, data_size);
				delete static_cast<char*>(data);
#endif
			}
		}

		fclose(stream);
		return 0;
	}

private:
	int num_params_;
	NetworkParms* network_param_array_;

	int num_modules_;
	NetworkModules* network_module_array_;
};
}
#endif // NET_H
