#include <iostream>
#include "tensor.h"
/*
MIT License

Copyright (c) 2021 adeobootpin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

namespace lten {

	Tensor TensorFromBuffer(const uint64_t* dims_ptr, int ndims, void* data_ptr, bool own_memory, TensorOps* options_ptr)
	{
		dtype data_type;

		if (options_ptr)
		{
			data_type = options_ptr->data_type;
		}
		else
		{
			data_type = FLOAT32;
		}

		if (data_type == FLOAT32)
		{
			TensorImpl<float>* tensorImpl;
			tensorImpl = new TensorImpl<float>;

			intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

			tensorImpl->allocate_from_buffer(dims_ptr, ndims, data_ptr, own_memory, options_ptr);

			return Tensor(intr_ptr);
		}
		else
		{
			if (data_type == INT32)
			{
				TensorImpl<int>* tensorImpl;
				tensorImpl = new TensorImpl<int>;

				intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

				tensorImpl->allocate_from_buffer(dims_ptr, ndims, data_ptr, own_memory, options_ptr);

				return Tensor(intr_ptr);
			}
			else
			{
				if (data_type == UINT8)
				{
					TensorImpl<uint8_t>* tensorImpl;
					tensorImpl = new TensorImpl<uint8_t>;

					intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

					tensorImpl->allocate_from_buffer(dims_ptr, ndims, data_ptr, own_memory, options_ptr);

					return Tensor(intr_ptr);
				}

				LTEN_ERR("Invalid tensor data type");
				return Tensor(); // keep compiler quiet
			}
		}
	}

	Tensor TensorFromBuffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory, TensorOps* options_ptr)
	{
		dtype data_type;

		if (options_ptr)
		{
			data_type = options_ptr->data_type;
		}
		else
		{
			data_type = FLOAT32;
		}

		if (data_type == FLOAT32)
		{
			TensorImpl<float>* tensorImpl;
			tensorImpl = new TensorImpl<float>;

			intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

			tensorImpl->allocate_from_buffer(dims, data_ptr, own_memory, options_ptr);

			return Tensor(intr_ptr);
		}
		else
		{
			if (data_type == INT32)
			{
				TensorImpl<int>* tensorImpl;
				tensorImpl = new TensorImpl<int>;

				intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

				tensorImpl->allocate_from_buffer(dims, data_ptr, own_memory, options_ptr);

				return Tensor(intr_ptr);
			}
			else
			{
				if (data_type == UINT8)
				{
					TensorImpl<uint8_t>* tensorImpl;
					tensorImpl = new TensorImpl<uint8_t>;

					intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

					tensorImpl->allocate_from_buffer(dims, data_ptr, own_memory, options_ptr);

					return Tensor(intr_ptr);
				}

				LTEN_ERR("Invalid tensor data type");
				return Tensor(); // keep compiler quiet
			}
		}
	}

	Tensor TensorFromBuffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr)
	{
		dtype data_type;

		if (options_ptr)
		{
			data_type = options_ptr->data_type;
		}
		else
		{
			data_type = FLOAT32;
		}

		if (data_type == FLOAT32)
		{
			TensorImpl<float>* tensorImpl;
			tensorImpl = new TensorImpl<float>;

			intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

			tensorImpl->allocate_from_buffer(dims, data_ptr, own_data_memory, gradient_ptr, own_gradient_memory, options_ptr);

			return Tensor(intr_ptr);
		}
		else
		{
			if (data_type == INT32)
			{
				TensorImpl<int>* tensorImpl;
				tensorImpl = new TensorImpl<int>;

				intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

				tensorImpl->allocate_from_buffer(dims, data_ptr, own_data_memory, gradient_ptr, own_gradient_memory, options_ptr);

				return Tensor(intr_ptr);
			}
			else
			{
				if (data_type == UINT8)
				{
					TensorImpl<uint8_t>* tensorImpl;
					tensorImpl = new TensorImpl<uint8_t>;

					intrusive_ptr<TensorImplBase> intr_ptr(tensorImpl);

					tensorImpl->allocate_from_buffer(dims, data_ptr, own_data_memory, gradient_ptr, own_gradient_memory, options_ptr);

					return Tensor(intr_ptr);
				}
				LTEN_ERR("Invalid tensor data type");
				return Tensor(); // keep compiler quiet
			}
		}
	}

	Tensor AllocateTensor(const std::initializer_list<uint64_t>& dims, TensorOps* options)
	{
		return TensorFromBuffer(dims, nullptr, true, options);
	}

	Tensor AllocateTensor(const uint64_t* dims_ptr, int ndims, TensorOps* options)
	{
		return TensorFromBuffer(dims_ptr, ndims, nullptr, true, options);
	}


	Tensor RandomTensor(const std::initializer_list<uint64_t>& dims, TensorOps* options)
	{
		uint64_t numels;
		uint64_t i;
		dtype data_type;
		void* data_ptr = nullptr;

		numels = 1;
		for (uint64_t dim : dims)
		{
			numels *= dim;
		}

		if (options)
		{
			data_type = options->data_type;
		}
		else
		{
			data_type = FLOAT32;
		}

		if (data_type == FLOAT32)
		{
			data_ptr = new float[numels];
			for (i = 0; i < numels; i++)
			{
				static_cast<float*>(data_ptr)[i] = rand() % 1000 * 0.001f;
				if (!(rand() % 2))
				{
					static_cast<float*>(data_ptr)[i] *= -1.0f;
				}
			}
		}
		else
		{
			if (data_type == INT32)
			{
				LTEN_ERR("Not yet implemented: RandomTensor INT32");
				data_ptr = new int[numels];
			}
			else
			{
				if (data_type == UINT8)
				{
					LTEN_ERR("Not yet implemented: RandomTensor UINT8");
					data_ptr = new uint8_t[numels];
				}
				else
				{
					LTEN_ERR("Invalid tensor data type");
					//return Tensor(); // keep compiler quiet
				}
			}
		}

		return TensorFromBuffer(dims, data_ptr, true, options);
	}

	Tensor RandomTensor(const uint64_t* dims_ptr, int ndims, TensorOps* options)
	{
		uint64_t numels;
		uint64_t i;
		dtype data_type;
		void* data_ptr = nullptr;

		numels = 1;
		for (i = 0; i < ndims; i++)
		{
			numels *= dims_ptr[i];
		}

		if (options)
		{
			data_type = options->data_type;
		}
		else
		{
			data_type = FLOAT32;
		}

		if (data_type == FLOAT32)
		{
			data_ptr = new float[numels];
			for (i = 0; i < numels; i++)
			{
				static_cast<float*>(data_ptr)[i] = rand() % 1000 * 0.001f;
				if (!(rand() % 2))
				{
					static_cast<float*>(data_ptr)[i] *= -1.0f;
				}
			}
		}
		else
		{
			if (data_type == INT32)
			{
				LTEN_ERR("Not yet implemented");
				data_ptr = new int[numels];
			}
			else
			{
				if (data_type == UINT8)
				{
					LTEN_ERR("Not yet implemented");
					data_ptr = new uint8_t[numels];
				}
				else
				{
					LTEN_ERR("Invalid tensor data type");
					//return Tensor(); // keep compiler quiet
				}
			}
		}

		return TensorFromBuffer(dims_ptr, ndims, data_ptr, true, options);
	}

}


std::ostream& operator<<(std::ostream& out, const lten::Tensor& tensor)
{
	if (tensor.get_device() == lten::GPU)
	{
		LTEN_ERR("<< only supports the CPU device type"); // TODO add support for displaying GPU tensors
	}

	if (tensor.get_data_type() == lten::FLOAT32)
	{
		out << tensor.get_mdarray<float>();
	}
	else
	{
		if (tensor.get_data_type() == lten::INT32)
		{
			out << tensor.get_mdarray<int>();
		}
		else
		{
			if (tensor.get_data_type() == lten::UINT8)
			{
				out << tensor.get_mdarray<uint8_t>();
			}
		}
	}

	return out;
}


void PrintBuildInfo()
{
	std::cout << "-------------------" << std::endl;
#ifdef USE_CUDA
	std::cout << "USE_CUDA: true" << std::endl;
#else
	std::cout << "USE_CUDA: false" << std::endl;
#endif

#ifdef USE_AVX_256
	std::cout << "USE_AVX_256: true" << std::endl;
#else
	std::cout << "USE_AVX_256: false" << std::endl;
#endif

#ifdef USE_OPENBLAS
	std::cout << "USE_OPENBLAS: true" << std::endl;
#else
	std::cout << "USE_OPENBLAS: false" << std::endl;
#endif

#ifdef USE_THREADPOOL
	std::cout << "USE_THREADPOOL: true" << std::endl;
#else
	std::cout << "USE_THREADPOOL: false" << std::endl;
#endif

#ifdef USE_MEMORYPOOL
	std::cout << "USE_MEMORYPOOL: true" << std::endl;
#else
	std::cout << "USE_MEMORYPOOL: false" << std::endl;
#endif

	std::cout << "-------------------" << std::endl;
}