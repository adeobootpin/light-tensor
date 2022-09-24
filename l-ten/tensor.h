#ifndef TENSOR_H
#define TENSOR_H

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

#include "shared_pointer.h"
#include "tensorimpl.h"
#include "tensor_fns.h"

namespace lten {

	class Tensor
	{
	public:
		Tensor()
		{
		}

		~Tensor()
		{
		}


		Tensor(intrusive_ptr<TensorImplBase> smart_ptr)
		{
			smart_ptr_ = smart_ptr;
			assert(smart_ptr_.get_real_object() != nullptr);
		}

#ifdef USE_MEMORYPOOL
		void* operator new(size_t size)
		{
			return lten::MISC_globals::singleton()->get_cpu_memorypool()->AllocateMemory(size);
		}


		void operator delete(void* memory)
		{
			return lten::MISC_globals::singleton()->get_cpu_memorypool()->FreeMemory(memory);
		}
#endif	
		void* get_data_ptr() { return smart_ptr_->get_data_ptr(); }
		void* get_grad_ptr() { return smart_ptr_->get_grad_ptr(); }
		uint64_t get_numels() { return smart_ptr_->get_numels(); }
		void set_autograd(bool setting) { smart_ptr_->set_autograd(setting); }
		bool autograd_on() { return smart_ptr_->autograd_on(); }
		const uint64_t* get_sizes() { return smart_ptr_->get_sizes(); }
		const uint64_t* get_strides() { return smart_ptr_->get_strides(); }
		void backward(MultiDimArray<float>* top_gradient = nullptr) { smart_ptr_->backward(top_gradient); }
		void clear_gradients() { smart_ptr_->clear_gradients(); }
		int get_ndims() const { return smart_ptr_->get_ndims(); }
		dtype get_data_type() const { return smart_ptr_->get_data_type(); }
		device get_device() const { return smart_ptr_->get_device(); }
		int get_device_index() const { return smart_ptr_->get_device_index(); }


		Tensor& operator=(const Tensor& x) &
		{
			smart_ptr_ = x.smart_ptr_;
			return *this;
		}

		Tensor matmul(const Tensor& other)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type != other.smart_ptr_->get_data_type())
			{
				LTEN_ERR("Tensor data types must match");
			}

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->matmul(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(other.smart_ptr_.get_real_object()));

				return Tensor(result);
			}

			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->matmul(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(other.smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->matmul(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(other.smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor operator[](int index)
		{
			const uint64_t* dims_ptr;
			dtype data_type;
			dims_ptr = smart_ptr_->get_sizes();

			if (index >= dims_ptr[0])
			{
				LTEN_ERR("Index out of range");
			}

			data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->sub_array(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), index);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->sub_array(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), index);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->sub_array(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), index);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();

		}

		Tensor index(const Tensor& index)
		{
			dtype data_type;

			/*
			const uint64_t* dims_ptr;	
			dims_ptr = smart_ptr_->get_sizes();
			
			if (index >= dims_ptr[0])
			{
				LTEN_ERR("Index out of range");
			}
			*/

			data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->index(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(index.smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					//resultImpl->sub_array(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), index);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						//resultImpl->sub_array(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), index);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor operator+(const Tensor& other)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type != other.smart_ptr_->get_data_type())
			{
				LTEN_ERR("Tensor data types must match");
			}

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->add(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(other.smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->add(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(other.smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->add(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(other.smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor operator-(const Tensor& other)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type != other.smart_ptr_->get_data_type())
			{
				LTEN_ERR("Tensor data types must match");
			}

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->sub(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(other.smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->sub(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(other.smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->sub(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(other.smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor div(const Tensor& other)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type != other.smart_ptr_->get_data_type())
			{
				LTEN_ERR("Tensor data types must match");
			}

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->div(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(other.smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->div(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(other.smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->div(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(other.smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor exp()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->exp(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->exp(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->exp(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}





		Tensor max()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->max(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->max(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->max(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}

				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor max(int dim)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->max(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dim);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->max(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dim);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->max(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dim);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor sum()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->sum(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->sum(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->sum(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor sum(int dim)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->sum(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dim);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->sum(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dim);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->sum(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dim);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor log()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->log(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->log(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->log(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor sig()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->sig(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->sig(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->sig(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor tanh()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->tanh(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->tanh(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->tanh(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor sqrt()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->sqrt(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->sqrt(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->sqrt(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor mean()
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->mean(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->mean(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->mean(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor mean(uint32_t axis)
		{
			return mean(&axis, 1);
		}

		Tensor mean(const uint32_t* axes, int naxes)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->mean(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), axes, naxes);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->mean(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), axes, naxes);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->mean(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), axes, naxes);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor var(uint32_t axis)
		{
			return var(&axis, 1);
		}

		Tensor var(const uint32_t* axes, int naxes)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->var(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), axes, naxes);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->var(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), axes, naxes);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->var(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), axes, naxes);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor std(int dim)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->std(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dim);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->std(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dim);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->std(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dim);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}


		Tensor operator*(float scalar)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->scalar_mul(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), scalar);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->scalar_mul(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), scalar);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->scalar_mul(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), scalar);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}


		Tensor operator*(const Tensor& other)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type != other.smart_ptr_->get_data_type())
			{
				LTEN_ERR("Tensor data types must match");;
			}

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->mul(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(other.smart_ptr_.get_real_object()));

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->mul(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(other.smart_ptr_.get_real_object()));

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->mul(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(other.smart_ptr_.get_real_object()));

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}


		friend Tensor operator*(float scalar, Tensor& rhs)
		{
			return rhs * scalar;
		}

		Tensor cat(const Tensor& other, int dim = 0)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->cat(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(other.smart_ptr_.get_real_object()), dim);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->cat(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(other.smart_ptr_.get_real_object()), dim);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->cat(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(other.smart_ptr_.get_real_object()), dim);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}


		Tensor reshape(const uint64_t* dims, int ndims)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->reshape(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dims, ndims);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->reshape(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dims, ndims);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->reshape(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dims, ndims);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor reshape(const std::initializer_list<uint64_t>& dims)
		{
			uint64_t dims_array[MAX_DIMS];
			int i;
			int ndims;

			ndims = static_cast<int>(dims.size());

			if (ndims > MAX_DIMS)
			{
				LTEN_ERR("ndims > MAX_DIMS");
			}

			i = 0;
			for (uint64_t dim : dims)
			{
				dims_array[i++] = dim;
			}


			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->reshape(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dims_array, ndims);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->reshape(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dims_array, ndims);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->reshape(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dims_array, ndims);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor squeeze(int dim)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->squeeze(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dim);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->squeeze(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dim);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->squeeze(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dim);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor unsqueeze(int dim)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->unsqueeze(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dim);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->unsqueeze(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dim);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->unsqueeze(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dim);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor transpose(int dim1, int dim2)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->transpose(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dim1, dim2);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->transpose(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dim1, dim2);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->transpose(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dim1, dim2);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor to(device target_device, int device_index = 0)
		{
			if (smart_ptr_->get_data_type() == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->to(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), target_device, device_index);

				return Tensor(result);
			}
			else
			{
				if (smart_ptr_->get_data_type() == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->to(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), target_device, device_index);

					return Tensor(result);
				}
				else
				{
					if (smart_ptr_->get_data_type() == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->to(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), target_device, device_index);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();

		}

		Tensor masked_fill(const Tensor& mask, double value)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->masked_fill(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<float>*>(mask.smart_ptr_.get_real_object()), value);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->masked_fill(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(mask.smart_ptr_.get_real_object()), value);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->masked_fill(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(mask.smart_ptr_.get_real_object()), value);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		Tensor repeat(const std::initializer_list<uint32_t>& repeats)
		{
			uint32_t rpts[MAX_DIMS];
			int i;
			int ndims;

			ndims = static_cast<int>(repeats.size());

			if (ndims > MAX_DIMS)
			{
				LTEN_ERR("ndims > MAX_DIMS");
			}

			i = 0;
			for (uint32_t rep : repeats)
			{
				rpts[i++] = rep;
			}

			return repeat(rpts, ndims);
		}

		Tensor repeat(const uint32_t* repeats, int nrepeats)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->repeat(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), repeats, nrepeats);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					//resultImpl->masked_fill(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(mask.smart_ptr_.get_real_object()), value);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						//resultImpl->masked_fill(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(mask.smart_ptr_.get_real_object()), value);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		// scratch is optional but may make processing faster by avoiding memory allocations
		// Note: nrepeats can be extremely large
		// Optional buffer scratch, if provided should be of size nrepeats + 1
		// It will be used to store the cummulative repeats array (see md_array repeat_interleave funs)
		// Note: if backward processing is required, scratch *must* point to memory that is valid during backward processing
		// Note: when using broadcast mode, scratch, if provided should be of size dims[dim] + 1 (i.e. actual number of repeats + 1)

		Tensor repeat_interleave(uint32_t repeat, int dim, uint32_t* scratch = nullptr) // broadcast version of function (repeat is broadcast to all 'rows' of dim)
		{
			return repeat_interleave(&repeat, 1, dim, scratch); // call general form of function
		}

		Tensor repeat_interleave(const uint32_t* repeats, int nrepeats, int dim, uint32_t* scratch = nullptr)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->repeat_interleave(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), repeats, nrepeats, dim, scratch);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					//resultImpl->masked_fill(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<int>*>(mask.smart_ptr_.get_real_object()), value);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						//resultImpl->masked_fill(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), *static_cast<TensorImpl<uint8_t>*>(mask.smart_ptr_.get_real_object()), value);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}


		/*
		Tensor reshape(const std::initializer_list<uint64_t>& dims)
		{
			uint64_t dims_array[MAX_DIMS];
			int i;
			int ndims;

			ndims = static_cast<int>(dims.size());

			if (ndims > MAX_DIMS)
			{
				LTEN_ERR("ndims > MAX_DIMS");
			}

			i = 0;
			for (uint64_t dim : dims)
			{
				dims_array[i++] = dim;
			}
		*/

		Tensor permute(const std::initializer_list<uint32_t>& perms)
		{
			uint32_t permutations[MAX_DIMS];
			int i;
			int ndims;

			ndims = static_cast<int>(perms.size());

			if (ndims > MAX_DIMS)
			{
				LTEN_ERR("ndims > MAX_DIMS");
			}

			i = 0;
			for (uint32_t perm : perms)
			{
				permutations[i++] = perm;
			}

			return permute(permutations, ndims);
		}

		Tensor permute(const uint32_t* permutations, int npermutations)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->permute(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), permutations, npermutations);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					//resultImpl->permute(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), permutations, npermutations);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						//resultImpl->permute(*static_cast<TensorImpl<uint8>*>(smart_ptr_.get_real_object()), permutations, npermutations);

						return Tensor(result);
					}
				}
			}

			LTEN_ERR("Invalid tesor data type");
			return Tensor();
		}

		template<typename Dtype>
		MultiDimArray<Dtype>* get_mdarray() const
		{
			return static_cast<TensorImpl<Dtype>*>(smart_ptr_.get_real_object())->get_mdarray();
		}

		template<typename Dtype>
		MultiDimArray<Dtype>* get_gradients_mdarray() const
		{
			return static_cast<TensorImpl<Dtype>*>(smart_ptr_.get_real_object())->get_gradients_mdarray();
		}



		intrusive_ptr<TensorImplBase> get_smart_ptr() { return smart_ptr_; }

	protected:
		intrusive_ptr<TensorImplBase> smart_ptr_;
	};

	Tensor TensorFromBuffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory = false, TensorOps* options_ptr = nullptr);
	Tensor TensorFromBuffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr = nullptr);
	Tensor TensorFromBuffer(const uint64_t* dims_ptr, int ndims, void* data_ptr, bool own_memory = false, TensorOps* options_ptr = nullptr);
	Tensor AllocateTensor(const std::initializer_list<uint64_t>& dims, TensorOps* options_ptr = nullptr);
	Tensor AllocateTensor(const uint64_t* dims_ptr, int ndims, TensorOps* options_ptr = nullptr);

	Tensor RandomTensor(const std::initializer_list<uint64_t>& dims, TensorOps* options = nullptr);
	Tensor RandomTensor(const uint64_t* dims_ptr, int ndims, TensorOps* options = nullptr);


}

void PrintBuildInfo();
std::ostream& operator<<(std::ostream& out, const lten::Tensor& tensor);

#endif // TENSOR_H