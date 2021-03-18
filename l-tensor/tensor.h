#ifndef TENSOR_H
#define TENSOR_H

#include "shared_pointer.h"
#include "tensorimpl.h"

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
		
	
		void* get_data_ptr() { return smart_ptr_->get_data_ptr(); }
		void* get_grad_ptr() { return smart_ptr_->get_grad_ptr(); }
		const uint64_t get_numels() { return smart_ptr_->get_numels(); }
		void set_autograd(bool setting) { smart_ptr_->set_autograd(setting); }
		const uint64_t* get_sizes() { return smart_ptr_->get_sizes(); }
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


		friend Tensor operator*(float fScalar, Tensor& rhs)
		{
			return rhs * fScalar;
		}

		Tensor reshape(const std::initializer_list<uint64_t>& dims)
		{
			dtype data_type = smart_ptr_->get_data_type();

			if (data_type == FLOAT32)
			{
				TensorImpl<float>* resultImpl;

				resultImpl = new TensorImpl<float>;

				intrusive_ptr<TensorImplBase> result(resultImpl);

				resultImpl->reshape(*static_cast<TensorImpl<float>*>(smart_ptr_.get_real_object()), dims);

				return Tensor(result);
			}
			else
			{
				if (data_type == INT32)
				{
					TensorImpl<int>* resultImpl;

					resultImpl = new TensorImpl<int>;

					intrusive_ptr<TensorImplBase> result(resultImpl);

					resultImpl->reshape(*static_cast<TensorImpl<int>*>(smart_ptr_.get_real_object()), dims);

					return Tensor(result);
				}
				else
				{
					if (data_type == UINT8)
					{
						TensorImpl<uint8_t>* resultImpl;

						resultImpl = new TensorImpl<uint8_t>;

						intrusive_ptr<TensorImplBase> result(resultImpl);

						resultImpl->reshape(*static_cast<TensorImpl<uint8_t>*>(smart_ptr_.get_real_object()), dims);

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
	
}

void PrintBuildInfo();
std::ostream& operator<<(std::ostream& out, const lten::Tensor& tensor);

#endif // TENSOR_H