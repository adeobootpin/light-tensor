#ifndef MD_ARRAY_H
#define MD_ARRAY_H

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

#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdlib.h>
#include "math_fns.h"
#include "error.h"


class md_array_dim_iterator
{
public:
	md_array_dim_iterator(const uint64_t* dims_ptr, int ndims, uint64_t numels = 0)
	{
		uint64_t i;

		assert(ndims);

		dims_ = dims_ptr;
		memset(indices_, 0, sizeof(uint64_t) * ndims);
		index_ = 0;

		ndims_ = ndims;
		numels_ = numels;

		if (!numels)
		{
			numels_ = 1;
			for (i = 0; i < ndims; i++)
			{
				numels_ *= dims_[i];
			}
		}
	}

	md_array_dim_iterator(uint64_t index)
	{
		index_ = index;
	}


	md_array_dim_iterator begin()
	{
		return md_array_dim_iterator(dims_, ndims_, numels_);
	}

	md_array_dim_iterator end()
	{
		return md_array_dim_iterator(numels_);
	}

	bool operator!=(const md_array_dim_iterator& other)
	{
		return index_ != other.index_;
	}


	md_array_dim_iterator& operator++()
	{
		int i;
		uint64_t carry;

		if (index_ < numels_)
		{
			carry = 1;
			for (i = ndims_ - 1; i >= 0; i--)
			{
				indices_[i] += carry;

				assert(indices_[i] <= dims_[i]);

				if (indices_[i] == dims_[i])
				{
					indices_[i] = 0;
					carry = 1;
				}
				else
				{
					carry = 0;
					break;
				}
			}
			index_++;
		}

		return *this;
	}


	md_array_dim_iterator operator++(int)
	{
		md_array_dim_iterator it = *this;
		++*this;
		return it;
	}


	const uint64_t* operator*()
	{
		return indices_;
	}


private:
	int ndims_;
	uint64_t numels_;
	const uint64_t* dims_;
	uint64_t indices_[MAX_DIMS];
	uint64_t index_;

};


template<typename Dtype>
class MultiDimArray
{
public:
	MultiDimArray()
	{
		Reset();
	}

	MultiDimArray(const std::initializer_list<uint64_t>& dims, Dtype* buffer_to_use_ptr, bool own_memory = true)
	{
		Reset();
		Allocate(dims, buffer_to_use_ptr, own_memory);
	}

	MultiDimArray(const uint64_t* dims_ptr, int ndims, Dtype* buffer_to_use_ptr, bool own_memory = true)
	{
		Reset();
		Allocate(dims_ptr, ndims, buffer_to_use_ptr, own_memory);
	}


	MultiDimArray(const MultiDimArray& other)
	{
		Reset();
		Allocate(other.GetSizes(), other.GetNDims());
		memcpy(data_ptr_, other.GetDataPtr(), sizeof(Dtype) * numels_);
	}


	virtual ~MultiDimArray()
	{
		if (own_memory_)
		{
			delete data_ptr_;
		}
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

	const uint64_t GetNumels() const { return numels_; }
	const uint64_t* GetSizes() const { return dims_array_; }
	const uint64_t* GetStrides() const { return strides_array_; }
	int GetNDims() const { return ndims_; }
	void SetMemoryOwnership(bool own_memory) { own_memory_ = own_memory; }

	MultiDimArray<uint64_t> GetSizesMdArray()
	{
		return MultiDimArray<uint64_t>({ (uint64_t)ndims_ }, dims_array_, false);
	}


	virtual int Allocate(const uint64_t* dims_ptr, int ndims, Dtype* buffer_to_use_ptr = nullptr, bool own_memory = true)
	{
		int i;
		int64_t numels;

		if (ndims > MAX_DIMS)
		{
			LTEN_ERR("ndims > MAX_DIMS");
		}

		ndims_ = ndims;

		numels = 1;
		for (i = ndims - 1; i >= 0; i--)
		{
			strides_array_[i] = numels;
			dims_array_[i] = dims_ptr[i];
			numels *= dims_array_[i];
		}

		if (buffer_to_use_ptr)
		{
			if (data_ptr_ && own_memory_)
			{
				delete data_ptr_;
			}
			data_ptr_ = buffer_to_use_ptr;
		}
		else
		{
			if (!own_memory_ || (numels_ != numels)) // (conservatively) avoid a memory allocation where possible
			{
				if (data_ptr_ && own_memory_)
				{
					delete data_ptr_;
				}

				data_ptr_ = new Dtype[numels];
				if (!data_ptr_)
				{
					std::terminate(); // no hope, bail
				}
			}
		}

		numels_ = numels;
		own_memory_ = own_memory;

		return 0;

	}

	virtual int Allocate(const std::initializer_list<uint64_t>& dims, Dtype* buffer_to_use_ptr = nullptr, bool own_memory = true)
	{
		uint64_t dims_array[MAX_DIMS];
		int i;
		size_t ndims;
		int ret;

		ndims = dims.size();

		if (ndims > MAX_DIMS)
		{
			LTEN_ERR("ndims > MAX_DIMS");
		}

		i = 0;
		for (uint64_t dim : dims)
		{
			dims_array[i++] = dim;
		}

		ret = Allocate(dims_array, static_cast<int>(ndims), buffer_to_use_ptr, own_memory);

		return ret;
	}

	int Reshape(int ndims) const
	{
		int i;
		int len_diff;
		int collapsible;
		int index;
		uint64_t dims_array[MAX_DIMS];
		uint64_t numels;

		if (ndims == ndims_)
		{
			return 0;
		}

		if (ndims > ndims_)
		{
			len_diff = ndims - ndims_;
			for (i = 0; i < len_diff; i++)
			{
				dims_array[i] = 1;
			}
			memcpy(&dims_array[len_diff], dims_array_, sizeof(uint64_t) * ndims_);

			numels = 1;
			for (i = ndims - 1; i >= 0; i--)
			{
				strides_array_[i] = numels;
				numels *= dims_array[i];
			}

			assert(numels == numels_); // sanity check

			memcpy(dims_array_, dims_array, sizeof(uint64_t) * ndims);
		}
		else
		{
			collapsible = 0;
			for (i = 0; i < ndims_; i++)
			{
				if (dims_array_[i] == 1)
				{
					collapsible++;
				}
			}

			if (collapsible < ndims_ - ndims)
			{
				LTEN_ERR("Not enough collapsible dimensions");
				return -1;
			}

			collapsible = ndims_ - ndims; // only collapse as many as requested
			index = 0;
			for (i = 0; i < ndims_; i++) // collapse from most significant dim
			{
				if ((dims_array_[i] != 1) || (collapsible <= 0)) // not a 1 or if we are done collapsing
				{
					dims_array[index++] = dims_array_[i];
				}
				else
				{
					collapsible--;
				}
			}

			numels = 1;
			for (i = ndims - 1; i >= 0; i--)
			{
				strides_array_[i] = numels;
				numels *= dims_array[i];
			}

			assert(numels == numels_); // sanity check
			memcpy(dims_array_, dims_array, sizeof(uint64_t) * ndims);

		}

		ndims_ = ndims;

		return 0;
	}

	virtual void ReleaseResources()
	{
		if (own_memory_)
		{
			delete data_ptr_;
		}
		data_ptr_ = nullptr;
	}

	Dtype& operator()(const uint64_t* dims, int ndims, bool broadcast = false) const
	{
		return data_ptr_[GetOffset(dims, ndims, broadcast)];
	}

	Dtype* GetDataPtr() const
	{
		return data_ptr_;
	}

	Dtype* GetDataPtr(const uint64_t* coordinates, int ndims, bool broadcast = false) const
	{
		return &data_ptr_[GetOffset(coordinates, ndims, broadcast)];
	}


	// fast custom accessors
	// 9x faster than operator()(std::initializer_list<int> indices)
	Dtype& operator()(int dim_1, int dim_2, bool broadcast = false)
	{
		if (ndims_ != 2)
		{
			LTEN_ERR("Invalid number of input dims (must be 2)");
		}

		if (broadcast)
		{
			dim_1 = std::min(dim_1, (int)dims_array_[0] - 1);
			dim_2 = std::min(dim_2, (int)dims_array_[1] - 1);
		}

		return data_ptr_[dim_1 * dims_array_[1] + dim_2];
	}



	Dtype& operator()(uint64_t dim_1, uint64_t dim_2, uint64_t dim_3, uint64_t dim_4, bool broadcast = false)
	{
		if (ndims_ != 4)
		{
			LTEN_ERR("Invalid number of input dims (must be 4)");
		}

		if (broadcast)
		{
			dim_1 = std::min(dim_1, dims_array_[0] - 1);
			dim_2 = std::min(dim_2, dims_array_[1] - 1);
			dim_3 = std::min(dim_3, dims_array_[2] - 1);
			dim_4 = std::min(dim_4, dims_array_[3] - 1);
		}

		return data_ptr_[dim_1 * (dims_array_[1] * dims_array_[2] * dims_array_[3]) + dim_2 * (dims_array_[2] * dims_array_[3]) + dim_3 * dims_array_[3] + dim_4];
	}

	Dtype& operator()(uint64_t dim_1, uint64_t dim_2, uint64_t dim_3, uint64_t dim_4, uint64_t dim_5, bool broadcast = false)
	{
		if (ndims_ != 5)
		{
			LTEN_ERR("Invalid number of input dims (must be 5)");
		}
		if (broadcast)
		{
			dim_1 = std::min((int)dim_1, (int)dims_array_[0] - 1);
			dim_2 = std::min((int)dim_2, (int)dims_array_[1] - 1);
			dim_3 = std::min((int)dim_3, (int)dims_array_[2] - 1);
			dim_4 = std::min((int)dim_4, (int)dims_array_[3] - 1);
			dim_5 = std::min((int)dim_5, (int)dims_array_[4] - 1);
		}

		return data_ptr_[
			dim_1 * (dims_array_[1] * dims_array_[2] * dims_array_[3] * dims_array_[4]) +
				dim_2 * (dims_array_[2] * dims_array_[3] * dims_array_[4]) +
				dim_3 * (dims_array_[3] * dims_array_[4]) +
				dim_4 * (dims_array_[4]) +
				dim_5];
	}


	MultiDimArray operator[](uint64_t index) const
	{
		uint64_t stride;
		int i;
		uint64_t dim;

		if (index >= dims_array_[0]);
		{
			LTEN_ERR("Index out of range");
		}

		if (ndims_ == 1)
		{
			dim = 1;
			return MultiDimArray(&dim, 1, &data_ptr_[index], false);
		}
		else
		{
			stride = 1;
			for (i = 1; i < ndims_; i++)
			{
				stride *= dims_array_[i];
			}

			return MultiDimArray(&dims_array_[1], ndims_ - 1, &data_ptr_[stride * index], false);
		}
	}

	MultiDimArray& operator=(Dtype scalar)
	{
		if ((ndims_ != 1) || (dims_array_[0] != 1))
		{
			LTEN_ERR("MultiDimArray must have dimension 1");
		}
		data_ptr_[0] = scalar;
		return *this;
	}

	virtual MultiDimArray& operator=(const MultiDimArray& other)
	{
		Allocate(other.GetSizes(), other.GetNDims());
		memcpy(data_ptr_, other.GetDataPtr(), sizeof(Dtype) * numels_);
		return *this;
	}

	virtual MultiDimArray& operator=(MultiDimArray&& other)
	{
		if (own_memory_)
		{
			delete data_ptr_;
		}
		Allocate(other.GetSizes(), other.GetNDims(), other.data_ptr_, other.own_memory_);
		other.own_memory_ = false;
		other.data_ptr_ = nullptr;
		return *this;
	}

	MultiDimArray operator+(const MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		uint64_t dims[MAX_DIMS];
		uint64_t i;
		MultiDimArray result;
		uint64_t h, w;
		uint64_t H, W;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			H = dims_result[ndims_ - 2];
			W = dims_result[ndims_ - 1];

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					memcpy(dims, higher_indices, sizeof(uint64_t) * (ndims_ - 2));
					for (h = 0; h < H; h++)
					{
						for (w = 0; w < W; w++)
						{
							dims[ndims_ - 2] = h; // tack on the lower dims to get the full dimensions
							dims[ndims_ - 1] = w;
							result(dims, ndims_) = (*this)(dims, ndims_, broadcast_required) + other(dims, ndims_, broadcast_required);
						}
					}
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);
					uint64_t len = H * W;

					for (i = 0; i < len; i++)
					{
						result_data[i] = lhs_data[i] + rhs_data[i];
					}
				}
			}
		}
		else
		{
			LTEN_ERR("MultiDimArrays must have more than 2 dimensions");
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray operator-(const MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		uint64_t dims[MAX_DIMS];
		uint64_t i;
		MultiDimArray result;
		uint64_t h, w;
		uint64_t H, W;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);

		if (ndims_ > 2)
		{
			if (broadcast_required)
			{
				md_array_dim_iterator it(dims_result, ndims_ - 2);

				H = dims_result[ndims_ - 2];
				W = dims_result[ndims_ - 1];

				for (auto higher_indices : it)
				{
					memcpy(dims, higher_indices, sizeof(uint64_t) * (ndims_ - 2));
					for (h = 0; h < H; h++)
					{
						for (w = 0; w < W; w++)
						{
							dims[ndims_ - 2] = h; // tack on the lower dims to get the full dimensions
							dims[ndims_ - 1] = w;
							result(dims, ndims_) = (*this)(dims, ndims_, broadcast_required) - other(dims, ndims_, broadcast_required);
						}
					}
				}
			}
			else
			{
				Dtype* result_data = result.GetDataPtr();
				Dtype* lhs_data = GetDataPtr();
				Dtype* rhs_data = other.GetDataPtr();
				uint64_t len = numels_;

				for (i = 0; i < len; i++)
				{
					result_data[i] = lhs_data[i] - rhs_data[i];
				}
			}

		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}


		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray operator*(float scalar) const
	{
		MultiDimArray result;
		uint64_t i;
		Dtype* data;

		result.Allocate(dims_array_, ndims_, nullptr, false);

		data = result.GetDataPtr();

		for (i = 0; i < numels_; i++)
		{
			data[i] = data_ptr_[i] * scalar;
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	friend MultiDimArray operator*(float scalar, const MultiDimArray& rhs)
	{
		return rhs * scalar;
	}


	MultiDimArray operator*(const MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		uint64_t dims[MAX_DIMS];
		uint64_t i;
		MultiDimArray result;
		uint64_t h, w;
		uint64_t H, W;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			H = dims_result[ndims_ - 2];
			W = dims_result[ndims_ - 1];

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					memcpy(dims, higher_indices, sizeof(uint64_t) * (ndims_ - 2));
					for (h = 0; h < H; h++)
					{
						for (w = 0; w < W; w++)
						{
							dims[ndims_ - 2] = h; // tack on the lower dims to get the full dimensions
							dims[ndims_ - 1] = w;
							result(dims, ndims_) = (*this)(dims, ndims_, broadcast_required) * other(dims, ndims_, broadcast_required);
						}
					}
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);
					uint64_t len = H * W;

					for (i = 0; i < len; i++)
					{
						result_data[i] = lhs_data[i] * rhs_data[i];
					}
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray operator/(const MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		uint64_t dims[MAX_DIMS];
		uint64_t i;
		MultiDimArray result;
		uint64_t h, w;
		uint64_t H, W;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();

		broadcast_required = check_broadcast_required(other_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			H = dims_result[ndims_ - 2];
			W = dims_result[ndims_ - 1];

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					memcpy(dims, higher_indices, sizeof(uint64_t) * (ndims_ - 2));
					for (h = 0; h < H; h++)
					{
						for (w = 0; w < W; w++)
						{
							dims[ndims_ - 2] = h; // tack on the lower dims to get the full dimensions
							dims[ndims_ - 1] = w;
							result(dims, ndims_) = (*this)(dims, ndims_, broadcast_required) / other(dims, ndims_, broadcast_required);
						}
					}
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2);
					uint64_t len = H * W;

					for (i = 0; i < len; i++)
					{
						result_data[i] = lhs_data[i] / rhs_data[i];
					}
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray masked_fill(const MultiDimArray& mask, Dtype value = 0)
	{
		const uint64_t* mask_dims_array;
		uint64_t dims_result[MAX_DIMS];
		uint64_t dims[MAX_DIMS];
		uint64_t i;
		MultiDimArray result;
		uint64_t h, w;
		uint64_t H, W;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != mask.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			mask.Reshape(3);
		}

		mask_dims_array = mask.GetSizes();

		broadcast_required = check_broadcast_required(mask_dims_array, dims_result);

		result.Allocate(dims_result, ndims_, nullptr, false);


		if (ndims_ > 2)
		{
			md_array_dim_iterator it(dims_result, ndims_ - 2);

			H = dims_result[ndims_ - 2];
			W = dims_result[ndims_ - 1];

			for (auto higher_indices : it)
			{
				if (broadcast_required)
				{
					memcpy(dims, higher_indices, sizeof(uint64_t) * (ndims_ - 2));
					for (h = 0; h < H; h++)
					{
						for (w = 0; w < W; w++)
						{
							dims[ndims_ - 2] = h; // tack on the lower dims to get the full dimensions
							dims[ndims_ - 1] = w;
							if (mask(dims, ndims_, broadcast_required) == 0)
							{
								result(dims, ndims_) = value;
							}
							else
							{
								result(dims, ndims_) = (*this)(dims, ndims_, broadcast_required);
							}

						}
					}
				}
				else
				{
					Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2);
					Dtype* rhs_data = mask.GetDataPtr(higher_indices, ndims_ - 2);
					uint64_t len = H * W;

					for (i = 0; i < len; i++)
					{
						if (rhs_data[i] == 0)
						{
							result_data[i] = value;
						}
						else
						{
							result_data[i] = lhs_data[i];
						}
					}
				}
			}
		}
		else
		{
			assert(0);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			mask.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray matmul(const MultiDimArray& other)
	{
		const uint64_t* other_dims_array;
		uint64_t dims_result[MAX_DIMS];
		MultiDimArray result;
		uint64_t M;
		uint64_t N;
		uint64_t K;
		int original_ndims = 0;
		bool broadcast_required;

		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		if (ndims_ < 3)
		{
			original_ndims = ndims_;
			Reshape(3);
			other.Reshape(3);
		}

		other_dims_array = other.GetSizes();
		if (dims_array_[ndims_ - 1] != other_dims_array[ndims_ - 2]) // check matrix dimension compatibility
		{
			LTEN_ERR("MultiDimArrays must have compatiple dimensions");
		}


		broadcast_required = check_broadcast_required(other_dims_array, dims_result, true);

		result.Allocate(dims_result, ndims_, nullptr, false);


		md_array_dim_iterator it(dims_result, ndims_ - 2);

		M = dims_result[ndims_ - 2];
		N = dims_result[ndims_ - 1];
		K = dims_array_[ndims_ - 1];

		for (auto higher_indices : it)
		{
			Dtype* result_data = result.GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);
			Dtype* lhs_data = GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);
			Dtype* rhs_data = other.GetDataPtr(higher_indices, ndims_ - 2, broadcast_required);

			cpu_gemm(false, false, M, N, K, static_cast<Dtype>(1), lhs_data, rhs_data, static_cast<Dtype>(0), result_data);
		}

		if (original_ndims)
		{
			Reshape(original_ndims);
			other.Reshape(original_ndims);
			result.Reshape(original_ndims);
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray cat(const MultiDimArray& other, int dim)
	{
		MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		int i;
		const uint64_t* other_dims_array;
		const uint64_t* other_strides_array;
		const uint64_t* result_strides_array;
		Dtype* other_data_ptr;
		uint64_t other_numels;
		uint64_t index;
		uint64_t cat_index;
		uint64_t dim_offset;
		Dtype* data;


		if (ndims_ != other.GetNDims())
		{
			LTEN_ERR("MultiDimArrays must have the same number of dimensions");
		}

		other_dims_array = other.GetSizes();
		for (i = 0; i < ndims_; i++)
		{
			if (i != dim)
			{
				if (dims_array_[i] != other_dims_array[i])
				{
					LTEN_ERR("MultiDimArrays must have compatiple dimensions");
				}
				dims_result[i] = dims_array_[i];
			}
			else
			{
				dims_result[i] = dims_array_[i] + other_dims_array[i];
			}
		}

		result.Allocate(dims_result, ndims_, nullptr, false);

		result_strides_array = result.GetStrides();
		//---------------------------------------------------------------------------------------------
		// alternative solution 
		//---------------------------------------------------------------------------------------------
		/*
		uint64_t coordinates[MAX_DIMS];
		for (index = 0; index < numels_; index++)
		{
			CoordinatesFromIndex(index, dims_array_, strides_array_, coordinates, ndims_);
			result(coordinates, ndims_) = data_ptr_[index];
		}

		dim_offset = dims_array_[dim];
		other_numels = other.GetNumels();
		other_strides_array = other.GetStrides();
		other_data_ptr = other.GetDataPtr();

		for (index = 0; index < other_numels; index++)
		{
			CoordinatesFromIndex(index, other_dims_array, other_strides_array, coordinates, ndims_);
			coordinates[dim] += dim_offset;
			result(coordinates, ndims_) = other_data_ptr[index];
		}
		*/
		//---------------------------------------------------------------------------------------------

		data = result.GetDataPtr();
		for (index = 0; index < numels_; index++)
		{
			uint64_t quotient;
			uint64_t remainder;

			if (dim > 0)
			{
				quotient = index / strides_array_[dim - 1];
				remainder = index % strides_array_[dim - 1];
				cat_index = quotient * result_strides_array[dim - 1] + remainder;
			}
			else
			{
				cat_index = index;
			}

			data[cat_index] = data_ptr_[index];
		}

		uint64_t axis_coord;


		dim_offset = dims_array_[dim];
		other_numels = other.GetNumels();
		other_strides_array = other.GetStrides();
		other_data_ptr = other.GetDataPtr();


		for (index = 0; index < other_numels; index++)
		{
			uint64_t quotient;
			uint64_t remainder;

			if (dim > 0)
			{
				quotient = index / other_strides_array[dim - 1];
				remainder = index % other_strides_array[dim - 1];
				axis_coord = remainder / other_strides_array[dim]; // axis_dim is other's coordinate in dimension dim
				remainder = remainder % other_strides_array[dim];
				cat_index = quotient * result_strides_array[dim - 1] + (axis_coord + dim_offset) * result_strides_array[dim] + remainder;
			}
			else
			{
				cat_index = index + numels_;
			}

			data[cat_index] = other_data_ptr[index];
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	MultiDimArray transpose(int dim_1, int dim_2)
	{
		MultiDimArray result;
		uint64_t dims_result[MAX_DIMS];
		uint64_t temp;

		if (dim_1 == dim_2 || dim_1 >= ndims_ || dim_2 >= ndims_)
		{
			LTEN_ERR("Invalid args");
		}


		memcpy(dims_result, dims_array_, sizeof(uint64_t) * ndims_);
		temp = dims_result[dim_1];
		dims_result[dim_1] = dims_result[dim_2];
		dims_result[dim_2] = temp;

		result.Allocate(dims_result, ndims_, nullptr, false);

		md_array_dim_iterator it(dims_array_, ndims_);

		for (auto indices : it)
		{
			memcpy(dims_result, indices, sizeof(uint64_t) * ndims_);
			temp = dims_result[dim_1];
			dims_result[dim_1] = dims_result[dim_2];
			dims_result[dim_2] = temp;

			result(dims_result, ndims_, false) = (*this)(indices, ndims_, false);
		}


		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}


	MultiDimArray transpose()
	{
		MultiDimArray result;
		uint64_t dims_result[2];
		uint64_t y;
		uint64_t x;
		uint64_t rows;
		uint64_t cols;
		Dtype* ptr;

		if (ndims_ != 2)
		{
			LTEN_ERR("This function only works for 2 dimensional MultiDimArrays");
		}

		rows = dims_array_[0];
		cols = dims_array_[1];

		dims_result[0] = cols;
		dims_result[1] = rows;

		result.Allocate(dims_result, ndims_, nullptr, false);
		ptr = result.GetDataPtr();

		for (y = 0; y < rows; y++)
		{
			for (x = 0; x < cols; x++)
			{
				ptr[x * rows + y] = data_ptr_[y * cols + x];
			}
		}

		return MultiDimArray(result.GetSizes(), result.GetNDims(), result.GetDataPtr(), true);
	}

	bool check_broadcast_required(const uint64_t* other_dims_array, uint64_t* max_dims_array = nullptr, bool mat_mul_check = false) const
	{
		bool broadcast_required;
		int i;
		int ndims;

		broadcast_required = false;

		ndims = ndims_;

		if (mat_mul_check)
		{
			if (max_dims_array)
			{
				max_dims_array[ndims - 1] = other_dims_array[ndims - 1];
				max_dims_array[ndims - 2] = dims_array_[ndims - 2];
			}

			ndims -= 2; // W & H may not match for matrix multiplication
		}

		for (i = 0; i < ndims; i++)
		{
			if (dims_array_[i] != other_dims_array[i])
			{
				if (dims_array_[i] != 1 && other_dims_array[i] != 1)
				{
					LTEN_ERR("MultiDimArrays must have compatiple dimensions");
				}

				broadcast_required = true;
			}

			if (max_dims_array)
			{
				max_dims_array[i] = std::max(dims_array_[i], other_dims_array[i]);
			}
		}

		return broadcast_required;
	}

protected:
	uint64_t GetOffset(const uint64_t* coordinates, int ndims, bool broadcast = false) const
	{
		int i;
		uint64_t dims[MAX_DIMS];
		uint64_t offset;

		if (broadcast)
		{
			for (i = 0; i < ndims; i++)
			{
				dims[i] = std::min(coordinates[i], dims_array_[i] - 1);
			}

			offset = 0;
			for (i = 0; i < ndims; i++)
			{
				offset += (dims[i] * strides_array_[i]);
			}
		}
		else
		{
			offset = 0;
			for (i = 0; i < ndims; i++)
			{
				offset += (coordinates[i] * strides_array_[i]);
			}
		}


		return offset;
	}

	virtual void Reset()
	{
		own_memory_ = false;
		data_ptr_ = nullptr;
		ndims_ = 0;
	}

	uint64_t numels_;
	bool own_memory_;
	Dtype* data_ptr_;
	mutable int ndims_;
	mutable uint64_t dims_array_[MAX_DIMS];
	mutable uint64_t strides_array_[MAX_DIMS];
};


//-----------------------------------------------------------------------------------------------------
// '+' sign in front of elements is for uint8_t promotion or else uint8_t arrays do not print correctly
//-----------------------------------------------------------------------------------------------------
template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const MultiDimArray<Dtype>& md_array)
{
	uint64_t W, w;
	uint64_t H, h;
	uint64_t index;
	uint64_t numels;
	Dtype* elements;
	int ndims;

	index = 0;
	numels = md_array.GetNumels();
	ndims = md_array.GetNDims();
	W = md_array.GetSizes()[ndims - 1];

	if (ndims == 1)
	{
		for (w = 0; w < W; w++)
		{
			out << +md_array.GetDataPtr()[w];
			index++;
			if (index < numels)
			{
				out << ",";
			}
		}
		out << "\n";
	}
	else
	{
		H = md_array.GetSizes()[ndims - 2];
		if (ndims == 2)
		{
			for (h = 0; h < H; h++)
			{
				for (w = 0; w < W; w++)
				{
					out << +md_array.GetDataPtr()[h * W + w];
					index++;
					if (index < numels)
					{
						out << ",";
					}
				}
				out << "\n";
			}
			out << "\n";
		}
		else
		{
			ndims -= 2;
			md_array_dim_iterator it(md_array.GetSizes(), ndims);

			for (auto indices : it)
			{
				elements = md_array.GetDataPtr(indices, ndims);
				for (h = 0; h < H; h++)
				{
					for (w = 0; w < W; w++)
					{
						out << +elements[h * W + w];
						index++;
						if (index < numels)
						{
							out << ",";
						}
					}
					out << "\n";
				}
				out << "\n";
			}
		}
	}


	return out;
}


template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const MultiDimArray<Dtype>* md_array)
{
	return out << *md_array;
}


#endif //MD_ARRAY_H
