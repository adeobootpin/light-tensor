#include <immintrin.h>
#include <iostream>
#include <assert.h>
#include <cmath>
#include "lten.h"
#include "threadpool2.h"
#include "math_fns.h"

template<typename Dtype>
void cpu_axpy(uint64_t N, Dtype alpha, Dtype* X_ptr, Dtype* Y_ptr, Dtype* C_ptr)
{
	uint64_t i;

	if (alpha == static_cast<Dtype>(1))
	{
		for (i = 0; i < N; i++)
		{
			C_ptr[i] = Y_ptr[i] + X_ptr[i];
		}
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			C_ptr[i] = Y_ptr[i] + alpha * X_ptr[i];
		}
	}
}

template<typename Dtype>
void cpu_axpby(uint64_t N, Dtype alpha, Dtype* X_ptr, Dtype beta, Dtype* Y_ptr, Dtype* C_ptr) // c = alpha * x + beta * y
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		C_ptr[i] = alpha * X_ptr[i] + beta * Y_ptr[i];
	}

}


template<typename Dtype>
void cpu_mul(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr)
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		C_ptr[i] = A_ptr[i] * B_ptr[i];
	}
}


template<typename Dtype>
void cpu_mul(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr, Dtype beta, Dtype* C_ptr) // c = beta * c + alpha * a * b
{
	uint64_t i;

	if (beta == 0)
	{
		if (alpha == 1)
		{
			for (i = 0; i < N; i++)
			{
				C_ptr[i] = A_ptr[i] * B_ptr[i];
			}
		}
		else
		{
			for (i = 0; i < N; i++)
			{
				C_ptr[i] = alpha * A_ptr[i] * B_ptr[i];
			}
		}
	}
	else
	{
		if (alpha == 1)
		{
			for (i = 0; i < N; i++)
			{
				C_ptr[i] = C_ptr[i] * beta + A_ptr[i] * B_ptr[i];
			}
		}
		else
		{
			for (i = 0; i < N; i++)
			{
				C_ptr[i] = C_ptr[i] * beta + alpha * A_ptr[i] * B_ptr[i];
			}
		}
	}
}


template<typename Dtype>
void cpu_mul(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr) // b = alpha * a
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		B_ptr[i] = alpha * A_ptr[i];
	}
}

template<typename Dtype>
void cpu_add(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr) // b = alpha + a
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		B_ptr[i] = alpha + A_ptr[i];
	}
}



template<typename Dtype>
void cpu_div(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr)
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		C_ptr[i] = A_ptr[i] / B_ptr[i];
	}
}

template<typename Dtype>
void cpu_div(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr, Dtype beta, Dtype* C_ptr) // c = (beta * c) +  (alpha * a / b)
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		C_ptr[i] = beta * C_ptr[i] + alpha * A_ptr[i] / B_ptr[i];
	}
}

template<typename Dtype>
void cpu_sig(uint64_t N, const Dtype* A_ptr, Dtype* B_ptr) // b = sig(a)
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		B_ptr[i] = static_cast<Dtype>(1.0f / (1.0f + expf(static_cast<float>(-A_ptr[i]))));
	}
}


template<typename Dtype>
void cpu_tanh(uint64_t N, const Dtype* A_ptr, Dtype* B_ptr) // b = tanh(a)
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		B_ptr[i] = static_cast<Dtype>(tanhf(static_cast<float>(A_ptr[i])));
	}
}

template<typename Dtype>
void cpu_powx(uint64_t N, const Dtype* A_ptr, Dtype x, Dtype* B_ptr) // b = pow(a,x)
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		B_ptr[i] = static_cast<Dtype>(powf(static_cast<float>(A_ptr[i]), static_cast<float>(x)));
	}
}

template<typename Dtype>
void cpu_div_back(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr, Dtype* D_ptr) // special function for processing div_backward during backpropagation (calculates d += a * (-b) / (c * c))
{
	uint64_t i;

	for (i = 0; i < N; i++)
	{
		D_ptr[i] = D_ptr[i] + A_ptr[i] * (-B_ptr[i]) / (C_ptr[i] * C_ptr[i]);
	}
}


template<typename Dtype>
void cpu_copy(uint64_t N, Dtype* A_ptr, Dtype* B_ptr) // b = a
{
	memcpy(B_ptr, A_ptr, sizeof(Dtype) * N);
}


// The tensor function max(dim) generates a new tensor one dimension smaller than the original tensor by recoding the maximum values over dimension dim.
// For each location in the smaller tensor, cpu_max/gpu_max computes the correspoinding "first" location in the original larger tensor.
// In other words, given the offset of a coordinate a,b,d in the resualt tensor, and assuming dim=2, calculate the offset of a,b,0,d in the original tensor.
// If this offset is known, then from each location a,b,c we can scan along a,b,x,d from 0 to the size of dim to find the max.
// Once the max is found it is stored in location a,b,c
// The mapping between the destination location and the "first" source location is given by this formula:
// source_offset = (destination_offset - remainder) * ratio + remainder
// where:
// remainder = destination_offset % the stride of dimension dim
// ratio = stride of dim - 1 / stride of dim (which is the same as the size of dim) 
// Note: when dim = 0, then ratio = 1
template<typename Dtype>
void cpu_max(const Dtype* src, Dtype* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype max;
	Dtype val;
	uint64_t max_index;
	uint64_t rem;

	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		rem = offset_dst % stride;
		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

		max = src[offset_src];
		max_index = 0;
		for (i = 1; i < dim_size; i++)  // iterate through required dimension
		{
			offset_src += stride;
			val = src[offset_src];
			if (val >= max)
			{
				max = val;
				max_index = i;
			}
		}

		indices[offset_dst] = max_index;
		dst[offset_dst] = max;
	}
}

template<typename Dtype>
void cpu_max_backward(Dtype* dst, const Dtype* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t max_index;
	uint64_t rem;

	for (offset_src = 0; offset_src < numels; offset_src++)
	{
		rem = offset_src % stride;
		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

		max_index = indices[offset_src];
		offset_dst += stride * max_index;

		dst[offset_dst] += src[offset_src];
	}
}

// ND tensor sum
template<typename Dtype>
void cpu_sum(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype sum;
	uint64_t rem;

	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		rem = offset_dst % stride;
		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

		sum = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			sum += src[offset_src];
			offset_src += stride;
		}
		dst[offset_dst] = sum;
	}
}


// ND tensor sum backward
template<typename Dtype>
void cpu_sum_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t rem;
	Dtype val;
	uint64_t i;

	for (offset_src = 0; offset_src < numels; offset_src++)
	{
		rem = offset_src % stride;
		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

		val = src[offset_src];
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			dst[offset_dst] += val;
			offset_dst += stride;
		}
	}
}

// ND tensor mean
template<typename Dtype>
void cpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes)
{
	uint64_t offset_dst;
	uint32_t partial_offset_dst;
	uint64_t u64i;
	int i;
	uint64_t len;
	Dtype sum;
	int naxes;
	
	
	OffsetCalc_mean_var offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);
	naxes = ndims_src - ndims_dst;

	len = 1;
	for (i = 0; i < naxes; i++)
	{
		len *= dims_src[axes[i]];
	}


	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		sum = static_cast<Dtype>(0);

		partial_offset_dst = offs_calc.GetPartialSrcOffset_d(offset_dst);;
		
		for (u64i = 0; u64i < len; u64i++)
		{
			uint32_t offset_src = offs_calc.GetPartialSrcOffset_s(u64i);
			sum += src[offset_src + partial_offset_dst];

			//sum += src[offs_calc.GetSrcOffset(static_cast<uint32_t>(offset_dst), static_cast<uint32_t>(u64i))];
		}

		dst[offset_dst] = sum / static_cast<Dtype>(len);
	}


	/*
	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		sum = static_cast<Dtype>(0);

		for (u64i = 0; u64i < len; u64i++)
		{
			sum += src[offs_calc.GetSrcOffset(static_cast<uint32_t>(offset_dst), static_cast<uint32_t>(u64i))];
		}

		dst[offset_dst] = sum / static_cast<Dtype>(len);
	}
	*/

	/*
	OffsetCalc_mean_std_simple offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);

	naxes = ndims_src - ndims_dst;

	len = 1;
	for (i = 0; i < naxes; i++)
	{
		len *= dims_src[axes[i]];
	}

	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		offset_src = offs_calc.GetOffsets(offset_dst);

		sum = static_cast<Dtype>(0);

		for (u64i = 0; u64i < len; u64i++)
		{
			sum += src[offset_src + offs_calc.GetWorkspaceOffsets(u64i)];
		}

		dst[offset_dst] = sum / static_cast<Dtype>(len);
	}
	*/

	/*
	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		offset_src_save = offs_calc.GetOffsets(offset_dst);

		sum = static_cast<Dtype>(0);

		for (u64i = 0; u64i < len; u64i++)
		{
			offset_src = offset_src_save;

			coordinate = u64i;
			for (i = 0; i < naxes; i++)
			{
				coordinate = coordinate / non_axis_strides[i];

				offset_src += (coordinate * important_strides[i]);

				coordinate = u64i % non_axis_strides[i];
			}

			sum += src[offset_src];
		}

		dst[offset_dst] = sum / static_cast<Dtype>(len);
	}
	*/
}

// ND tensor mean
template<typename Dtype>
void cpu_mean(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype sum;
	uint64_t rem;

	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		rem = offset_dst % stride;
		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

		sum = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			sum += src[offset_src];
			offset_src += stride;
		}
		dst[offset_dst] = sum / static_cast<Dtype>(dim_size);
	}
}


// ND tensor mean backward
template<typename Dtype>
void cpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t rem;
	Dtype val;
	uint64_t i;

	for (offset_src = 0; offset_src < numels; offset_src++)
	{
		rem = offset_src % stride;
		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

		val = src[offset_src];
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			dst[offset_dst] += val / static_cast<Dtype>(dim_size);
			offset_dst += stride;
		}
	}
}


template<typename Dtype>
void cpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, Dtype scale)
{
	int i;
	uint32_t index;

	for (i = 0; i < numoutputs; i++)
	{
		index = offs->GetOffset(i);
		dst[i] = scale * src[index];
	}
}

// ND tensor variance
template<typename Dtype>
void cpu_var(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype temp;
	Dtype var;
	Dtype mean;
	uint64_t rem;

	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		rem = offset_dst % stride;
		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

		mean = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			mean += src[offset_src];
			offset_src += stride;
		}
		if (sample_mode)
		{
			mean /= static_cast<Dtype>(dim_size - 1);
		}
		else
		{
			mean /= static_cast<Dtype>(dim_size);
		}

		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer
		var = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension again
		{
			temp = src[offset_src] - mean;
			var += (temp * temp);
			offset_src += stride;
		}

		if (sample_mode)
		{
			var /= static_cast<Dtype>(dim_size - 1);
		}
		else
		{
			var /= static_cast<Dtype>(dim_size);
		}

		dst[offset_dst] = var;
	}
}

template<typename Dtype>
void cpu_var(Dtype* dst, const Dtype* src, const uint64_t num_outputs, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased)
{
	uint64_t i;
	uint64_t j;
	uint64_t loop_count;
	uint32_t partial_offset;
	uint32_t src_offset;
	uint32_t num_inputs;
	int naxes;
	Dtype var;

	Dtype mean;
	Dtype count;
	Dtype mean_b;
	Dtype count_b;
	Dtype Ms_b;
	Dtype n_ab;
	Dtype temp;
	Dtype delta;
	Dtype Ms;

	OffsetCalc_mean_var offs_calc(strides_dst, strides_src, ndims_dst, ndims_src, dims_src, axes);
	naxes = ndims_src - ndims_dst;
	
	num_inputs = 1;
	for (i = 0; i < ndims_src; i++)
	{
		num_inputs *= dims_src[i];
	}

	loop_count = num_inputs / num_outputs;

	for (i = 0; i < num_outputs; i++)
	{
		partial_offset = offs_calc.GetPartialSrcOffset_d(i);
		src_offset = offs_calc.GetPartialSrcOffset_s(0);
		
		mean = src[partial_offset + src_offset];
		Ms = 0;
		count = 1.0f;

		for (j = 1; j < loop_count; j++)
		{
			src_offset = offs_calc.GetPartialSrcOffset_s(j);

			count_b = 1;
			mean_b = src[partial_offset + src_offset];
			Ms_b = 0;

			n_ab = count + count_b;
			temp = 1.0f / n_ab;

			delta = mean_b - mean;
			mean = ((count * mean) + (count_b * mean_b)) * temp;
			Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
			count = n_ab;
		}

		if (unbiased)
		{
			dst[i] = Ms / (loop_count - 1);
		}
		else
		{
			dst[i] = Ms / loop_count;
		}
		
	}

	/*
	// simple (non-Welford version)
	for (i = 0; i < num_outputs; i++)
	{
		float mean = 0;
		partial_offset = offs_calc.GetPartialSrcOffset_d(i);
		for (j = 0; j < loop_count; j++)
		{
			src_offset = offs_calc.GetPartialSrcOffset_s(j);
			mean += src[partial_offset + src_offset];
		}
		mean = mean / (loop_count - 1);

		var = 0;
		for (j = 0; j < loop_count; j++)
		{
			src_offset = offs_calc.GetPartialSrcOffset_s(j);
			var += pow(src[partial_offset + src_offset] - mean, 2.0f);
		}

		if (unbiased)
		{
			var = var / (loop_count - 1);
		}
		else
		{
			var = var / loop_count;
		}
		

		dst[i] = var;
	}
	*/

}

// ND tensor variance backward
template<typename Dtype>
void cpu_var_backward(Dtype* dst, const Dtype* src, const Dtype* op1, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t rem;
	Dtype val;
	Dtype mean;
	uint64_t i;



	for (offset_src = 0; offset_src < numels; offset_src++)
	{
		rem = offset_src % stride;
		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

		mean = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			mean += op1[offset_dst];
			offset_dst += stride;
		}
		mean = mean / static_cast<Dtype>(dim_size);



		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			val = (static_cast<Dtype>(2) * (op1[offset_dst] - mean)) / static_cast<Dtype>(dim_size - 1 );
			dst[offset_dst] += val * src[offset_src];
			offset_dst += stride;
		}

	}

}


// ND tensor variance
template<typename Dtype>
void cpu_std(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode)
{
	uint64_t i;
	uint64_t offset_src;
	uint64_t offset_dst;
	Dtype temp;
	Dtype var;
	Dtype mean;
	uint64_t rem;

	for (offset_dst = 0; offset_dst < numels; offset_dst++)
	{
		rem = offset_dst % stride;
		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer

		mean = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			mean += src[offset_src];
			offset_src += stride;
		}
		
		if (sample_mode)
		{
			mean = mean / static_cast<Dtype>(dim_size - 1);
		}
		else
		{
			mean = mean / static_cast<Dtype>(dim_size);
		}


		offset_src = (offset_dst - rem) * ratio + rem; // calculate corresponding offset in src buffer given an offset in the destination buffer
		var = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension again
		{
			temp = src[offset_src] - mean;
			var += (temp * temp);
			offset_src += stride;
		}

		if (sample_mode)
		{
			var /= static_cast<Dtype>(dim_size - 1);
		}
		else
		{
			var /= static_cast<Dtype>(dim_size);
		}

		//dst[offset_dst] = static_cast<Dtype>(powf(static_cast<float>(var + 1e-5), 0.5));
		dst[offset_dst] = static_cast<Dtype>(powf(static_cast<float>(var), 0.5));
	}
}


// ND tensor mean backward
template<typename Dtype>
void cpu_std_backward(Dtype* dst, const Dtype* src, const Dtype* op1, const Dtype* std, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode)
{
	uint64_t offset_src;
	uint64_t offset_dst;
	uint64_t rem;
	Dtype val;
	Dtype mean;
	uint64_t i;



	for (offset_src = 0; offset_src < numels; offset_src++)
	{
		rem = offset_src % stride;
		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer

		mean = static_cast<Dtype>(0);
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			mean += op1[offset_dst];
			offset_dst += stride;
		}
		mean = mean / static_cast<Dtype>(dim_size);



		offset_dst = (offset_src - rem) * ratio + rem; // calculate corresponding offset in dst buffer given an offset in the source buffer
		for (i = 0; i < dim_size; i++)  // iterate through required dimension
		{
			if (sample_mode)
			{
				val = (static_cast<Dtype>(2) * (op1[offset_dst] - mean)) / static_cast<Dtype>(dim_size - 1); // d_var/dxi
			}
			else
			{
				val = (static_cast<Dtype>(2) * (op1[offset_dst] - mean)) / static_cast<Dtype>(dim_size); // d_var/dxi
			}
			val = (val * static_cast<Dtype>(0.5)) / std[offset_src];
			dst[offset_dst] += val * src[offset_src];
			offset_dst += stride;
		}

	}

}

template<typename Dtype>
void cpu_layer_norm(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd)
{
	uint32_t i;
	uint32_t j;
	uint32_t num_workers;
	uint32_t loop_count;
	uint32_t ln_dim;
	uint32_t src_offset;

	Dtype mean;
	Dtype count;
	Dtype mean_b;
	Dtype count_b;
	Dtype Ms_b;
	Dtype n_ab;
	Dtype temp;
	Dtype delta;
	Dtype Ms;
	Dtype std;
	Dtype inv_std;
	Dtype scale;
	float epsilon = 1e-5;

	ln_dim = dims_src[ndims_src - 1];
	
	num_workers = numels / ln_dim;

	scale = 1.0f / ln_dim;

	for (i = 0; i < num_workers; i++)
	{
		src_offset = i * ln_dim;

		mean = src[src_offset];
		Ms = 0;
		count = 1.0f;

		for (j = 1; j < ln_dim; j++) // traverse ln axis and compute var using Welford's online algorithm
		{
			count_b = 1;
			mean_b = src[src_offset + j];
			Ms_b = 0;

			n_ab = count + count_b;
			temp = 1.0f / n_ab;

			delta = mean_b - mean;
			mean = ((count * mean) + (count_b * mean_b)) * temp;
			Ms = Ms + Ms_b + (delta * delta) * (count * count_b) * temp;
			count = n_ab;
		}

		std = sqrt(Ms * scale + epsilon);
		inv_std = 1.0f / std;
		
		if (sd)
		{
			sd[i] = std; // save this for backprop
		}
		

		for (j = 0; j < ln_dim; j++) // traverse ln axis and compute layer norm
		{
			temp = src[src_offset + j];
			temp = (temp - mean) * inv_std;

			if (ln)
			{
				ln[src_offset + j] = temp; // save this for backprop
			}

			if (weight)
			{
				temp = temp * weight[j] + bias[j];
			}

			dst[src_offset + j] = temp; // offset_src == offset_dst
		}
	}
}

template<typename Dtype>
void cpu_layer_norm_backwards(void* vlayer_norm, Dtype* x, Dtype* top_gradient, Dtype* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, Dtype* feeder_gradient)
{
	lten::LayerNorm* layer_norm;
	Dtype* ln;
	Dtype* sd;
	Dtype* wt;
	Dtype* wt_grad;
	Dtype* bias_grad;
	uint32_t i;
	uint32_t j;
	uint64_t numels;
	uint64_t loop_count;

	layer_norm = (lten::LayerNorm*)vlayer_norm;

	numels = 1;
	for (i = 0; i < ndims; i++)
	{
		numels *= dst_dims[i];
	}

	ln = layer_norm->get_ln()->get_mdarray<Dtype>()->GetDataPtr();
	sd = layer_norm->get_sd()->get_mdarray<Dtype>()->GetDataPtr();

	if (layer_norm->is_affine())
	{
		uint32_t len;
		float w_grd;
		float b_grd;
		float top_grd;
		float wts;
		uint32_t ln_dim;

		wt = layer_norm->get_weights()->get_mdarray<Dtype>()->GetDataPtr();
		wt_grad = layer_norm->get_weights()->get_gradients_mdarray<Dtype>()->GetDataPtr();
		bias_grad = layer_norm->get_bias()->get_gradients_mdarray<Dtype>()->GetDataPtr();

		ln_dim = dst_dims[ndims - 1];
		len = numels / ln_dim;

		for (i = 0; i < ln_dim; i++)
		{
			wts = wt[i];
			w_grd = 0;
			b_grd = 0;

			for (j = 0; j < len; j++)
			{
				top_grd = top_gradient[j * ln_dim + i];
				w_grd += top_grd * ln[j * ln_dim + i];
				b_grd += top_grd;

				feeder_gradient[j * ln_dim + i] = top_grd * wts;
			}

			wt_grad[i] = w_grd;
			bias_grad[i] = b_grd;	
			
		}
	}
	else
	{
		feeder_gradient = top_gradient;
	}

	loop_count = dst_dims[ndims - 1];

	uint32_t offset;
	uint32_t index;
	uint32_t num_workers;
	float m_g;
	float f_g;
	float tmp;
	float invsd;
	float l_n;
	uint32_t ln_dim;
	float inv_dim;


	num_workers = numels / loop_count;
	ln_dim = dst_dims[ndims - 1];
	inv_dim = 1.0f / dst_dims[ndims - 1];

	index = 0;
	for (i = 0; i < num_workers; i++)
	{
		m_g = 0;
		tmp = 0;
		invsd = 1.0f / sd[i];

		for (j = 0; j < loop_count; j++)
		{
			f_g = feeder_gradient[i * loop_count + j];
			l_n = ln[i * loop_count + j];
			m_g -= f_g * invsd;
			tmp -= f_g * l_n;
		}

		for (j = 0; j < loop_count; j++)
		{
			f_g = feeder_gradient[i * loop_count + j];
			l_n = ln[i * loop_count + j];
			bottom_gradient[i * loop_count + j] = (f_g + inv_dim * l_n * tmp) *  invsd + inv_dim * m_g;
		}

	}

}

template<typename Dtype>
void cpu_sig_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels)
{
	uint64_t i;
	Dtype val;

	for (i = 0; i < numels; i++)
	{
		val = middle[i];
		bottom[i] = top[i] * val * (static_cast<Dtype>(1) - val);
	}

}

template<typename Dtype>
void cpu_tanh_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels)
{
	uint64_t i;
	Dtype val;

	for (i = 0; i < numels; i++)
	{
		val = middle[i];
		bottom[i] = top[i] * (static_cast<Dtype>(1) - val * val);
	}
}



template<typename Dtype>
void cpu_dropout(Dtype* dst, Dtype* src, unsigned int* mask, unsigned int threshold, Dtype scale, uint64_t len)
{
	uint64_t i;

	for (i = 0; i < len; i++)
	{
		dst[i] = src[i] * (mask[i] > threshold) * scale;
	}
}


void cpu_transpose(float* src, float* dst, int dim_1, int dim_2,
	int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1,
	int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels)
{
	unsigned int pre;
	unsigned int mid;
	unsigned int post;
	unsigned int remaining;

	unsigned int  idx;
	unsigned int  idx_transpose;

	int coord1;
	int coord2;
	float temp;

	if (dim_1 >= dim_2)
	{
		LTEN_ERR("Second dimension must be strictly greater than first dimension");
	}

	for (idx = 0; idx < numels; idx++)
	{

		if (dim_1 == 0)
		{
			remaining = idx;
		}
		else
		{
			remaining = idx % stride_src_dim_1_minus_1;
		}
		coord1 = remaining / stride_src_dim_1;

		remaining = idx % stride_src_dim_2_minus_1;
		coord2 = remaining / stride_src_dim_2;


		if (dim_1 == 0)
		{
			pre = 0;
		}
		else
		{
			pre = idx / stride_src_dim_1_minus_1;
		}
		remaining = idx % stride_src_dim_1;
		mid = remaining / stride_src_dim_2_minus_1;

		remaining = idx % stride_src_dim_2;
		post = remaining;

		if (dim_1 == 0)
		{
			idx_transpose = mid * stride_trn_dim_2_minus_1 + post;
		}
		else
		{
			idx_transpose = pre * stride_trn_dim_1_minus_1 + mid * stride_trn_dim_2_minus_1 + post;
		}

		idx_transpose += coord2 * stride_trn_dim_1 + coord1 * stride_trn_dim_2;

		temp = src[idx];
		src[idx] = dst[idx_transpose];
		dst[idx_transpose] = temp;
	}
}





template void cpu_axpy<float>(uint64_t N, float alpha, float* X_ptr, float* Y_ptr, float* C_ptr);
template void cpu_axpby<float>(uint64_t N, float alpha, float* X_ptr, float beta, float* Y_ptr, float* C_ptr);
template void cpu_mul<float>(uint64_t N, float* A_ptr, float* B_ptr, float* C_ptr);
template void cpu_mul<float>(uint64_t N, float alpha, float* A_ptr, float* B_ptr, float beta, float* C_ptr);
template void cpu_mul<float>(uint64_t N, float alpha, float* A_ptr, float* B_ptr);
template void cpu_div<float>(uint64_t N, float* A_ptr, float* B_ptr, float* C_ptr);
template void cpu_div<float>(uint64_t N, float alpha, float* A_ptr, float* B_ptr, float beta, float* C_ptr);
template void cpu_div_back<float>(uint64_t N, float* A_ptr, float* B_ptr, float* C_ptr, float* D_ptr);
template void cpu_add<float>(uint64_t N, float alpha, float* A_ptr, float* B_ptr);
template void cpu_copy<float>(uint64_t N, float* A_ptr, float* B_ptr);
template void cpu_max<float>(const float* src, float* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_max_backward<float>(float* dst, const float* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_sum<float>(const float* src, float* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_sum_backward<float>(float* dst, const float* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean<float>(const float* src, float* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean_backward<float>(float* dst, const float* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean_backward<float>(float* dst, const float* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, float scale);
template void cpu_var<float>(const float* src, float* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_var_backward<float>(float* dst, const float* src, const float* op1, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_std<float>(const float* src, float* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_std_backward<float>(float* dst, const float* src, const float* op1, const float* std, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_dropout<float>(float* dst, float* src, unsigned int* mask, unsigned int threshold, float scale, uint64_t len);
template void cpu_sig<float>(uint64_t N, const float* A_ptr, float* B_ptr);
template void cpu_tanh<float>(uint64_t N, const float* A_ptr, float* B_ptr);
template void cpu_powx<float>(uint64_t N, const float* A_ptr, float x, float* B_ptr);
template void cpu_tanh_backward<float>(float* bottom, const float* top, const float* middle, const uint64_t numels);
template void cpu_sig_backward<float>(float* bottom, const float* top, const float* middle, const uint64_t numels);

template void cpu_axpy<int>(uint64_t N, int alpha, int* X_ptr, int* Y_ptr, int* C_ptr);
template void cpu_axpby<int>(uint64_t N, int alpha, int* X_ptr, int beta, int* Y_ptr, int* C_ptr);
template void cpu_mul<int>(uint64_t N, int* A_ptr, int* B_ptr, int* C_ptr);
template void cpu_mul<int>(uint64_t N, int alpha, int* A_ptr, int* B_ptr, int beta, int* C_ptr);
template void cpu_mul<int>(uint64_t N, int alpha, int* A_ptr, int* B_ptr);
template void cpu_div<int>(uint64_t N, int* A_ptr, int* B_ptr, int* C_ptr);
template void cpu_div<int>(uint64_t N, int alpha, int* A_ptr, int* B_ptr, int beta, int* C_ptr);
template void cpu_div_back<int>(uint64_t N, int* A_ptr, int* B_ptr, int* C_ptr, int* D_ptr);
template void cpu_add<int>(uint64_t N, int alpha, int* A_ptr, int* B_ptr);
template void cpu_copy<int>(uint64_t N, int* A_ptr, int* B_ptr);
template void cpu_max<int>(const int* src, int* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_max_backward<int>(int* dst, const int* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_sum<int>(const int* src, int* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_sum_backward<int>(int* dst, const int* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean<int>(const int* src, int* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean_backward<int>(int* dst, const int* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean_backward<int>(int* dst, const int* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, int scale);
template void cpu_var<int>(const int* src, int* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_var_backward<int>(int* dst, const int* src, const int* op1, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_std<int>(const int* src, int* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_std_backward<int>(int* dst, const int* src, const int* op1, const int* std, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_dropout<int>(int* dst, int* src, unsigned int* mask, unsigned int threshold, int scale, uint64_t len);
template void cpu_sig<int>(uint64_t N, const int* A_ptr, int* B_ptr);
template void cpu_tanh<int>(uint64_t N, const int* A_ptr, int* B_ptr);
template void cpu_powx<int>(uint64_t N, const int* A_ptr, int x, int* B_ptr);
template void cpu_tanh_backward<int>(int* bottom, const int* top, const int* middle, const uint64_t numels);
template void cpu_sig_backward<int>(int* bottom, const int* top, const int* middle, const uint64_t numels);

template void cpu_axpy<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* X_ptr, uint8_t* Y_ptr, uint8_t* C_ptr);
template void cpu_axpby<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* X_ptr, uint8_t beta, uint8_t* Y_ptr, uint8_t* C_ptr);
template void cpu_mul<uint8_t>(uint64_t N, uint8_t* A_ptr, uint8_t* B_ptr, uint8_t* C_ptr);
template void cpu_mul<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A_ptr, uint8_t* B_ptr, uint8_t beta, uint8_t* C_ptr);
template void cpu_mul<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A_ptr, uint8_t* B_ptr);
template void cpu_div<uint8_t>(uint64_t N, uint8_t* A_ptr, uint8_t* B_ptr, uint8_t* C_ptr);
template void cpu_div<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A_ptr, uint8_t* B_ptr, uint8_t beta, uint8_t* C_ptr);
template void cpu_div_back<uint8_t>(uint64_t N, uint8_t* A_ptr, uint8_t* B_ptr, uint8_t* C_ptr, uint8_t* D_ptr);
template void cpu_add<uint8_t>(uint64_t N, uint8_t alpha, uint8_t* A_ptr, uint8_t* B_ptr);
template void cpu_copy<uint8_t>(uint64_t N, uint8_t* A_ptr, uint8_t* B_ptr);
template void cpu_max<uint8_t>(const uint8_t* src, uint8_t* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_max_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_sum<uint8_t>(const uint8_t* src, uint8_t* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_sum_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean<uint8_t>(const uint8_t* src, uint8_t* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_mean_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, uint8_t scale);
template void cpu_var<uint8_t>(const uint8_t* src, uint8_t* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_var_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint8_t* op1, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);
template void cpu_std<uint8_t>(const uint8_t* src, uint8_t* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_std_backward<uint8_t>(uint8_t* dst, const uint8_t* src, const uint8_t* op1, const uint8_t* std, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode);
template void cpu_dropout<uint8_t>(uint8_t* dst, uint8_t* src, unsigned int* mask, unsigned int threshold, uint8_t scale, uint64_t len);
template void cpu_sig<uint8_t>(uint64_t N, const uint8_t* A_ptr, uint8_t* B_ptr);
template void cpu_tanh<uint8_t>(uint64_t N, const uint8_t* A_ptr, uint8_t* B_ptr);
template void cpu_powx<uint8_t>(uint64_t N, const uint8_t* A_ptr, uint8_t x, uint8_t* B_ptr);
template void cpu_tanh_backward<uint8_t>(uint8_t* bottom, const uint8_t* top, const uint8_t* middle, const uint64_t numels);
template void cpu_sig_backward<uint8_t>(uint8_t* bottom, const uint8_t* top, const uint8_t* middle, const uint64_t numels);

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Native GEMM (first cut, requires optimization)
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#define BIG_M_BLOCK 4
#define BIG_K_BLOCK 128

#ifdef USE_THREADPOOL
template<typename Dtype>
struct MultiplyBlockParams
{
	bool transA;
	bool transB;
	Dtype* A;
	Dtype* B;
	Dtype* C;
	uint64_t M;
	uint64_t N;
	uint64_t K;
	uint64_t lda;
	uint64_t ldb;
	uint64_t ldc;
	int m_offset;
	int n_offset;
	int k_offset;
	Dtype alpha;
};
#endif




void generic_matmul(bool transA, bool transB, float* A, float* B, float* C, uint64_t M, uint64_t N, uint64_t K, uint64_t lda, uint64_t ldb, uint64_t ldc)
{
	uint64_t m;
	uint64_t n;
	uint64_t k;

	if (transA && transB)
	{
		for (n = 0; n < N; n++)
		{
			for (m = 0; m < M; m++)
			{
				for (k = 0; k < K; k++)
				{
					C[m * ldc + n] += A[k * lda + m] * B[n * ldb + k];
				}
			}
		}
	}
	else
	{
		if (transA)
		{
			for (n = 0; n < N; n++)
			{
				for (m = 0; m < M; m++)
				{
					for (k = 0; k < K; k++)
					{
						C[m * ldc + n] += A[k * lda + m] * B[k * ldb + n];
					}
				}
			}
		}
		else
		{
			if (transB)
			{
				for (n = 0; n < N; n++)
				{
					for (m = 0; m < M; m++)
					{
						for (k = 0; k < K; k++)
						{
							C[m * ldc + n] += A[m * lda + k] * B[n * ldb + k];
						}
					}
				}
			}
			else
			{
				for (n = 0; n < N; n++)
				{
					for (m = 0; m < M; m++)
					{
						for (k = 0; k < K; k++)
						{
							C[m * ldc + n] += A[m * lda + k] * B[k * ldb + n];
						}
					}
				}
			}
		}
	}

}

#ifdef USE_AVX_256
void MultiplyElementsAVX(bool transA, bool transB, float* A, float* B, float* C, uint64_t K, uint64_t lda, uint64_t ldb, uint64_t ldc, uint64_t m_offset, uint64_t n_offset, uint64_t k_offset)
{
	__m256 a_0;
	__m256 a_1;
	__m256 a_2;
	__m256 a_3;

	__m256 b_0_7;
	__m256 b_8_15;

	__m256 c_0_7_0;
	__m256 c_0_7_1;
	__m256 c_0_7_2;
	__m256 c_0_7_3;

	__m256 c_8_15_0;
	__m256 c_8_15_1;
	__m256 c_8_15_2;
	__m256 c_8_15_3;


	int64_t k;
	int64_t k_stop;

	c_0_7_0 = _mm256_setzero_ps();
	c_0_7_1 = _mm256_setzero_ps();
	c_0_7_2 = _mm256_setzero_ps();
	c_0_7_3 = _mm256_setzero_ps();

	c_8_15_0 = _mm256_setzero_ps();
	c_8_15_1 = _mm256_setzero_ps();
	c_8_15_2 = _mm256_setzero_ps();
	c_8_15_3 = _mm256_setzero_ps();

	k_stop = K + k_offset;


	if (transA && transB)
	{
		for (k = k_offset; k < k_stop; k++)
		{
			//------------------------------------------------------
			// transA == true transB == true
			a_0 = _mm256_broadcast_ss(A + k * lda + m_offset + 0);
			a_1 = _mm256_broadcast_ss(A + k * lda + m_offset + 1);
			a_2 = _mm256_broadcast_ss(A + k * lda + m_offset + 2);
			a_3 = _mm256_broadcast_ss(A + k * lda + m_offset + 3);

			float* b = &B[n_offset * ldb];
			b_0_7 = _mm256_set_ps(b[k + 7 * ldb], b[k + 6 * ldb], b[k + 5 * ldb], b[k + 4 * ldb], b[k + 3 * ldb], b[k + 2 * ldb], b[k + 1 * ldb], b[k]);
			b_8_15 = _mm256_set_ps(b[k + 15 * ldb], b[k + 14 * ldb], b[k + 13 * ldb], b[k + 12 * ldb], b[k + 11 * ldb], b[k + 10 * ldb], b[k + 9 * ldb], b[k + 8 * ldb]);
			//------------------------------------------------------

			c_0_7_0 = _mm256_add_ps(c_0_7_0, _mm256_mul_ps(a_0, b_0_7));
			c_0_7_1 = _mm256_add_ps(c_0_7_1, _mm256_mul_ps(a_1, b_0_7));
			c_0_7_2 = _mm256_add_ps(c_0_7_2, _mm256_mul_ps(a_2, b_0_7));
			c_0_7_3 = _mm256_add_ps(c_0_7_3, _mm256_mul_ps(a_3, b_0_7));

			c_8_15_0 = _mm256_add_ps(c_8_15_0, _mm256_mul_ps(a_0, b_8_15));
			c_8_15_1 = _mm256_add_ps(c_8_15_1, _mm256_mul_ps(a_1, b_8_15));
			c_8_15_2 = _mm256_add_ps(c_8_15_2, _mm256_mul_ps(a_2, b_8_15));
			c_8_15_3 = _mm256_add_ps(c_8_15_3, _mm256_mul_ps(a_3, b_8_15));
		}
	}
	else
	{
		if (transA)
		{
			for (k = k_offset; k < k_stop; k++)
			{
				//------------------------------------------------------
				// transA == true transB == false
				a_0 = _mm256_broadcast_ss(A + k * lda + m_offset + 0);
				a_1 = _mm256_broadcast_ss(A + k * lda + m_offset + 1);
				a_2 = _mm256_broadcast_ss(A + k * lda + m_offset + 2);
				a_3 = _mm256_broadcast_ss(A + k * lda + m_offset + 3);

				b_0_7 = _mm256_loadu_ps(B + k * ldb + n_offset);
				b_8_15 = _mm256_loadu_ps(B + k * ldb + n_offset + 8);
				//------------------------------------------------------

				c_0_7_0 = _mm256_add_ps(c_0_7_0, _mm256_mul_ps(a_0, b_0_7));
				c_0_7_1 = _mm256_add_ps(c_0_7_1, _mm256_mul_ps(a_1, b_0_7));
				c_0_7_2 = _mm256_add_ps(c_0_7_2, _mm256_mul_ps(a_2, b_0_7));
				c_0_7_3 = _mm256_add_ps(c_0_7_3, _mm256_mul_ps(a_3, b_0_7));

				c_8_15_0 = _mm256_add_ps(c_8_15_0, _mm256_mul_ps(a_0, b_8_15));
				c_8_15_1 = _mm256_add_ps(c_8_15_1, _mm256_mul_ps(a_1, b_8_15));
				c_8_15_2 = _mm256_add_ps(c_8_15_2, _mm256_mul_ps(a_2, b_8_15));
				c_8_15_3 = _mm256_add_ps(c_8_15_3, _mm256_mul_ps(a_3, b_8_15));
			}
		}
		else
		{
			if (transB)
			{
				for (k = k_offset; k < k_stop; k++)
				{
					//------------------------------------------------------
					// transA == false transB == true
					a_0 = _mm256_broadcast_ss(A + (m_offset + 0) * lda + k);
					a_1 = _mm256_broadcast_ss(A + (m_offset + 1) * lda + k);
					a_2 = _mm256_broadcast_ss(A + (m_offset + 2) * lda + k);
					a_3 = _mm256_broadcast_ss(A + (m_offset + 3) * lda + k);

					float* b = &B[n_offset * ldb];
					b_0_7 = _mm256_set_ps(b[k + 7 * ldb], b[k + 6 * ldb], b[k + 5 * ldb], b[k + 4 * ldb], b[k + 3 * ldb], b[k + 2 * ldb], b[k + 1 * ldb], b[k]);
					b_8_15 = _mm256_set_ps(b[k + 15 * ldb], b[k + 14 * ldb], b[k + 13 * ldb], b[k + 12 * ldb], b[k + 11 * ldb], b[k + 10 * ldb], b[k + 9 * ldb], b[k + 8 * ldb]);
					//-------------------------------------------------------

					c_0_7_0 = _mm256_add_ps(c_0_7_0, _mm256_mul_ps(a_0, b_0_7));
					c_0_7_1 = _mm256_add_ps(c_0_7_1, _mm256_mul_ps(a_1, b_0_7));
					c_0_7_2 = _mm256_add_ps(c_0_7_2, _mm256_mul_ps(a_2, b_0_7));
					c_0_7_3 = _mm256_add_ps(c_0_7_3, _mm256_mul_ps(a_3, b_0_7));

					c_8_15_0 = _mm256_add_ps(c_8_15_0, _mm256_mul_ps(a_0, b_8_15));
					c_8_15_1 = _mm256_add_ps(c_8_15_1, _mm256_mul_ps(a_1, b_8_15));
					c_8_15_2 = _mm256_add_ps(c_8_15_2, _mm256_mul_ps(a_2, b_8_15));
					c_8_15_3 = _mm256_add_ps(c_8_15_3, _mm256_mul_ps(a_3, b_8_15));
				}
			}
			else
			{
				for (k = k_offset; k < k_stop; k++)
				{
					//------------------------------------------------------
					// transA == false transB == false
					a_0 = _mm256_broadcast_ss(A + (m_offset + 0) * lda + k);
					a_1 = _mm256_broadcast_ss(A + (m_offset + 1) * lda + k);
					a_2 = _mm256_broadcast_ss(A + (m_offset + 2) * lda + k);
					a_3 = _mm256_broadcast_ss(A + (m_offset + 3) * lda + k);

					b_0_7 = _mm256_loadu_ps(B + k * ldb + n_offset);
					b_8_15 = _mm256_loadu_ps(B + k * ldb + n_offset + 8);

					//------------------------------------------------------

					c_0_7_0 = _mm256_add_ps(c_0_7_0, _mm256_mul_ps(a_0, b_0_7));
					c_0_7_1 = _mm256_add_ps(c_0_7_1, _mm256_mul_ps(a_1, b_0_7));
					c_0_7_2 = _mm256_add_ps(c_0_7_2, _mm256_mul_ps(a_2, b_0_7));
					c_0_7_3 = _mm256_add_ps(c_0_7_3, _mm256_mul_ps(a_3, b_0_7));

					c_8_15_0 = _mm256_add_ps(c_8_15_0, _mm256_mul_ps(a_0, b_8_15));
					c_8_15_1 = _mm256_add_ps(c_8_15_1, _mm256_mul_ps(a_1, b_8_15));
					c_8_15_2 = _mm256_add_ps(c_8_15_2, _mm256_mul_ps(a_2, b_8_15));
					c_8_15_3 = _mm256_add_ps(c_8_15_3, _mm256_mul_ps(a_3, b_8_15));
				}

			}
		}
	}


	float* c = &C[m_offset * ldc + n_offset];

	_mm256_store_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), c_0_7_0));
	_mm256_store_ps(c + ldc, _mm256_add_ps(_mm256_loadu_ps(c + ldc), c_0_7_1));
	_mm256_store_ps(c + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc), c_0_7_2));
	_mm256_store_ps(c + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc), c_0_7_3));

	_mm256_store_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), c_8_15_0));
	_mm256_store_ps(c + ldc + 8, _mm256_add_ps(_mm256_loadu_ps(c + ldc + 8), c_8_15_1));
	_mm256_store_ps(c + 2 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc + 8), c_8_15_2));
	_mm256_store_ps(c + 3 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc + 8), c_8_15_3));

}

void simd_matmul(bool transA, bool transB, float* A, float* B, float* C, int64_t M, int64_t N, int64_t K, int64_t lda, uint64_t ldb, uint64_t ldc, uint64_t m_offset, uint64_t n_offset, uint64_t k_offset)
{
	int64_t n;
	int64_t m;

	int n_block = 16;
	int m_block = 4;

	for (n = 0; n < static_cast<int64_t>(N) - n_block + 1; n += n_block)
	{
		for (m = 0; m < static_cast<int64_t>(M) - m_block + 1; m += m_block)
		{
			MultiplyElementsAVX(transA, transB, A, B, C, K, lda, ldb, ldc, m_offset + m, n_offset + n, k_offset);
		}
	}

	m = (M / m_block) * m_block;
	n = (N / n_block) * n_block;


	if (transA && transB)
	{
		if (m < M)
		{
			generic_matmul(transA, transB, A + k_offset * lda + m_offset + m, B + k_offset, C + (m_offset + m) * ldc, M - m, n, K, lda, ldb, ldc);
		}

		if (n < N)
		{
			generic_matmul(transA, transB, A + k_offset * lda + m_offset, B + k_offset + n * ldb, C + m_offset * ldc + n, m, N - n, K, lda, ldb, ldc);
		}

		if (m < M && n < N)
		{
			generic_matmul(transA, transB, A + k_offset * lda + m_offset + m, B + k_offset + n * ldb, C + (m_offset + m) * ldc + n, M - m, N - n, K, lda, ldb, ldc);
		}
	}
	else
	{
		if (transA)
		{
			if (m < M)
			{
				generic_matmul(transA, transB, A + k_offset * lda + m_offset + m, B + k_offset * ldb, C + (m_offset + m) * ldc, M - m, n, K, lda, ldb, ldc);
			}

			if (n < N)
			{
				generic_matmul(transA, transB, A + k_offset * lda + m_offset, B + k_offset * ldb + n, C + m_offset * ldc + n, m, N - n, K, lda, ldb, ldc);
			}

			if (m < M && n < N)
			{
				generic_matmul(transA, transB, A + k_offset * lda + m_offset + m, B + k_offset * ldb + n, C + (m_offset + m) * ldc + n, M - m, N - n, K, lda, ldb, ldc);
			}
		}
		else
		{
			if (transB)
			{
				if (m < M)
				{
					generic_matmul(transA, transB, A + (m_offset + m) * lda + k_offset, B + k_offset, C + (m_offset + m) * ldc, M - m, n, K, lda, ldb, ldc);
				}

				if (n < N)
				{
					generic_matmul(transA, transB, A + m_offset * lda + k_offset, B + k_offset + n * ldb, C + m_offset * ldc + n, m, N - n, K, lda, ldb, ldc);
				}

				if (m < M && n < N)
				{
					generic_matmul(transA, transB, A + (m_offset + m) * lda + k_offset, B + k_offset + n * ldb, C + (m_offset + m) * ldc + n, M - m, N - n, K, lda, ldb, ldc);
				}
			}
			else
			{
				if (m < M)
				{
					generic_matmul(transA, transB, A + (m_offset + m) * lda + k_offset, B + k_offset * ldb, C + (m_offset + m) * ldc, M - m, n, K, lda, ldb, ldc);
				}

				if (n < N)
				{
					generic_matmul(transA, transB, A + m_offset * lda + k_offset, B + k_offset * ldb + n, C + m_offset * ldc + n, m, N - n, K, lda, ldb, ldc);
				}

				if (m < M && n < N)
				{
					generic_matmul(transA, transB, A + (m_offset + m) * lda + k_offset, B + k_offset * ldb + n, C + (m_offset + m) * ldc + n, M - m, N - n, K, lda, ldb, ldc);
				}
			}
		}
	}
}
#endif

#ifdef USE_THREADPOOL
int matmul_mp_threadproc(void* params, int block_index, int total_blocks)
{
	int k_block = BIG_K_BLOCK;
	int m_block = BIG_M_BLOCK;

	MultiplyBlockParams<float>* mbp = (MultiplyBlockParams<float>*)params;

	bool transA = mbp->transA;
	bool transB = mbp->transB;
	float* A = mbp->A;
	float* B = mbp->B;
	float* C = mbp->C;
	uint64_t M = mbp->M;
	uint64_t N = mbp->N;
	uint64_t K = mbp->K;
	uint64_t lda = mbp->lda;
	uint64_t ldb = mbp->ldb;
	uint64_t ldc = mbp->ldc;

	float alpha = mbp->alpha;

#ifdef USE_AVX_256
	for (int k = 0; k < K; k += k_block)
	{
		for (int m = block_index * m_block; m < M; m += (m_block * total_blocks))
		{
			simd_matmul(transA, transB, A, B, C, std::min(static_cast<int>(M) - m, m_block), N, std::min(static_cast<int>(K) - k, k_block), lda, ldb, ldc, m, 0, k);
		}
	}
#else
	for (int m = block_index * m_block; m < M; m += (m_block * total_blocks))
	{
		if (transA)
		{
			generic_matmul(transA, transB, A + m, B, C + m * ldc, std::min(static_cast<int>(M) - m, m_block), N, K, lda, ldb, ldc);
		}
		else
		{
			generic_matmul(transA, transB, A + m * lda, B, C + m * ldc, std::min(static_cast<int>(M) - m, m_block), N, K, lda, ldb, ldc);
		}
	}
#endif

	return 0;
}

void matmul_mp(bool transA, bool transB, int64_t M, int64_t N, int64_t K, float alpha, float* A, float* B, float beta, float* C)
{
	int64_t i;
	uint64_t lda;
	uint64_t ldb;
	uint64_t ldc;
	MultiplyBlockParams<float> mbp;
	ThreadPool* threadpool;
	int* ret;

	lda = (!transA) ? K : M;
	ldb = (!transB) ? N : K;
	ldc = N;

	int k_block = BIG_K_BLOCK;
	int m_block = BIG_M_BLOCK;


	for (i = 0; i < M * N; i++)
	{
		C[i] = C[i] * beta;
	}

	for (i = 0; i < M * K; i++)
	{
		A[i] = A[i] * alpha;
	}
	
	threadpool = lten::MISC_globals::singleton()->get_threadpool();

	mbp.transA = transA;
	mbp.transB = transB;
	mbp.A = A;
	mbp.B = B;
	mbp.C = C;
	mbp.M = M;
	mbp.N = N;
	mbp.K = K;
	mbp.lda = lda;
	mbp.ldb = ldb;
	mbp.ldc = ldc;
	mbp.alpha = alpha;

	threadpool->Execute(matmul_mp_threadproc, &mbp, threadpool->get_thread_count());
	threadpool->WaitForTaskCompletion(&ret);

	if (*ret)
	{
		char error_msg[100];
		snprintf(error_msg, sizeof(error_msg), "Multithreaded matmul returned an error [%d]", *ret);
		LTEN_ERR(error_msg);
	}
}
#endif

template<typename Dtype>
void matmul(bool transA, bool transB, int64_t M, int64_t N, int64_t K, Dtype alpha, Dtype* A, Dtype* B, Dtype beta, Dtype* C)
{
	int64_t i;
	uint64_t lda;
	uint64_t ldb;
	uint64_t ldc;


	lda = (!transA) ? K : M;
	ldb = (!transB) ? N : K;
	ldc = N;

	int k_block = BIG_K_BLOCK;
	int m_block = BIG_M_BLOCK;

	if (beta == 0)
	{
		for (i = 0; i < M * N; i++)
		{
			C[i] = 0;
		}
	}
	else
	{
		for (i = 0; i < M * N; i++)
		{
			C[i] = C[i] * beta;
		}
	}

	if (alpha != 1.0f)
	{
		for (i = 0; i < M * K; i++)
		{
			A[i] = A[i] * alpha;
		}
	}

#ifdef USE_AVX_256
	for (int k = 0; k < K; k += k_block)
	{
		for (int m = 0; m < M; m += m_block)
		{
			simd_matmul(transA, transB, A, B, C, std::min(static_cast<int>(M) - m, m_block), N, std::min(static_cast<int>(K) - k, k_block), lda, ldb, ldc, m, 0, k);
		}
	}
#else
	generic_matmul(transA, transB, A, B, C, M, N, K, lda, ldb, ldc);
#endif

}

template<>
void cpu_gemm<float>(bool transA, bool transB, uint64_t M, uint64_t N, uint64_t K, float alpha, float* A, float* B, float beta, float* C)
{
#ifdef USE_OPENBLAS
	CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
	CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
	int lda = static_cast<int>((TransA == CblasNoTrans) ? K : M);
	int ldb = static_cast<int>((TransB == CblasNoTrans) ? N : K);

	cblas_sgemm(CblasRowMajor, TransA, TransB, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), alpha, A, lda, B, ldb, beta, C, static_cast<int>(N));
#else
#ifdef USE_THREADPOOL
	if (M > BIG_M_BLOCK) // threads are partitioned wrt M so make sure there are > 1 M blocks
	{
		matmul_mp(transA, transB, M, N, K, alpha, A, B, beta, C);
	}
	else
	{
		matmul(transA, transB, M, N, K, alpha, A, B, beta, C);
	}
	return;
#endif
	matmul(transA, transB, M, N, K, alpha, A, B, beta, C);
#endif
}

/*
template<typename Dtype>
void cpu_gemm<Dtype>(bool traspose_A, bool traspose_B, uint64_t M, uint64_t N, uint64_t K, Dtype alpha, Dtype* A, Dtype* B, Dtype beta, Dtype* C)
{
	LTEN_ERR("Not yet implemented");
}

template void cpu_gemm<int>(bool traspose_A, bool traspose_B, uint64_t M, uint64_t N, uint64_t K, int alpha, int* A, int* B, int beta, int* C);
template void cpu_gemm<uint8_t>(bool traspose_A, bool traspose_B, uint64_t M, uint64_t N, uint64_t K, uint8_t alpha, uint8_t* A, uint8_t* B, uint8_t beta, uint8_t* C);
*/

template<>
void cpu_gemm<int>(bool traspose_A, bool traspose_B, uint64_t M, uint64_t N, uint64_t K, int alpha, int* A, int* B, int beta, int* C)
{
	LTEN_ERR("Not yet implemented");
}

template<>
void cpu_gemm<uint8_t>(bool transA, bool transB, uint64_t M, uint64_t N, uint64_t K, uint8_t alpha, uint8_t* A, uint8_t* B, uint8_t beta, uint8_t* C)
{
	LTEN_ERR("Not yet implemented");
}

//-----------------------------------------------------
// quantized matrix multiplication
// workspace must be at least (M + N) * sizeof(int) bytes
//-----------------------------------------------------
void quantized_matmul(bool traspose_A, bool traspose_B, uint64_t M, uint64_t N, uint64_t K, uint8_t alpha, uint8_t* A, uint8_t* B, uint8_t beta, uint8_t* C, QuantizationParams* qparms, int* bias, int* workspace)
{
	uint64_t i;
	uint64_t j;
	uint64_t k;

	QuantizationParams* qpA;
	QuantizationParams* qpB;
	QuantizationParams* qpC;

	qpA = &qparms[0];
	qpB = &qparms[1];
	qpC = &qparms[2];

	int* row_vector = workspace;
	int* col_vector = workspace + N;

	uint64_t broadcast_val;
	int element;

	if (alpha != 1 || beta != 0)
	{
		LTEN_ERR("Support for alpha != 1 and beta != 0 not implemented");
	}

	if (traspose_A && traspose_B)
	{
		for (i = 0; i < N; i++)
		{
			row_vector[i] = 0;
			for (j = 0; j < K; j++)
			{
				row_vector[i] += B[i * K + j];
			}
			row_vector[i] *= (-qpA->zero_point);
		}

		for (i = 0; i < M; i++)
		{
			col_vector[i] = 0;
			for (j = 0; j < K; j++)
			{
				col_vector[i] += A[j * M + i];
			}
			col_vector[i] *= (-qpB->zero_point);
		}
	}
	else
	{
		if (traspose_B)
		{
			for (i = 0; i < N; i++)
			{
				row_vector[i] = 0;
				for (j = 0; j < K; j++)
				{
					row_vector[i] += B[i * K + j];
				}
				row_vector[i] *= (-qpA->zero_point);
			}

			for (i = 0; i < M; i++)
			{
				col_vector[i] = 0;
				for (j = 0; j < K; j++)
				{
					col_vector[i] += A[i * K + j];
				}
				col_vector[i] *= (-qpB->zero_point);
			}
		}
		else
		{
			if (traspose_A)
			{
				for (i = 0; i < N; i++)
				{
					row_vector[i] = 0;
					for (j = 0; j < K; j++)
					{
						row_vector[i] += B[j * N + i];
					}
					row_vector[i] *= (-qpA->zero_point);
				}

				for (i = 0; i < M; i++)
				{
					col_vector[i] = 0;
					for (j = 0; j < K; j++)
					{
						col_vector[i] += A[j * M + i];
					}
					col_vector[i] *= (-qpB->zero_point);
				}
			}
			else
			{
				for (i = 0; i < N; i++)
				{
					row_vector[i] = 0;
					for (j = 0; j < K; j++)
					{
						row_vector[i] += B[j * N + i];
					}
					row_vector[i] *= (-qpA->zero_point);
				}

				for (i = 0; i < M; i++)
				{
					col_vector[i] = 0;
					for (j = 0; j < K; j++)
					{
						col_vector[i] += A[i * K + j];
					}
					col_vector[i] *= (-qpB->zero_point);
				}
			}
		}
	}



	broadcast_val = K * qpA->zero_point * qpB->zero_point;
	assert(broadcast_val <= std::numeric_limits<int>::max());

	float real_multiplier = qpA->scale * qpB->scale / qpC->scale; // TODO convert to fixed point

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			element = 0;
			for (k = 0; k < K; k++)
			{
				if (traspose_A && traspose_B)
				{
					element += (A[k * M + i] * B[j * K + k]);
				}
				else
				{
					if (traspose_B)
					{
						element += (A[i * K + k] * B[j * K + k]);
					}
					else
					{
						if (traspose_A)
						{
							element += (A[k * M + i] * B[j + N * k]);
						}
						else
						{
							element += (A[i * K + k] * B[j + N * k]);
						}
					}
				}
			}

			element += row_vector[j] + col_vector[i] + static_cast<int>(broadcast_val);
			if (bias)
			{
				element += bias[j];
			}
			C[i * N + j] = (unsigned char)((real_multiplier * element) + qpC->zero_point);
		}
	}

}


// newly implemented cpu functions (not optimized yet though)
template void cpu_mean<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void cpu_mean<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);
template void cpu_mean<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);


template void cpu_var<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);
template void cpu_var<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);
template void cpu_var<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);

template void cpu_layer_norm<float>(float* dst, const float* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, float* weight, float* bias, float* ln, float* sd);
template void cpu_layer_norm<int>(int* dst, const int* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, int* weight, int* bias, int* ln, int* sd);
template void cpu_layer_norm<uint8_t>(uint8_t* dst, const uint8_t* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, uint8_t* weight, uint8_t* bias, uint8_t* ln, uint8_t* sd);

template void cpu_layer_norm_backwards<float>(void* vlayer_norm, float* x, float* top_gradient, float* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, float* feeder_gradient);
template void cpu_layer_norm_backwards<int>(void* vlayer_norm, int* x, int* top_gradient, int* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, int* feeder_gradient);
template void cpu_layer_norm_backwards<uint8_t>(void* vlayer_norm, uint8_t* x, uint8_t* top_gradient, uint8_t* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, uint8_t* feeder_gradient);


