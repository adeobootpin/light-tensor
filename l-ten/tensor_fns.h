#ifndef TENSOR_FNS_H
#define TENSOR_FNS_H

enum { MAX_DIMS = 16 };

#include "offset_calc.h"

typedef struct TAG_POINTER_ARRAYS
{
	void** a_array;
	void** b_array;
	void** c_array;
	void* buffer;
}POINTER_ARRAYS;

typedef struct TAG_OFFSET_ARRAYS
{
	uint32_t* a_array;
	uint32_t* b_array;
	uint32_t* c_array;
	void* buffer;
}OFFSET_ARRAYS;


#ifdef USE_OPENBLAS
extern "C"
{
#include <cblas.h>
}
#endif

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_mean_backward(Dtype* bottom_gradient, const Dtype* top_gradient, const uint64_t numels);

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels_dst, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template<typename Dtype>
void gpu_var(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template<typename Dtype>
void gpu_layer_norm(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd);

template<typename Dtype>
void gpu_transpose(const Dtype* A, Dtype* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);

template<typename Dtype>
void gpu_repeat(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);

template<typename Dtype>
void gpu_repeat_interleave(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);

template<typename Dtype>
void gpu_index(Dtype* dst, const Dtype* src, const int* indices, uint64_t copy_len, const uint64_t numels);

template<typename Dtype>
void gpu_permute(Dtype* dst, const Dtype* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutations);

template<typename Dtype>
void set_addresses(Dtype* A, Dtype* B, Dtype* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);

#endif // TENSOR_FNS_H