#ifndef TENSOR_FNS_H
#define TENSOR_FNS_H

#include "offset_calc.h"




#ifdef USE_OPENBLAS
extern "C"
{
#include <cblas.h>
}
#endif

template<typename Dtype>
void gpu_mul(uint64_t N, Dtype* A, Dtype* B, Dtype* C);

template<typename Dtype>
void gpu_mul_backward(uint64_t N, Dtype* operand, Dtype* top_gradient, Dtype* bottom_gradient);

template<typename Dtype>
void gpu_mul_backward(Dtype* top_gradient, Dtype* bottom_gradient, Dtype* other_operand, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);

template<typename Dtype>
void gpu_mul(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);


template<typename Dtype>
void gpu_div(uint64_t N, Dtype* A, Dtype* B, Dtype* C);

template<typename Dtype>
void gpu_div_backward(uint64_t N, Dtype* operand1, Dtype* operand2, Dtype* top_gradient, Dtype* bottom_gradient, bool divisor);

template<typename Dtype>
void gpu_div_backward(Dtype* top_gradient, Dtype* bottom_gradient, Dtype* operand1, Dtype* operand2, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, bool divisor);

template<typename Dtype>
void gpu_div(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);


template<typename Dtype>
void gpu_add(uint64_t N, Dtype* A, Dtype* B, Dtype* C);

template<typename Dtype>
void gpu_add_backward(uint64_t N, Dtype* top_gradient, Dtype* bottom_gradient);

template<typename Dtype>
void gpu_add_backward(Dtype* top_gradient, Dtype* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides);

template<typename Dtype>
void gpu_add(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);

template<typename Dtype>
void gpu_sub(uint64_t N, Dtype* A, Dtype* B, Dtype* C);

template<typename Dtype>
void gpu_sub_backward(uint64_t N, Dtype* top_gradient, Dtype* bottom_gradient, Dtype scale);

template<typename Dtype>
void gpu_sub_backward(Dtype* top_gradient, Dtype* bottom_gradient, const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* tg_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* tg_strides, Dtype scale);

template<typename Dtype>
void gpu_sub(Dtype* A, Dtype* B, Dtype* C, const uint64_t numels, const uint64_t* a_strides, const uint64_t* b_strides, const uint64_t* c_strides, const uint64_t* a_dims, const uint64_t* b_dims, const uint64_t* c_dims, const int ndims);

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_mean_backward(Dtype* bottom_gradient, const Dtype* top_gradient, const uint64_t numels);

template<typename Dtype>
void gpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numoutputs, OffsetCalc_broadcast* offs, Dtype scale);

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels_dst, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template<typename Dtype>
void gpu_var(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, bool unbiased);

template<typename Dtype>
void gpu_layer_norm(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes, Dtype* weight, Dtype* bias, Dtype* ln, Dtype* sd);

template<typename Dtype>
void gpu_transpose(const Dtype* A, Dtype* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);

template<typename Dtype>
void gpu_repeat(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, const int ndims);

template<typename Dtype>
void gpu_repeat_backward(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t numels_src, const uint64_t* dims_src, int ndims_src, OffsetCalc_repeat_backwards* offs);

template<typename Dtype>
void gpu_repeat_interleave(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_array, const uint32_t* cummulative_times, int ndims, int ndims_times, int dim);

template<typename Dtype>
void gpu_repeat_interleave_backward(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t numels_src, OffsetCalc_repeat_interleave* offs);

template<typename Dtype>
void gpu_repeat_interleave_broadcast_backward(Dtype* dst, const Dtype* src, uint64_t numels_dst, uint64_t numels_src, uint32_t repeat_dim_dim, uint32_t repeat, uint32_t stride, OffsetCalc_repeat_interleave* offs); // special case for when all repeat values are the same (much faster)

template<typename Dtype>
void gpu_index(Dtype* dst, const Dtype* src, const int* indices, uint64_t copy_len, const uint64_t numels);

template<typename Dtype>
void gpu_index_backward(Dtype* dst, uint64_t numels_dst, const Dtype* src, const int* indices, int num_indices, uint64_t copy_len);

template<typename Dtype>
void gpu_permute(Dtype* dst, const Dtype* src, int ndims, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutations, bool reverse = false); // use reverse mode for back prop

template<typename Dtype>
void set_addresses(Dtype* A, Dtype* B, Dtype* C, POINTER_ARRAYS* addresses, const OFFSET_ARRAYS* offsets, const uint64_t num_addresses);

template<typename Dtype>
void gpu_layer_norm_backwards(void* vlayer_norm, Dtype* x, Dtype* top_gradient, Dtype* bottom_gradient, const uint64_t* dst_dims, uint32_t ndims, const uint32_t* axes, uint32_t naxes, Dtype* feeder_gradient);

template<typename Dtype>
void gpu_gelu(Dtype* dst, Dtype* src, uint64_t len);

template<typename Dtype>
void gpu_gelu_backward(Dtype* bottom_gradient, const Dtype* top_gradient, const Dtype* src, uint64_t len);

template<typename Dtype>
void gpu_nll_backward(Dtype* bottom_gradient, const Dtype* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);

template<typename Dtype>
void gpu_nll(Dtype* loss, const Dtype* probabilities, const Dtype* one_hot_indices, uint64_t len, OffsetCalc_nll* ofs);

template<typename Dtype>
void gpu_reduce(uint32_t numels_dst, uint32_t numels_src, OffsetCalc_reverse_broadcast* offs, Dtype* dst, Dtype* src);


#endif // TENSOR_FNS_H
