#ifndef TENSOR_FNS_H
#define TENSOR_FNS_H

enum { MAX_DIMS = 16 };

#include "offset_calc.h"

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
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template<typename Dtype>
void gpu_var(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes);

template<typename Dtype>
void gpu_transpose(const Dtype* A, Dtype* At, const uint64_t numels, const uint64_t* a_strides, const uint64_t* at_strides, const int ndims);

#endif // TENSOR_FNS_H