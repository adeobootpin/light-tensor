#ifndef TENSOR_FNS_H
#define TENSOR_FNS_H

#ifdef USE_OPENBLAS
extern "C"
{
#include <cblas.h>
}
#endif

template<typename Dtype>
void gpu_mean(Dtype* dst, const Dtype* src, const uint64_t numels);

#endif // TENSOR_FNS_H