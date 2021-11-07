#ifndef MATH_FNS_H
#define MATH_FNS_H

#ifdef USE_OPENBLAS
extern "C" 
{
#include <cblas.h>
}
#endif


struct QuantizationParams
{
	float scale;
	uint8_t zero_point;
};

template<typename Dtype>
void cpu_gemm(bool transA, bool transB, uint64_t M, uint64_t N, uint64_t K, Dtype alpha, Dtype* A, Dtype* B, Dtype beta, Dtype* C);

template<typename Dtype>
void cpu_axpy(uint64_t N, Dtype alpha, Dtype* X, Dtype* Y, Dtype* C); // c = alpha * x + y

template<typename Dtype>
void cpu_axpby(uint64_t N, Dtype alpha, Dtype* X_ptr, Dtype beta, Dtype* Y_ptr, Dtype* C_ptr); // c = alpha * x + beta * y

template<typename Dtype>
void cpu_mul(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr); // c = a * b

template<typename Dtype>
void cpu_mul(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr, Dtype beta, Dtype* C_ptr); // c = beta * c + alpha * a * b

template<typename Dtype>
void cpu_mul(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr); // b = alpha * a

template<typename Dtype>
void cpu_add(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr); // b = alpha + a

template<typename Dtype>
void cpu_div(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr); // c += a / b

template<typename Dtype>
void cpu_div(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr, Dtype beta, Dtype* C_ptr); // c = (beta * c) +  (alpha * a / b)

template<typename Dtype>
void cpu_div_back(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr, Dtype* D_ptr); // special function for processing div_backward during backpropagation (calculates d += a * (-b) / (c * c))

template<typename Dtype>
void cpu_sig(uint64_t N, const Dtype* A_ptr, Dtype* B_ptr); // b = sig(a)

template<typename Dtype>
void cpu_tanh(uint64_t N, const Dtype* A_ptr, Dtype* B_ptr); // b = tanh(a)

template<typename Dtype>
void cpu_powx(uint64_t N, const Dtype* A_ptr, Dtype x, Dtype* B_ptr); // b = pow(a,x)

template<typename Dtype>
void cpu_copy(uint64_t N, Dtype* A_ptr, Dtype* B_ptr); // b = a

template<typename Dtype>
void cpu_max(const Dtype* src, Dtype* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_max_backward(Dtype* dst, const Dtype* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_sum(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_sum_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_mean(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_var(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_var_backward(Dtype* dst, const Dtype* src, const Dtype* op1, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void cpu_std(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode = true);

template<typename Dtype>
void cpu_std_backward(Dtype* dst, const Dtype* src, const Dtype* op1, const Dtype* std, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode = true);

template<typename Dtype>
void cpu_dropout(Dtype* dst, Dtype* src, unsigned int* mask, unsigned int threshold, Dtype scale, uint64_t len);

template<typename Dtype>
void cpu_sig_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels);

template<typename Dtype>
void cpu_tanh_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels);

void cpu_transpose(float* src, float* dst, int dim_1, int dim_2,
	int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1,
	int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels);


void quantized_matmul(bool traspose_A, bool traspose_B, uint64_t M, uint64_t N, uint64_t K, uint8_t alpha, uint8_t* A, uint8_t* B, uint8_t beta, uint8_t* C, QuantizationParams* qparms, int* bias, int* workspace); // workspace must be at least (M + N) * sizeof(int) bytes





//-------------------------------------CUDA functions----------------------------------------------------------------

template<typename Dtype>
void gpu_sum(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);

template<typename Dtype>
void gpu_sub(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);

// C =  A * B
template<typename Dtype>
void gpu_mul(uint64_t N, Dtype* A_ptr, Dtype* B_ptr, Dtype* C_ptr);

// C = beta * C + A * B
// supports broadcast semantics
// use this when C dims >= both A and B dims (avoid atomics)
template<typename Dtype>
void gpu_mul(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, Dtype beta);

// C = C + A * B
// supports broadcast semantics
// use this when C dims < either A or B dims (uses atomics, but no support for uint8_t)
void gpu_mul(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint64_t height_C, uint64_t width_C);

template<typename Dtype>
void gpu_mul(uint64_t N, Dtype alpha, Dtype* A, Dtype* B, Dtype beta, Dtype* C); // c = beta * c + alpha * a * b

template<typename Dtype>
void gpu_mul(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr); // b = alpha * a

// C =  A / B
// supports broadcast semantics
// use this when C dims >= both A and B dims (avoid atomics)
template<typename Dtype>
void gpu_div(Dtype* A, Dtype* B, Dtype* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B);

// C = C + A / B
// use this when C numels == A numels == B numels (no broadcast therefore fastest)
template<typename Dtype>
void gpu_div(uint64_t N, Dtype* A, Dtype* B, Dtype* C);

// C = A / B
// use this when C numels == A numels == B numels (no broadcast therefore fastest)
//template<typename Dtype>
//void gpu_div2(uint64_t N, Dtype* A, Dtype* B, Dtype* C);

// C = C + A / B
// supports broadcast semantics
// use this when C dims < either A or B dims (uses atomics, but no support for uint8_t)
void gpu_div(float* A, float* B, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint64_t height_C, uint64_t width_C);

// special function for processing div_backward during backpropagation (calculates d += a * (-b) / (c * c))
// supports broadcast semantics
// use this when D dims < A or B or C dims (uses atomics, but no support for uint8_t)
//template<typename Dtype>
void gpu_div_back(float* A, float* B, float* C, float* D, uint64_t height_A, uint64_t width_A, uint64_t height_B, uint64_t width_B, uint64_t height_C, uint64_t width_C, uint64_t height_D, uint64_t width_D);

template<typename Dtype>
void gpu_sum(Dtype* data, Dtype* sum, uint64_t len);

template<typename Dtype>
void gpu_nll(Dtype* input, Dtype* target, Dtype* loss, uint64_t len, uint64_t batches);

template<typename Dtype>
void gpu_scalar_mul(Dtype* A, Dtype* B, Dtype scalar, uint64_t len); // B = scaler * A

// C += alpha * A
// supports broadcast semantics
// use this when C dims < A dims (uses atomics, but no support for uint8_t)
void gpu_scalar_mul(float alpha, float* A, float* C, uint64_t height_A, uint64_t width_A, uint64_t height_C, uint64_t width_C);

template<typename Dtype>
void gpu_fill(Dtype* memory, uint64_t size, Dtype value);

template<typename Dtype>
void gpu_fill(Dtype* memory, uint64_t len, Dtype* value);

void gpu_sgd_step(float* weight_ptr, float* weight_grad_ptr, float* velocity_ptr, int64_t numels, float mo, float wd, float lr);

template<typename Dtype>
void gpu_max(const Dtype* src, Dtype* dst, uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_max_backward(Dtype* dst, const Dtype* src, const uint64_t* indices, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_powx(uint64_t N, const Dtype* A, Dtype x, Dtype* B); // b = pow(a,x)

template<typename Dtype>
void gpu_exp(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_log(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_sig(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_sig_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels);

template<typename Dtype>
void gpu_tanh(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_tanh_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels);

template<typename Dtype>
void gpu_sum(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_sum_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_add(uint64_t N, Dtype alpha, Dtype* A_ptr, Dtype* B_ptr); // b = alpha + a

template<typename Dtype>
void gpu_axpy(uint64_t N, Dtype alpha, Dtype* X_ptr, Dtype* Y_ptr, Dtype* C_ptr);

template<typename Dtype>
void gpu_axpby(uint64_t N, Dtype alpha, Dtype* X_ptr, Dtype beta, Dtype* Y_ptr, Dtype* C_ptr); // c = alpha * x + beta * y

template<typename Dtype>
void gpu_relu(Dtype* dst, Dtype* src, uint64_t len);

template<typename Dtype>
void gpu_relu_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t len);


template<typename Dtype>
void gpu_dropout(Dtype* dst, Dtype* src, unsigned int* mask, unsigned int threshold, Dtype scale, uint64_t len);

template<typename Dtype>
void gpu_transpose(Dtype* src, Dtype* dst, int dim_1, int dim_2,
	int stride_src_dim_1, int stride_src_dim_1_minus_1, int stride_src_dim_2, int stride_src_dim_2_minus_1,
	int stride_trn_dim_1, int stride_trn_dim_1_minus_1, int stride_trn_dim_2, int stride_trn_dim_2_minus_1, uint64_t numels);

template<typename Dtype>
void gpu_cat(Dtype* dest, Dtype* op1, Dtype* op2, uint64_t dest_stride_1, uint64_t dest_stride_2, uint64_t op1_stride, uint64_t op2_stride_1, uint64_t op2_stride_2, uint64_t dim_offset, uint64_t op1_numels, uint64_t op2_numels);


template<typename Dtype>
void gpu_cat_backward(Dtype* dest, Dtype* src, uint64_t dest_stride, uint64_t src_stride, uint64_t dest_numels);

template<typename Dtype>
void gpu_cat_backward(Dtype* dest, Dtype* src, uint64_t dest_stride_1, uint64_t dest_stride_2, uint64_t src_stride_1, uint64_t src_stride_2, uint64_t dim_offset, uint64_t op1_numels, uint64_t dest_numels);

template<typename Dtype>
void gpu_embedding(Dtype* dst, Dtype* wts, int* indices, uint64_t numels, uint64_t indices_per_batch, unsigned int embedding_dim);

template<typename Dtype>
void gpu_embedding_backward(Dtype* dst, Dtype* wts, int* indices, uint64_t numels, uint64_t indices_per_batch, unsigned int embedding_dim);

template<typename Dtype>
void gpu_sqrt(Dtype* dst, const Dtype* src, const uint64_t numels);

template<typename Dtype>
void gpu_sqrt_backward(Dtype* bottom, const Dtype* top, const Dtype* middle, const uint64_t numels);

template<typename Dtype>
void gpu_mean(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_mean_backward(Dtype* dst, const Dtype* src, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_var(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_var_backward(Dtype* dst, const Dtype* src, const Dtype* op1, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride);

template<typename Dtype>
void gpu_std(const Dtype* src, Dtype* dst, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode = true);

template<typename Dtype>
void gpu_std_backward(Dtype* dst, const Dtype* src, const Dtype* op1, const Dtype* std, const uint64_t numels, const uint64_t ratio, const uint64_t dim_size, const uint64_t stride, bool sample_mode = true);

#endif // MATH_FNS_H
