#include <cstdint>

/*
COPYRIGHT

All contributions by the University of California:
Copyright (c) 2014-2017 The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014-2017, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright over
their contributions to Caffe. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTRIBUTION AGREEMENT

By contributing to the BVLC/caffe repository through pull-request, comment,
or otherwise, the contributor releases their content to the
license and copyright terms herein.*/


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

//#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

const int CAFFE_CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	Dtype* data_col) {
	CUDA_KERNEL_LOOP(index, n) {
		const int h_index = index / width_col;
		const int h_col = h_index % height_col;
		const int w_col = index % width_col;
		const int c_im = h_index / height_col;
		const int c_col = c_im * kernel_h * kernel_w;
		const int h_offset = h_col * stride_h - pad_h;
		const int w_offset = w_col * stride_w - pad_w;
		Dtype* data_col_ptr = data_col;
		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
		const Dtype* data_im_ptr = data_im;
		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
		for (int i = 0; i < kernel_h; ++i) {
			for (int j = 0; j < kernel_w; ++j) {
				int h_im = h_offset + i * dilation_h;
				int w_im = w_offset + j * dilation_w;
				*data_col_ptr =
					(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
					data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}


template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	Dtype* data_col) {
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
	im2col_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS >> > (
			num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
			pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
			width_col, data_col);
	//CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
	const int height, const int width, const int channels,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	Dtype* data_im) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype val = 0;
		const int w_im = index % width + pad_w;
		const int h_im = (index / width) % height + pad_h;
		const int c_im = index / (width * height);
		int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
		int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
		// compute the start and end of the output
		const int w_col_start =
			(w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
		const int w_col_end = min(w_im / stride_w + 1, width_col);
		const int h_col_start =
			(h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
		const int h_col_end = min(h_im / stride_h + 1, height_col);
		// TODO: use LCM of stride and dilation to avoid unnecessary loops
		for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
			for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
				int h_k = (h_im - h_col * stride_h);
				int w_k = (w_im - w_col * stride_w);
				if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
					h_k /= dilation_h;
					w_k /= dilation_w;
					int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
						height_col + h_col) * width_col + w_col;
					val += data_col[data_col_index];
				}
			}
		}
		data_im[index] = val;
	}
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const int dilation_h, const int dilation_w,
	Dtype* data_im) {
	int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
		stride_h + 1;
	int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
		stride_w + 1;
	int num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	// NOLINT_NEXT_LINE(whitespace/operators)
	col2im_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS >> > (
			num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
			pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
			height_col, width_col, data_im);
	//CUDA_POST_KERNEL_CHECK;
}

template void im2col_gpu<float>(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, float* data_col);


template void col2im_gpu<float>(const float* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const int dilation_h, const int dilation_w,
	float* data_im);

template void im2col_gpu<int>(const int* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, int* data_col);

template void col2im_gpu<int>(const int* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const int dilation_h, const int dilation_w,
	int* data_im);

template void im2col_gpu<uint8_t>(const uint8_t* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, uint8_t* data_col);

template void col2im_gpu<uint8_t>(const uint8_t* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const int dilation_h, const int dilation_w,
	uint8_t* data_im);


