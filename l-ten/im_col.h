#ifndef IM_COL_H
#define IM_COL_H


template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const uint64_t channels, const uint64_t height, const uint64_t width, const uint64_t kernel_h, const uint64_t kernel_w, const uint64_t pad_h, const uint64_t pad_w, const uint64_t stride_h, const uint64_t stride_w, Dtype* data_col)
{
	uint64_t height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	uint64_t width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	uint64_t channels_col = channels * kernel_h * kernel_w;

	for (uint64_t c = 0; c < channels_col; ++c)
	{
		uint64_t w_offset = c % kernel_w;
		uint64_t h_offset = (c / kernel_w) % kernel_h;
		uint64_t c_im = c / kernel_h / kernel_w;

		for (uint64_t h = 0; h < height_col; ++h)
		{
			for (uint64_t w = 0; w < width_col; ++w)
			{
				uint64_t h_pad = h * stride_h - pad_h + h_offset;
				uint64_t w_pad = w * stride_w - pad_w + w_offset;

				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					data_col[(c * height_col + h) * width_col + w] = data_im[(c_im * height + h_pad) * width + w_pad];
				else
					data_col[(c * height_col + h) * width_col + w] = 0;
			}
		}
	}
}

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const uint64_t channels, const uint64_t height, const uint64_t width, const uint64_t patch_h, const uint64_t patch_w, const uint64_t pad_h, const uint64_t pad_w, const uint64_t stride_h, const uint64_t stride_w,	Dtype* data_im)
{
	//caffe_set(height * width * channels, Dtype(0), data_im);
	uint64_t height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	uint64_t width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	uint64_t channels_col = channels * patch_h * patch_w;
	for (int c = 0; c < channels_col; ++c) 
	{
		uint64_t w_offset = c % patch_w;
		uint64_t h_offset = (c / patch_w) % patch_h;
		uint64_t c_im = c / patch_h / patch_w;
		for (uint64_t h = 0; h < height_col; ++h)
		{
			for (uint64_t w = 0; w < width_col; ++w)
			{
				uint64_t h_pad = h * stride_h - pad_h + h_offset;
				uint64_t w_pad = w * stride_w - pad_w + w_offset;
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					data_im[(c_im * height + h_pad) * width + w_pad] +=	data_col[(c * height_col + h) * width_col + w];
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
	Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const int dilation_h, const int dilation_w,
	Dtype* data_im);

#endif  // IM_COL_H
