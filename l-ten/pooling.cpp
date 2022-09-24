#include "lten.h"
#include "im_col.h"
#include "utils.h"

#ifdef USE_CUDA
namespace lten {
	bool pooling_CUDNN::init()
	{
		cudnnErrCheck(cudnnCreatePoolingDescriptor(&poolingDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));

		const int poolDims = 2;
		int windowDimA[poolDims] = { kernel_h_, kernel_w_ };
		int paddingA[poolDims] = { pad_h_, pad_w_ };
		int strideA[poolDims] = { stride_h_, stride_w_ };

		cudnnErrCheck(cudnnSetPoolingNdDescriptor(poolingDesc_, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, poolDims, windowDimA, paddingA, strideA));

		return true;
	}

	Tensor pooling_CUDNN::forward(Tensor& input)
	{
		const uint64_t* dims;
		TensorOps options;
		int ndims;
		float alpha;
		float beta;
		uint64_t result_dims[4]; // NCHW
		cudnnHandle_t cudnnHandle;


		dims = input.get_sizes();

		ndims = input.get_ndims();

		if (ndims != 4)
		{
			LTEN_ERR("pooling_CUDNN requires tensors with exactly 4 dimensions (NCHW)");
		}

		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("Invalid tensor device type");
		}

		cudnnErrCheck(cudnnSetTensor4dDescriptor(inputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]), static_cast<int>(dims[3])));

		cudnnErrCheck(cudnnGetPoolingNdForwardOutputDim(poolingDesc_, inputDesc_, 4, output_dims_));

		cudnnErrCheck(cudnnSetTensor4dDescriptor(outputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_dims_[0], output_dims_[1], output_dims_[2], output_dims_[3]));


		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		result_dims[0] = output_dims_[0];
		result_dims[1] = output_dims_[1];
		result_dims[2] = output_dims_[2];
		result_dims[3] = output_dims_[3];

		resultImpl->allocate(result_dims, ndims, &options);

		alpha = 1.0f;
		beta = 0;


		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		cudnnErrCheck(cudnnPoolingForward(cudnnHandle, poolingDesc_, &alpha, inputDesc_, input.get_data_ptr(), &beta, outputDesc_, result->get_data_ptr()));

		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(pooling_cudnn_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);

	}

	bool pooling3d_CUDNN::init()
	{
		cudnnErrCheck(cudnnCreatePoolingDescriptor(&poolingDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));

		const int poolDims = 3;
		int windowDimA[poolDims] = { kernel_h_, kernel_w_, kernel_c_ };
		int paddingA[poolDims] = { pad_h_, pad_w_, pad_c_ };
		int strideA[poolDims] = { stride_h_, stride_w_, stride_c_ };

		cudnnErrCheck(cudnnSetPoolingNdDescriptor(poolingDesc_, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, poolDims, windowDimA, paddingA, strideA));

		return true;
	}

	Tensor pooling3d_CUDNN::forward(Tensor& input)
	{
		const uint64_t* dims;
		TensorOps options;
		int ndims;
		float alpha;
		float beta;
		int dimsA[5];
		int strideA[5];
		uint64_t result_dims[5]; // NCDHW
		cudnnHandle_t cudnnHandle;
		int i;


		dims = input.get_sizes();

		ndims = input.get_ndims();

		if (ndims != 5)
		{
			LTEN_ERR("pooling_CUDNN requires tensors with exactly 4 dimensions (NCHW)");
		}

		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("Invalid tensor device type");
		}


		for (i = 0; i < ndims; i++)
		{
			dimsA[i] = static_cast<int>(dims[i]);
		}

		GetStrides(dimsA, strideA, ndims);
		cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, ndims, dimsA, strideA));
		//cudnnErrCheck(cudnnSetTensor4dDescriptor(inputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]), static_cast<int>(dims[3])));

		cudnnErrCheck(cudnnGetPoolingNdForwardOutputDim(poolingDesc_, inputDesc_, 5, output_dims_));


		GetStrides(output_dims_, strideA, ndims);
		cudnnErrCheck(cudnnSetTensorNdDescriptor(outputDesc_, CUDNN_DATA_FLOAT, ndims, output_dims_, strideA));
		//cudnnErrCheck(cudnnSetTensor4dDescriptor(outputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_dims_[0], output_dims_[1], output_dims_[2], output_dims_[3]));


		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		result_dims[0] = output_dims_[0];
		result_dims[1] = output_dims_[1];
		result_dims[2] = output_dims_[2];
		result_dims[3] = output_dims_[3];
		result_dims[4] = output_dims_[4];

		resultImpl->allocate(result_dims, ndims, &options);

		alpha = 1.0f;
		beta = 0;


		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		cudnnErrCheck(cudnnPoolingForward(cudnnHandle, poolingDesc_, &alpha, inputDesc_, input.get_data_ptr(), &beta, outputDesc_, result->get_data_ptr()));

		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(pooling_cudnn_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);

	}
} // namespace lten
#endif