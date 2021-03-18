#include <random>
#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "utils.h"


namespace lten {
	Tensor relu(Tensor& input)
	{
		uint64_t len;
		uint64_t i;
		TensorOps options;
		float* src;
		float* dst;

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(input.get_sizes(), input.get_ndims(), &options);

		len = input.get_numels();
		src = static_cast<float*>(input.get_data_ptr());
		dst = static_cast<float*>(resultImpl->get_data_ptr());

		if (CPU == options.device_type)
		{
			for (i = 0; i < len; i++)
			{
				if (src[i] > 0)
				{
					dst[i] = src[i];
				}
				else
				{
					dst[i] = 0;
				}
			}
		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_relu(dst, src, len);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(relu_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);
	}


	Tensor softmax(Tensor& input, int dim)
	{
		Tensor exps;
		Tensor max;
		Tensor input_minus_max;
		Tensor sum;

		max = input.max(dim);
		max = max.squeeze(0);
		max = max.unsqueeze(dim);

		input_minus_max = input - max;

		exps = input_minus_max.exp();

		sum = exps.sum(dim);
		sum = sum.squeeze(0);
		sum = sum.unsqueeze(dim);

		return exps.div(sum);
	}


	Tensor log_softmax(Tensor& input, int dim)
	{
		Tensor exps;
		Tensor max;
		Tensor input_minus_max;
		Tensor sum;

		max = input.max(dim);
		max = max.squeeze(0);
		max = max.unsqueeze(dim);

		input_minus_max = input - max;

		exps = input_minus_max.exp();

		sum = exps.sum(dim);

		sum = sum.squeeze(0);
		sum = sum.unsqueeze(dim);

		return exps.div(sum).log();
	}


#ifdef USE_CUDA
	bool softmax_CUDNN::init()
	{
		cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc_));

		return true;
	}

	Tensor softmax_CUDNN::forward(Tensor& input)
	{
		int ndims;
		TensorOps options;
		int n, c, h, w;
		float alpha;
		float beta;
		const uint64_t* dims;
		cudnnHandle_t cudnnHandle;

		dims = input.get_sizes();

		ndims = input.get_ndims();
		
		if (ndims != 4)
		{
			LTEN_ERR("softmax_CUDNN requires tensors with exactly 4 dimensions (NCHW)");
		}
		
		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("softmax_CUDNN only supports the GPU device type");
		}

		n = static_cast<int>(dims[0]);
		c = static_cast<int>(dims[1]);
		h = static_cast<int>(dims[2]);
		w = static_cast<int>(dims[3]);
		
		const int nDims = 4;
		int dimA[nDims] = { n,c,h,w };
		int strideA[nDims] = { c*h*w, h*w, w, 1 };


		cudnnErrCheck(cudnnSetTensorNdDescriptor(inputDesc_, CUDNN_DATA_FLOAT, 4, dimA, strideA));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(outputDesc_, CUDNN_DATA_FLOAT, 4, dimA, strideA));

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);


		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		resultImpl->allocate(dims, ndims, &options);

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);
		alpha = 1.0f;
		beta = 0;

		cudnnErrCheck(cudnnSoftmaxForward(cudnnHandle, algo_, CUDNN_SOFTMAX_MODE_INSTANCE,
			&alpha,
			inputDesc_,
			input.get_data_ptr(),
			&beta,
			outputDesc_,
			resultImpl->get_data_ptr()));



		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(softmax_cudnn_backward);
		resultImpl->set_autograd(true);
		

		return Tensor(result);

	}
#endif

	Tensor mse_loss(Tensor& input, Tensor& target)
	{
		Tensor diff;
		uint64_t numels;

		numels = input.get_numels();

		diff = input - target;
		diff = (diff * diff).sum();
		return (1.0f / numels) * diff;
	}


	// like libtorch, assumes input is result of log_loftmax
	Tensor nll_loss(Tensor& input, Tensor& target)
	{
		uint64_t dims[1];
		uint64_t len;
		uint64_t i;
		TensorOps options;
		float* input_data;
		float* target_data;
		float* loss;
		float val;

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		dims[0] = 1;
		resultImpl->allocate(dims, 1, &options);

		len = input.get_numels();
		input_data = static_cast<float*>(input.get_data_ptr());
		target_data = static_cast<float*>(target.get_data_ptr());
		loss = static_cast<float*>(resultImpl->get_data_ptr());

		if (CPU == options.device_type)
		{
			val = 0;
			for (i = 0; i < len; i++)
			{
				val += input_data[i] * target_data[i];
			}

			*loss = val * (-1.0f / input.get_sizes()[0]);

		}
		else
		{
			if (GPU == options.device_type)
			{
#ifdef USE_CUDA
				gpu_nll(input_data, target_data, loss, len, input.get_sizes()[0]);
#else
				LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
			}
			else
			{
				LTEN_ERR("Invalid tensor device type");
			}
		}


		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(target.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(nll_backward);
		resultImpl->set_autograd(true);

		return Tensor(result);

	}

} // namespace lten