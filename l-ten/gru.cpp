#include <random>
#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "utils.h"


namespace lten {
	bool GRU::init()
	{
		float* raw_data_ptr;
		uint64_t numels;
		uint64_t i;
		TensorOps options;
		TensorOps options_wts;

		options_wts.alloc_gradient_buffer = true;
		weights_u_ptr_ = new Tensor;
		*weights_u_ptr_ = AllocateTensor({ 1, input_dim_, hidden_dim_ * 3 }, &options_wts);
		weights_u_ptr_->set_autograd(true);

		weights_w_ptr_ = new Tensor;
		*weights_w_ptr_ = AllocateTensor({ 1, hidden_dim_ , hidden_dim_ * 3 }, &options_wts);
		weights_w_ptr_->set_autograd(true);

		if (use_bias_)
		{
			bias_u_ptr_ = new Tensor;
			*bias_u_ptr_ = AllocateTensor({ 1, 1, hidden_dim_ * 3 }, &options_wts);
			bias_u_ptr_->set_autograd(true);

			bias_w_ptr_ = new Tensor;
			*bias_w_ptr_ = AllocateTensor({ 1, 1, hidden_dim_ * 3 }, &options_wts);
			bias_w_ptr_->set_autograd(true);
		}

		if (bidirectional_)
		{
			weights_u_rev_ptr_ = new Tensor;
			*weights_u_rev_ptr_ = AllocateTensor({ 1, input_dim_, hidden_dim_ * 3 }, &options_wts);
			weights_u_rev_ptr_->set_autograd(true);

			weights_w_rev_ptr_ = new Tensor;
			*weights_w_rev_ptr_ = AllocateTensor({ 1, hidden_dim_ , hidden_dim_ * 3 }, &options_wts);
			weights_w_rev_ptr_->set_autograd(true);

			if (use_bias_)
			{
				bias_u_rev_ptr_ = new Tensor;
				*bias_u_rev_ptr_ = AllocateTensor({ 1, 1, hidden_dim_ * 3 }, &options_wts);
				bias_u_rev_ptr_->set_autograd(true);

				bias_w_rev_ptr_ = new Tensor;
				*bias_w_rev_ptr_ = AllocateTensor({ 1, 1, hidden_dim_ * 3 }, &options_wts);
				bias_w_rev_ptr_->set_autograd(true);
			}
		}




		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-sqrtf(1.0f/hidden_dim_), sqrtf(1.0f / hidden_dim_));

		raw_data_ptr = (float*)weights_u_ptr_->get_data_ptr();
		numels = weights_u_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
			//raw_data_ptr[i] = 0.000362f;
		}

		raw_data_ptr = (float*)weights_w_ptr_->get_data_ptr();
		numels = weights_w_ptr_->get_numels();

		for (i = 0; i < numels; i++)
		{
			raw_data_ptr[i] = distribution(generator);
			//raw_data_ptr[i] = 0.000362f;
		}


		if (use_bias_)
		{
			raw_data_ptr = (float*)bias_u_ptr_->get_data_ptr();
			numels = bias_u_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
				//raw_data_ptr[i] = 0.000362f;
			}

			raw_data_ptr = (float*)bias_w_ptr_->get_data_ptr();
			numels = bias_w_ptr_->get_numels();
			for (i = 0; i < numels; i++)
			{
				raw_data_ptr[i] = distribution(generator);
				//raw_data_ptr[i] = 0.000362f;
			}
		}

		scratch_hidden_state_ = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
		matmul_0_ = AllocateTensor({ 1, 1, 3 * hidden_dim_ }, &options);
		matmul_1_ = AllocateTensor({ 1, 1, 3 * hidden_dim_ }, &options);


		extra_workspace_[0] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
		extra_workspace_[1] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
		extra_workspace_[2] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
		extra_workspace_[3] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
		extra_workspace_[4] = AllocateTensor({ 1, 1, 3 * hidden_dim_ }, &options);
		extra_workspace_[5] = AllocateTensor({ 1, 1, 3 * hidden_dim_ }, &options);
		extra_workspace_[6] = AllocateTensor({ 1, 1, 3 * hidden_dim_ }, &options);

		return true;
	}


	Tensor GRU::forward(Tensor& input)
	{
		uint64_t sequence_len;
		int ndims;
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		uint64_t batches;
		int i;
		int j;
		uint64_t offset;
		float* hidden_state;
		float* input_step;
		float* matmul_0;
		float* matmul_1;
		float* u_r_x;
		float* u_z_x;
		float* u_hc_x;
		float* w_r_h;
		float* w_z_h;

		float* weights_u;
		float* weights_w;
		float* bias_u = nullptr;
		float* bias_w = nullptr;

		float* weights_u_rev = nullptr;
		float* weights_w_rev = nullptr;
		float* bias_u_rev = nullptr;
		float* bias_w_rev = nullptr;

		float* tmp_5;
		float* z_t;
		float* r_t;
		float* w_hc_h;
		float* hc_t;


		ndims = input.get_ndims();
		if (ndims != 3) // [batches, sequ_len, input_dim]
		{
			LTEN_ERR("GRU requires tensors with exactly 3 dimensions (batches, sequ_len, input_dim)");
		}

		if (input.get_device() != weights_u_ptr_->get_device() || // for speed, check just one of the weight pointers (it is safe to assume the others are on the same device)
			input.get_device_index() != weights_u_ptr_->get_device_index() ||
			input.get_data_type() != weights_u_ptr_->get_data_type())
		{
			LTEN_ERR("Input tensor device type or data type are not the same as this layer's parameters");
		}

		memcpy(dims, input.get_sizes(), sizeof(uint64_t) * ndims);

		if (dims[ndims - 1] != input_dim_)
		{
			LTEN_ERR("Last dimension must be equal to the size of the input");
		}
		dims[ndims - 1] = hidden_dim_;


		sequence_len = input.get_sizes()[input.get_ndims() - 2];

		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		if (bidirectional_)
		{
			dims[ndims - 1] *= 2;
		}
		resultImpl->allocate(dims, ndims, &options);

		batches = input.get_sizes()[0];

		hidden_state = static_cast<float*>(scratch_hidden_state_.get_data_ptr());
		matmul_0 = static_cast<float*>(matmul_0_.get_data_ptr());
		matmul_1 = static_cast<float*>(matmul_1_.get_data_ptr());


		weights_u = static_cast<float*>(weights_u_ptr_->get_data_ptr());
		weights_w = static_cast<float*>(weights_w_ptr_->get_data_ptr());
		if (use_bias_)
		{
			bias_u = static_cast<float*>(bias_u_ptr_->get_data_ptr());
			bias_w = static_cast<float*>(bias_w_ptr_->get_data_ptr());;
		}

		if (bidirectional_)
		{
			weights_u_rev = static_cast<float*>(weights_u_rev_ptr_->get_data_ptr());
			weights_w_rev = static_cast<float*>(weights_w_rev_ptr_->get_data_ptr());
			if (use_bias_)
			{
				bias_u_rev = static_cast<float*>(bias_u_rev_ptr_->get_data_ptr());
				bias_w_rev = static_cast<float*>(bias_w_rev_ptr_->get_data_ptr());;
			}
		}


		//--------------------------------------------------------------------------------------------
		for (i = 0; i < batches; i++)
		{
			for (j = 0; j < sequence_len; j++)
			{
				hidden_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);

				tmp_5_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
				z_t_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
				r_t_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
				w_hc_h_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
				hc_t_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);

				if (bidirectional_)
				{
					hidden_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);

					tmp_5_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
					z_t_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
					r_t_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
					w_hc_h_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
					hc_t_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options);
				}
			}
			hidden_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options); // need one more hidden state for output
			if (bidirectional_)
			{
				hidden_rev_[i * sequence_len + j] = AllocateTensor({ 1, 1, hidden_dim_ }, &options); // need one more reverse hidden state for output
			}
		}
		//--------------------------------------------------------------------------------------------



		for (i = 0; i < batches; i++)
		{
			if (CPU == options.device_type)
			{
				memset(hidden_[i * sequence_len + 0].get_data_ptr(), 0, hidden_dim_ * sizeof(float));
				memset(hidden_state, 0, hidden_dim_ * sizeof(float)); // logically hidden_state = h[0]

				for (j = 0; j < sequence_len; j++)
				{
					input_step = static_cast<float*>(input.get_data_ptr()) + i * sequence_len * input_dim_ + j * input_dim_;

					cpu_gemm(false, false, 1, 3 * hidden_dim_, input_dim_, 1.0f, input_step, weights_u, 0.0f, matmul_0);
					cpu_gemm(false, false, 1, 3 * hidden_dim_, hidden_dim_, 1.0f, hidden_state, weights_w, 0.0f, matmul_1);

					if (use_bias_)
					{
						cpu_axpy(3 * hidden_dim_, 1.0f, bias_u, matmul_0, matmul_0);
						cpu_axpy(3 * hidden_dim_, 1.0f, bias_w, matmul_1, matmul_1);
					}

					offset = i * sequence_len + j;
					u_r_x = matmul_0;
					u_z_x = u_r_x + hidden_dim_;
					u_hc_x = u_r_x + 2 * hidden_dim_;

					w_r_h = matmul_1;
					w_z_h = w_r_h + hidden_dim_;
					w_hc_h = static_cast<float*>(w_hc_h_[offset].get_data_ptr()); // need to save w_hc_h for backprop
					memcpy(w_hc_h, w_r_h + 2 * hidden_dim_, sizeof(float) * hidden_dim_);

					//
					//r_t = (w_r_h + u_r_x).sig();
					//
					r_t = static_cast<float*>(r_t_[offset].get_data_ptr()); // need to save r_t for backprop
					cpu_axpy(hidden_dim_, 1.0f, w_r_h, u_r_x, r_t); // tmp0 (w_r_h + u_r_x)
					cpu_sig(hidden_dim_, r_t, r_t); // r_t = sig(tmp0)

					//
					//z_t = (w_z_h + u_z_x).sig();
					//
					z_t = static_cast<float*>(z_t_[offset].get_data_ptr()); // need to save z_t for backprop
					cpu_axpy(hidden_dim_, 1.0f, w_z_h, u_z_x, z_t); //tmp1 (w_z_h + u_z_x)
					cpu_sig(hidden_dim_, z_t, z_t); // z_t = sig(tmp1)


					//
					//hc_t = (u_hc_x + (r_t * w_hc_h)).tanh();
					//
					hc_t = static_cast<float*>(hc_t_[offset].get_data_ptr()); // need to save hc_t for backprop
					cpu_mul(hidden_dim_, r_t, w_hc_h, hc_t); // tmp2 (r_t * w_hc_h)
					cpu_axpy(hidden_dim_, 1.0f, u_hc_x, hc_t, hc_t); // tmp3 (u_hc_x + tmp2)
					cpu_tanh(hidden_dim_, hc_t, hc_t); // hc_t = tanh(tmp3)


					//
					// hidden_state[i+1] = hc_t + z_t * ((-1.0f * hc_t) + hidden_state[i];
					//
					tmp_5 = static_cast<float*>(tmp_5_[offset].get_data_ptr()); // need to save tmp_5 for backprop
					cpu_axpy(hidden_dim_, -1.0f, hc_t, hidden_state, tmp_5); // tmp4/tmp5 (((-1.0f * hc_t) + hidden_state[i])
					cpu_mul(hidden_dim_, z_t, tmp_5, hidden_state); // tmp6 = tmp5 * z_t
					cpu_axpy(hidden_dim_, 1.0f, hc_t, hidden_state, hidden_state); // h[i+1] = tmp6 + hc_t

					memcpy(hidden_[offset + 1].get_data_ptr(), hidden_state, sizeof(float) * hidden_dim_);

					if (bidirectional_)
					{
						memcpy(static_cast<float*>(resultImpl->get_data_ptr()) + i * sequence_len * 2 * hidden_dim_ + 2 * hidden_dim_ * j, hidden_state, sizeof(float) * hidden_dim_);
					}
					else
					{
						memcpy(static_cast<float*>(resultImpl->get_data_ptr()) + i * sequence_len * hidden_dim_ + hidden_dim_ * j, hidden_state, sizeof(float) * hidden_dim_);
					}
				}
			}
			else
			{
				if (GPU == options.device_type)
				{
#ifdef USE_CUDA
					cublasStatus_t status;
					cublasHandle_t hCuBlas;

					hCuBlas = CUDA_globlas::singleton()->get_cublas_handle(options.device_index);

					float alpha = 1.0f;
					float beta = 0.0f;

					int M;
					int N;
					int K;

					M = 1;
					N = 3 * static_cast<int>(hidden_dim_);

					ZeroMemoryOnGPU(hidden_[i * sequence_len + 0].get_data_ptr(), hidden_dim_ * sizeof(float));
					ZeroMemoryOnGPU(hidden_state, hidden_dim_ * sizeof(float)); // logically hidden_state = h[0]

					for (j = 0; j < sequence_len; j++)
					{
						input_step = static_cast<float*>(input.get_data_ptr()) + i * sequence_len * input_dim_ + j * input_dim_;

						K = static_cast<int>(input_dim_);
						LTEN_CUBLAS_CHECK_2(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weights_u, N, input_step, K, &beta, matmul_0, N));

						K = static_cast<int>(hidden_dim_);
						LTEN_CUBLAS_CHECK_2(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weights_w, N, hidden_state, K, &beta, matmul_1, N));

						if (use_bias_)
						{
							gpu_axpy(3 * hidden_dim_, 1.0f, bias_u, matmul_0, matmul_0);
							gpu_axpy(3 * hidden_dim_, 1.0f, bias_w, matmul_1, matmul_1);
						}

						offset = i * sequence_len + j;
						u_r_x = matmul_0;
						u_z_x = u_r_x + hidden_dim_;
						u_hc_x = u_r_x + 2 * hidden_dim_;

						w_r_h = matmul_1;
						w_z_h = w_r_h + hidden_dim_;
						w_hc_h = static_cast<float*>(w_hc_h_[offset].get_data_ptr()); // need to save w_hc_h for backprop
						GPUToGPUCopy(w_hc_h, w_r_h + 2 * hidden_dim_, sizeof(float) * hidden_dim_);

						//
						//r_t = (w_r_h + u_r_x).sig();
						//
						r_t = static_cast<float*>(r_t_[offset].get_data_ptr()); // need to save r_t for backprop
						gpu_axpy(hidden_dim_, 1.0f, w_r_h, u_r_x, r_t); // tmp0 (w_r_h + u_r_x)
						gpu_sig(r_t, r_t, hidden_dim_); // r_t = sig(tmp0)

						//
						//z_t = (w_z_h + u_z_x).sig();
						//
						z_t = static_cast<float*>(z_t_[offset].get_data_ptr()); // need to save z_t for backprop
						gpu_axpy(hidden_dim_, 1.0f, w_z_h, u_z_x, z_t); //tmp1 (w_z_h + u_z_x)
						gpu_sig(z_t, z_t, hidden_dim_); // z_t = sig(tmp1)


						//
						//hc_t = (u_hc_x + (r_t * w_hc_h)).tanh();
						//
						hc_t = static_cast<float*>(hc_t_[offset].get_data_ptr()); // need to save hc_t for backprop
						gpu_mul(hidden_dim_, 1.0f, r_t, w_hc_h, 0.0f, hc_t); // tmp2 (r_t * w_hc_h)
						gpu_axpy(hidden_dim_, 1.0f, u_hc_x, hc_t, hc_t); // tmp3 (u_hc_x + tmp2)
						gpu_tanh(hc_t, hc_t, hidden_dim_); // hc_t = tanh(tmp3)


						//
						// hidden_state[i+1] = hc_t + z_t * ((-1.0f * hc_t) + hidden_state[i];
						//
						tmp_5 = static_cast<float*>(tmp_5_[offset].get_data_ptr()); // need to save tmp_5 for backprop
						gpu_axpy(hidden_dim_, -1.0f, hc_t, hidden_state, tmp_5); // tmp4/tmp5 (((-1.0f * hc_t) + hidden_state[i])
						gpu_mul(hidden_dim_, 1.0f, z_t, tmp_5, 0.0f, hidden_state); // tmp6 = tmp5 * z_t
						gpu_axpy(hidden_dim_, 1.0f, hc_t, hidden_state, hidden_state); // h[i+1] = tmp6 + hc_t

						GPUToGPUCopy(hidden_[offset + 1].get_data_ptr(), hidden_state, sizeof(float) * hidden_dim_);

						if (bidirectional_)
						{
							GPUToGPUCopy(static_cast<float*>(resultImpl->get_data_ptr()) + i * sequence_len * 2 * hidden_dim_ + 2 * hidden_dim_ * j, hidden_state, sizeof(float) * hidden_dim_);
						}
						else
						{
							GPUToGPUCopy(static_cast<float*>(resultImpl->get_data_ptr()) + i * sequence_len * hidden_dim_ + hidden_dim_ * j, hidden_state, sizeof(float) * hidden_dim_);
						}
					}
#else
					LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
				}
				else
				{
					LTEN_ERR("Invalid tensor device type");
				}
			}

		}

		if (bidirectional_)
		{
			for (i = 0; i < batches; i++)
			{
				if (CPU == options.device_type)
				{
					memset(hidden_rev_[i * sequence_len + 0].get_data_ptr(), 0, hidden_dim_ * sizeof(float));
					memset(hidden_state, 0, hidden_dim_ * sizeof(float)); // logically hidden_rev_state = h_rev[0]

					for (j = 0; j < sequence_len; j++)
					{
						input_step = static_cast<float*>(input.get_data_ptr()) + i * sequence_len * input_dim_ + (sequence_len - j - 1) * input_dim_;

						cpu_gemm(false, false, 1, 3 * hidden_dim_, input_dim_, 1.0f, input_step, weights_u_rev, 0.0f, matmul_0);
						cpu_gemm(false, false, 1, 3 * hidden_dim_, hidden_dim_, 1.0f, hidden_state, weights_w_rev, 0.0f, matmul_1);

						if (use_bias_)
						{
							cpu_axpy(3 * hidden_dim_, 1.0f, bias_u_rev, matmul_0, matmul_0);
							cpu_axpy(3 * hidden_dim_, 1.0f, bias_w_rev, matmul_1, matmul_1);
						}

						offset = i * sequence_len + j;
						u_r_x = matmul_0;
						u_z_x = u_r_x + hidden_dim_;
						u_hc_x = u_r_x + 2 * hidden_dim_;

						w_r_h = matmul_1;
						w_z_h = w_r_h + hidden_dim_;
						w_hc_h = static_cast<float*>(w_hc_h_rev_[offset].get_data_ptr()); // need to save w_hc_h for backprop
						memcpy(w_hc_h, w_r_h + 2 * hidden_dim_, sizeof(float) * hidden_dim_);

						//
						//r_t = (w_r_h + u_r_x).sig();
						//
						r_t = static_cast<float*>(r_t_rev_[offset].get_data_ptr()); // need to save r_t for backprop
						cpu_axpy(hidden_dim_, 1.0f, w_r_h, u_r_x, r_t); // tmp0 (w_r_h + u_r_x)
						cpu_sig(hidden_dim_, r_t, r_t); // r_t = sig(tmp0)

						//
						//z_t = (w_z_h + u_z_x).sig();
						//
						z_t = static_cast<float*>(z_t_rev_[offset].get_data_ptr()); // need to save z_t for backprop
						cpu_axpy(hidden_dim_, 1.0f, w_z_h, u_z_x, z_t); //tmp1 (w_z_h + u_z_x)
						cpu_sig(hidden_dim_, z_t, z_t); // z_t = sig(tmp1)


						//
						//hc_t = (u_hc_x + (r_t * w_hc_h)).tanh();
						//
						hc_t = static_cast<float*>(hc_t_rev_[offset].get_data_ptr()); // need to save hc_t for backprop
						cpu_mul(hidden_dim_, r_t, w_hc_h, hc_t); // tmp2 (r_t * w_hc_h)
						cpu_axpy(hidden_dim_, 1.0f, u_hc_x, hc_t, hc_t); // tmp3 (u_hc_x + tmp2)
						cpu_tanh(hidden_dim_, hc_t, hc_t); // hc_t = tanh(tmp3)


						//
						// hidden_state[i+1] = hc_t + z_t * ((-1.0f * hc_t) + hidden_state[i];
						//
						tmp_5 = static_cast<float*>(tmp_5_rev_[offset].get_data_ptr()); // need to save tmp_5 for backprop
						cpu_axpy(hidden_dim_, -1.0f, hc_t, hidden_state, tmp_5); // tmp4/tmp5 (((-1.0f * hc_t) + hidden_state[i])
						cpu_mul(hidden_dim_, z_t, tmp_5, hidden_state); // tmp6 = tmp5 * z_t
						cpu_axpy(hidden_dim_, 1.0f, hc_t, hidden_state, hidden_state); // h[i+1] = tmp6 + hc_t

						memcpy(hidden_rev_[offset + 1].get_data_ptr(), hidden_state, sizeof(float) * hidden_dim_);

						memcpy(static_cast<float*>(resultImpl->get_data_ptr()) + i * sequence_len * 2 * hidden_dim_ + 2 * hidden_dim_ * (sequence_len - j - 1) + hidden_dim_, hidden_state, sizeof(float) * hidden_dim_);
					}
				}
				else
				{
					if (GPU == options.device_type)
					{
#ifdef USE_CUDA
						cublasStatus_t status;
						cublasHandle_t hCuBlas;

						hCuBlas = CUDA_globlas::singleton()->get_cublas_handle(options.device_index);

						float alpha = 1.0f;
						float beta = 0.0f;

						int M;
						int N;
						int K;

						M = 1;
						N = 3 * static_cast<int>(hidden_dim_);

						ZeroMemoryOnGPU(hidden_rev_[i * sequence_len + 0].get_data_ptr(), hidden_dim_ * sizeof(float));
						ZeroMemoryOnGPU(hidden_state, hidden_dim_ * sizeof(float)); // logically hidden_rev_state = h_rev[0]

						for (j = 0; j < sequence_len; j++)
						{
							input_step = static_cast<float*>(input.get_data_ptr()) + i * sequence_len * input_dim_ + (sequence_len - j - 1) * input_dim_;

							K = static_cast<int>(input_dim_);
							LTEN_CUBLAS_CHECK_2(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weights_u_rev, N, input_step, K, &beta, matmul_0, N));
							
							K = static_cast<int>(hidden_dim_);
							LTEN_CUBLAS_CHECK_2(cublasSgemm(hCuBlas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weights_w_rev, N, hidden_state, K, &beta, matmul_1, N));

							if (use_bias_)
							{
								gpu_axpy(3 * hidden_dim_, 1.0f, bias_u_rev, matmul_0, matmul_0);
								gpu_axpy(3 * hidden_dim_, 1.0f, bias_w_rev, matmul_1, matmul_1);
							}

							offset = i * sequence_len + j;
							u_r_x = matmul_0;
							u_z_x = u_r_x + hidden_dim_;
							u_hc_x = u_r_x + 2 * hidden_dim_;

							w_r_h = matmul_1;
							w_z_h = w_r_h + hidden_dim_;
							w_hc_h = static_cast<float*>(w_hc_h_rev_[offset].get_data_ptr()); // need to save w_hc_h for backprop
							GPUToGPUCopy(w_hc_h, w_r_h + 2 * hidden_dim_, sizeof(float) * hidden_dim_);

							//
							//r_t = (w_r_h + u_r_x).sig();
							//
							r_t = static_cast<float*>(r_t_rev_[offset].get_data_ptr()); // need to save r_t for backprop
							gpu_axpy(hidden_dim_, 1.0f, w_r_h, u_r_x, r_t); // tmp0 (w_r_h + u_r_x)
							gpu_sig(r_t, r_t, hidden_dim_); // r_t = sig(tmp0)

							//
							//z_t = (w_z_h + u_z_x).sig();
							//
							z_t = static_cast<float*>(z_t_rev_[offset].get_data_ptr()); // need to save z_t for backprop
							gpu_axpy(hidden_dim_, 1.0f, w_z_h, u_z_x, z_t); //tmp1 (w_z_h + u_z_x)
							gpu_sig(z_t, z_t, hidden_dim_); // z_t = sig(tmp1)


							//
							//hc_t = (u_hc_x + (r_t * w_hc_h)).tanh();
							//
							hc_t = static_cast<float*>(hc_t_rev_[offset].get_data_ptr()); // need to save hc_t for backprop
							gpu_mul(hidden_dim_, 1.0f, r_t, w_hc_h, 0.0f, hc_t); // tmp2 (r_t * w_hc_h)
							gpu_axpy(hidden_dim_, 1.0f, u_hc_x, hc_t, hc_t); // tmp3 (u_hc_x + tmp2)
							gpu_tanh(hc_t, hc_t, hidden_dim_); // hc_t = tanh(tmp3)


							//
							// hidden_state[i+1] = hc_t + z_t * ((-1.0f * hc_t) + hidden_state[i];
							//
							tmp_5 = static_cast<float*>(tmp_5_rev_[offset].get_data_ptr()); // need to save tmp_5 for backprop
							gpu_axpy(hidden_dim_, -1.0f, hc_t, hidden_state, tmp_5); // tmp4/tmp5 (((-1.0f * hc_t) + hidden_state[i])
							gpu_mul(hidden_dim_, 1.0f, z_t, tmp_5, 0.0f, hidden_state); // tmp6 = tmp5 * z_t
							gpu_axpy(hidden_dim_, 1.0f, hc_t, hidden_state, hidden_state); // h[i+1] = tmp6 + hc_t

							GPUToGPUCopy(hidden_rev_[offset + 1].get_data_ptr(), hidden_state, sizeof(float) * hidden_dim_);

							GPUToGPUCopy(static_cast<float*>(resultImpl->get_data_ptr()) + i * sequence_len * 2 * hidden_dim_ + 2 * hidden_dim_ * (sequence_len - j - 1) + hidden_dim_, hidden_state, sizeof(float) * hidden_dim_);
						}
#else
						LTEN_ERR("The USE_CUDA flag was not be set during the build (this flag must be set in order to use GPU tensors)");
#endif
					}
					else
					{
						LTEN_ERR("Invalid tensor device type");
					}
				}
			}


		}



		resultImpl->misc1_ = sequence_len;
		resultImpl->misc_ptr1_ = this;
		resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
		resultImpl->set_grad_fn(gru_backward);
		resultImpl->set_autograd(true);


		return Tensor(result);

	}


	void GRU::clear_gradients()
	{
		if (weights_u_ptr_)
		{
			weights_u_ptr_->clear_gradients();
		}
		if (weights_w_ptr_)
		{
			weights_w_ptr_->clear_gradients();
		}

		if (use_bias_)
		{
			if (bias_u_ptr_)
			{
				bias_u_ptr_->clear_gradients();
			}

			if (bias_w_ptr_)
			{
				bias_w_ptr_->clear_gradients();
			}
		}

		if (bidirectional_)
		{
			if (weights_u_rev_ptr_)
			{
				weights_u_rev_ptr_->clear_gradients();
			}
			if (weights_w_rev_ptr_)
			{
				weights_w_rev_ptr_->clear_gradients();
			}

			if (use_bias_)
			{
				if (bias_u_rev_ptr_)
				{
					bias_u_rev_ptr_->clear_gradients();
				}
				if (bias_w_rev_ptr_)
				{
					bias_w_rev_ptr_->clear_gradients();
				}
			}
		}
	}

	std::vector<Tensor*> GRU::get_all_weights()
	{
		std::vector<Tensor*> weights;

		weights.push_back(weights_u_ptr_);
		weights.push_back(weights_w_ptr_);

		if (use_bias_)
		{
			weights.push_back(bias_u_ptr_);
			weights.push_back(bias_w_ptr_);
		}

		if (bidirectional_)
		{
			weights.push_back(weights_u_rev_ptr_);
			weights.push_back(weights_w_rev_ptr_);

			if (use_bias_)
			{
				weights.push_back(bias_u_rev_ptr_);
				weights.push_back(bias_w_rev_ptr_);
			}
		}
		return weights;
	}

	void GRU::to(device target_device, int target_device_index)
	{
		if (weights_u_ptr_)
		{
			*weights_u_ptr_ = weights_u_ptr_->to(target_device, target_device_index);
		}

		if (weights_w_ptr_)
		{
			*weights_w_ptr_ = weights_w_ptr_->to(target_device, target_device_index);
		}

		if (bias_u_ptr_)
		{
			*bias_u_ptr_ = bias_u_ptr_->to(target_device, target_device_index);
		}

		if (bias_w_ptr_)
		{
			*bias_w_ptr_ = bias_w_ptr_->to(target_device, target_device_index);
		}


		if (weights_u_rev_ptr_)
		{
			*weights_u_rev_ptr_ = weights_u_rev_ptr_->to(target_device, target_device_index);
		}

		if (weights_w_rev_ptr_)
		{
			*weights_w_rev_ptr_ = weights_w_rev_ptr_->to(target_device, target_device_index);
		}

		if (bias_u_rev_ptr_)
		{
			*bias_u_rev_ptr_ = bias_u_rev_ptr_->to(target_device, target_device_index);
		}

		if (bias_w_rev_ptr_)
		{
			*bias_w_rev_ptr_ = bias_w_rev_ptr_->to(target_device, target_device_index);
		}


		scratch_hidden_state_ = scratch_hidden_state_.to(target_device, target_device_index);
		matmul_0_ = matmul_0_.to(target_device, target_device_index);
		matmul_1_ = matmul_1_.to(target_device, target_device_index);

		for (int i = 0; i < 7; i++)
		{
			extra_workspace_[i] = extra_workspace_[i].to(target_device, target_device_index);
		}

	}

#ifdef USE_CUDA

	GRU_CUDNN::~GRU_CUDNN()
	{
		if (!initialized_)
		{
			return;
		}

		cudaFree(workspace_);
		cudaFree(reserveSpace_);
		cudaFree(w_);
		cudaFree(dw_);

		for (unsigned int i = 0; i < max_sequence_len_; i++)
		{
			cudnnDestroyTensorDescriptor(xDesc_[i]);
			cudnnDestroyTensorDescriptor(yDesc_[i]);
			cudnnDestroyTensorDescriptor(dxDesc_[i]);
			cudnnDestroyTensorDescriptor(dyDesc_[i]);
		}

		free(xDesc_);
		free(yDesc_);
		free(dxDesc_);
		free(dyDesc_);

		cudnnDestroyTensorDescriptor(hxDesc_);
		cudnnDestroyTensorDescriptor(cxDesc_);
		cudnnDestroyTensorDescriptor(hyDesc_);
		cudnnDestroyTensorDescriptor(cyDesc_);
		cudnnDestroyTensorDescriptor(dhxDesc_);
		cudnnDestroyTensorDescriptor(dcxDesc_);
		cudnnDestroyTensorDescriptor(dhyDesc_);
		cudnnDestroyTensorDescriptor(dcyDesc_);

		cudnnDestroyDropoutDescriptor(dropoutDesc_);
		cudnnDestroyRNNDescriptor(rnnDesc_);
		cudnnDestroyFilterDescriptor(wDesc_);
		cudnnDestroyFilterDescriptor(dwDesc_);


		for (int i = 0; i < (bidirectional_ ? 2 : 1) * num_linear_layers_; i++)
		{
			delete weights_[i];
			if (use_bias_)
			{
				delete bias_[i];
			}
		}

		delete weights_;
		if (use_bias_)
		{
			delete bias_;
		}
	}

	// copied from Nvidia cudnn RNN sample (cudnn_samples_v6) with a few modifications
	bool GRU_CUDNN::init()
	{
		int dimIn[3];
		int dimOut[3];
		int dimHidden[3];
		int strideIn[3];
		int strideOut[3];
		int strideHidden[3];
		int numLayers;
		int i;
		float* initial_Weights;
		cudnnHandle_t cudnnHandle;

		numLayers = 1;

		dimIn[0] = static_cast<int>(max_batch_size_);
		dimIn[1] = static_cast<int>(input_dim_);
		dimIn[2] = 1;

		dimOut[0] = static_cast<int>(max_batch_size_);
		dimOut[1] = static_cast<int>(hidden_dim_ * (bidirectional_ ? 2 : 1));
		dimOut[2] = 1;

		dimHidden[0] = static_cast<int>(numLayers * bidirectional_ ? 2 : 1);
		dimHidden[1] = static_cast<int>(max_batch_size_);
		dimHidden[2] = static_cast<int>(hidden_dim_);

		strideIn[0] = dimIn[1] * dimIn[2];
		strideIn[1] = dimIn[2];
		strideIn[2] = 1;

		strideOut[0] = dimOut[1] * dimOut[2];
		strideOut[1] = dimOut[2];
		strideOut[2] = 1;

		strideHidden[0] = dimHidden[1] * dimHidden[2];
		strideHidden[1] = dimHidden[2];
		strideHidden[2] = 1;

		int hiddenTensorSize = dimHidden[0] * dimHidden[1] * dimHidden[2];
		int outputTensorSize = dimOut[0] * dimOut[1] * dimOut[2];

		xDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));
		yDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));
		dxDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));
		dyDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));


		for (unsigned int i = 0; i < max_sequence_len_; i++)
		{
			cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc_[i]));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc_[i]));

			cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc_[i]));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc_[i]));

			cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc_[i], CUDNN_DATA_FLOAT, 3, dimIn, strideIn));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc_[i], CUDNN_DATA_FLOAT, 3, dimIn, strideIn));

			cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc_[i], CUDNN_DATA_FLOAT, 3, dimOut, strideOut));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc_[i], CUDNN_DATA_FLOAT, 3, dimOut, strideOut));
		}


		cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc_));

		cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc_));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc_));

		cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));

		cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));


		unsigned long long seed = 0;
		float dropout = 0;

		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc_));

		size_t stateSize;
		void *states;
		cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

		cudaErrCheck(cudaMalloc(&states, stateSize));

		cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc_, cudnnHandle, dropout, states, stateSize, seed));


		cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc_));

		cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnnHandle,
			rnnDesc_,
			static_cast<int>(hidden_dim_),
			1,
			dropoutDesc_,
			CUDNN_LINEAR_INPUT,
			bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
			CUDNN_GRU,
			CUDNN_RNN_ALGO_STANDARD,
			CUDNN_DATA_FLOAT));

		if (!use_bias_)
		{
			cudnnSetRNNBiasMode(rnnDesc_, CUDNN_RNN_NO_BIAS);
		}



		cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc_));
		cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc_));


		cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc_, xDesc_[0], &weightsSize_, CUDNN_DATA_FLOAT));

		int dimW[3];
		dimW[0] = static_cast<int>(weightsSize_ / sizeof(float));
		dimW[1] = 1;
		dimW[2] = 1;

		cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
		cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

		cudaErrCheck(cudaMalloc((void **)&w_, weightsSize_));
		cudaErrCheck(cudaMalloc((void **)&dw_, weightsSize_));

		cudaErrCheck(cudaMemset(dw_, 0, weightsSize_));


		cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc_, max_sequence_len_, xDesc_, &workSize_));
		cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc_, max_sequence_len_, xDesc_, &reserveSize_));

		cudaErrCheck(cudaMalloc((void **)&workspace_, workSize_));
		cudaErrCheck(cudaMalloc((void **)&reserveSpace_, reserveSize_));


		weights_ = new Tensor*[(bidirectional_ ? 2 : 1) * num_linear_layers_];
		if (use_bias_)
		{
			bias_ = new Tensor*[(bidirectional_ ? 2 : 1) * num_linear_layers_];
		}

		int index = 0;
		TensorOps options;
		options.device_type = GPU;

		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-sqrtf(1.0f / hidden_dim_), sqrtf(1.0f / hidden_dim_));

		int debug_count = 0;

		for (int layer = 0; layer < numLayers * (bidirectional_ ? 2 : 1); layer++)
		{
			for (int linLayerID = 0; linLayerID < num_linear_layers_; linLayerID++)
			{
				cudnnDataType_t dataType;
				cudnnTensorFormat_t format;
				int nbDims;
				int filterDimA[3];
				uint64_t len;

				cudnnFilterDescriptor_t linLayerMatDesc;
				float *linLayerMat;

				//Initialize layer weights
				cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
				cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams(cudnnHandle,
					rnnDesc_,
					layer,
					xDesc_[0],
					wDesc_,
					w_,
					linLayerID,
					linLayerMatDesc,
					(void **)&linLayerMat));


				assert(linLayerMat);
				cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &format, &nbDims, filterDimA));

				len = filterDimA[0] * filterDimA[1] * filterDimA[2];
				assert(dataType == CUDNN_DATA_FLOAT);
				
				initial_Weights = new float[len];
				for (i = 0; i < len; i++)
				{
					initial_Weights[i] = distribution(generator);
					//initial_Weights[i] = 0.000362f;
				}
				CopyDataToGPU(linLayerMat, initial_Weights, sizeof(float) * len);
				delete initial_Weights;

				weights_[index] = new Tensor;
				*weights_[index] = TensorFromBuffer({ (uint64_t)filterDimA[0], (uint64_t)filterDimA[1], (uint64_t)filterDimA[2] }, linLayerMat, false, (float*)dw_ + ((float*)linLayerMat - (float*)w_), false, &options);

				cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));


				if (use_bias_)
				{
					//Initialize layer bias
					cudnnFilterDescriptor_t linLayerBiasDesc;
					float *linLayerBias;

					cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
					cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(cudnnHandle,
						rnnDesc_,
						layer,
						xDesc_[0],
						wDesc_,
						w_,
						linLayerID,
						linLayerBiasDesc,
						(void **)&linLayerBias));

					assert(linLayerBias);
					cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType, &format, &nbDims, filterDimA));

					len = filterDimA[0] * filterDimA[1] * filterDimA[2];
					assert(dataType == CUDNN_DATA_FLOAT);

					initial_Weights = new float[len];
					for (i = 0; i < len; i++)
					{
						initial_Weights[i] = distribution(generator);
						//initial_Weights[i] = 0.000362f;
					}
					CopyDataToGPU(linLayerBias, initial_Weights, sizeof(float) * len);
					delete initial_Weights;

					bias_[index] = new Tensor;
					*bias_[index] = TensorFromBuffer({ (uint64_t)filterDimA[0], (uint64_t)filterDimA[1], (uint64_t)filterDimA[2] }, linLayerBias, false, (float*)dw_ + ((float*)linLayerBias - (float*)w_), false, &options);

					cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
				}

				index++;
			}
		}

		initialized_ = true;
		return true;
	}


	Tensor GRU_CUDNN::forward(Tensor& input, Tensor& h0)
	{
		uint64_t sequence_len;
		unsigned int batches;
		TensorOps options;
		uint64_t dims[MAX_DIMS];
		int ndims;
		cudnnHandle_t cudnnHandle;

		ndims = input.get_ndims();

		if (input.get_device() != lten::GPU)
		{
			LTEN_ERR("GRU_CUDNN only supports the GPU device type");
		}

		memcpy(dims, input.get_sizes(), sizeof(uint64_t) * ndims);

		if (dims[ndims - 1] != input_dim_)
		{
			LTEN_ERR("Last dimension must be equal to the size of the input");
		}
		dims[ndims - 1] = hidden_dim_;


		sequence_len = input.get_sizes()[0];
		batches = static_cast<unsigned int>(input.get_sizes()[input.get_ndims() - 2]);


		TensorImpl<float>* resultImpl;
		resultImpl = new TensorImpl<float>;
		intrusive_ptr<TensorImplBase> result(resultImpl);

		options.data_type = input.get_data_type();
		options.device_index = input.get_device_index();
		options.device_type = input.get_device();

		if (bidirectional_)
		{
			dims[ndims - 1] *= 2;
		}
		resultImpl->allocate(dims, ndims, &options);



		cudnnHandle = CUDA_globlas::singleton()->get_cudnn_handle(0);

		if (batches != max_batch_size_ || sequence_len > max_sequence_len_)
		{
			{
				cudaFree(workspace_);
				cudaFree(reserveSpace_);

				for (unsigned int i = 0; i < max_sequence_len_; i++)
				{
					cudnnDestroyTensorDescriptor(xDesc_[i]);
					cudnnDestroyTensorDescriptor(yDesc_[i]);
					cudnnDestroyTensorDescriptor(dxDesc_[i]);
					cudnnDestroyTensorDescriptor(dyDesc_[i]);
				}

				free(xDesc_);
				free(yDesc_);
				free(dxDesc_);
				free(dyDesc_);

				cudnnDestroyTensorDescriptor(hxDesc_);
				cudnnDestroyTensorDescriptor(cxDesc_);
				cudnnDestroyTensorDescriptor(hyDesc_);
				cudnnDestroyTensorDescriptor(cyDesc_);
				cudnnDestroyTensorDescriptor(dhxDesc_);
				cudnnDestroyTensorDescriptor(dcxDesc_);
				cudnnDestroyTensorDescriptor(dhyDesc_);
				cudnnDestroyTensorDescriptor(dcyDesc_);

			}


			int dimIn[3];
			int dimOut[3];
			int dimHidden[3];
			int strideIn[3];
			int strideOut[3];
			int strideHidden[3];
			int numLayers;

			numLayers = 1;
			max_batch_size_ = batches;
			max_sequence_len_ = static_cast<unsigned int>(sequence_len);

			dimIn[0] = max_batch_size_;
			dimIn[1] = static_cast<int>(input_dim_);
			dimIn[2] = 1;

			dimOut[0] = max_batch_size_;
			dimOut[1] = static_cast<int>(hidden_dim_ * (bidirectional_ ? 2 : 1));
			dimOut[2] = 1;

			dimHidden[0] = numLayers * bidirectional_ ? 2 : 1;
			dimHidden[1] = max_batch_size_;
			dimHidden[2] = static_cast<int>(hidden_dim_);

			strideIn[0] = dimIn[1] * dimIn[2];
			strideIn[1] = dimIn[2];
			strideIn[2] = 1;

			strideOut[0] = dimOut[1] * dimOut[2];
			strideOut[1] = dimOut[2];
			strideOut[2] = 1;

			strideHidden[0] = dimHidden[1] * dimHidden[2];
			strideHidden[1] = dimHidden[2];
			strideHidden[2] = 1;

			int hiddenTensorSize = dimHidden[0] * dimHidden[1] * dimHidden[2];
			int outputTensorSize = dimOut[0] * dimOut[1] * dimOut[2];

			xDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));
			yDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));
			dxDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));
			dyDesc_ = (cudnnTensorDescriptor_t *)malloc(max_sequence_len_ * sizeof(cudnnTensorDescriptor_t));


			for (unsigned int i = 0; i < max_sequence_len_; i++)
			{
				cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc_[i]));
				cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc_[i]));

				cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc_[i]));
				cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc_[i]));

				cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc_[i], CUDNN_DATA_FLOAT, 3, dimIn, strideIn));
				cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc_[i], CUDNN_DATA_FLOAT, 3, dimIn, strideIn));

				cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc_[i], CUDNN_DATA_FLOAT, 3, dimOut, strideOut));
				cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc_[i], CUDNN_DATA_FLOAT, 3, dimOut, strideOut));
			}

			cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc_));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc_));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc_));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc_));

			cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc_));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc_));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc_));
			cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc_));

			cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));

			cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc_, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden));



			cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc_, max_sequence_len_, xDesc_, &workSize_));
			cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc_, max_sequence_len_, xDesc_, &reserveSize_));

			cudaErrCheck(cudaMalloc((void **)&workspace_, workSize_));
			cudaErrCheck(cudaMalloc((void **)&reserveSpace_, reserveSize_));

		}


		if (is_training_)
		{
			cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle,
				rnnDesc_,
				static_cast<int>(sequence_len),
				xDesc_,
				input.get_data_ptr(), // x
				hxDesc_,
				(&h0 == lten::MISC_globals::singleton()->get_null_tensor()) ? nullptr : h0.get_data_ptr(), // call cudnn with nullptr if no h0 passed in
				cxDesc_,
				nullptr, //cx
				wDesc_,
				w_,
				yDesc_,
				resultImpl->get_data_ptr(), //y
				hyDesc_,
				nullptr, //hy
				cyDesc_,
				nullptr, //cy
				workspace_,
				workSize_,
				reserveSpace_,
				reserveSize_));


			h0_ = &h0;
			resultImpl->misc1_ = sequence_len;
			resultImpl->misc_ptr1_ = this;
			resultImpl->add_child(*(static_cast<TensorImpl<float>*>(input.get_smart_ptr().get_real_object())));
			resultImpl->set_grad_fn(gru_cudnn_backward);
			resultImpl->set_autograd(true);
		}
		else
		{
			cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle,
				rnnDesc_,
				static_cast<int>(sequence_len),
				xDesc_,
				input.get_data_ptr(), // x
				hxDesc_,
				(&h0 == lten::MISC_globals::singleton()->get_null_tensor()) ? nullptr : h0.get_data_ptr(), // call cudnn with nullptr if no h0 passed in
				cxDesc_,
				nullptr, // cx
				wDesc_,
				w_,
				yDesc_,
				resultImpl->get_data_ptr(), //y
				hyDesc_,
				nullptr, // hy
				cyDesc_,
				nullptr, // cy
				workspace_,
				workSize_));
		}

		return Tensor(result);

	}


	std::vector<Tensor*> GRU_CUDNN::get_all_weights()
	{
		int numLayers = 1;
		int weight_index = 0;
		int bias_index = 0;
		std::vector<Tensor*> weights;

		for (int layer = 0; layer < numLayers * (bidirectional_ ? 2 : 1); layer++)
		{
			for (int linLayerID = 0; linLayerID < 6; linLayerID++)
			{
				weights.push_back(weights_[weight_index]);
				weight_index++;
			}

			if (use_bias_)
			{
				for (int linLayerID = 0; linLayerID < 6; linLayerID++)
				{
					weights.push_back(bias_[bias_index]);
					bias_index++;
				}
			}
		}

		return weights;
	}

	void GRU_CUDNN::clear_gradients()
	{
		if (dw_)
		{
			cudaErrCheck(cudaMemset(dw_, 0, weightsSize_));
		}
	}
#endif

}