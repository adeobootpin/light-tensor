#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#ifdef USE_CUDA
#include <curand.h>
#endif

#include "tensor.h"
#include "math_fns.h"


namespace lten {
	class Module
	{
	public:
		Module() {}
		virtual ~Module() {}

		virtual bool init() = 0;
		virtual std::vector<Tensor*> get_all_weights() = 0;
		virtual void clear_gradients() = 0;
		virtual void to(device target_device, int target_device_index = 0) = 0;
		void train(bool is_training) { is_training_ = is_training; }
		
		virtual void set_qparams_in(QuantizationParams qparams_in) {}
		virtual void set_qparams_out(QuantizationParams qparams_out) {}
		virtual void set_qparams_params(QuantizationParams* qparams, int count) {}
	protected:
		bool is_training_ = true;
	};



	class FullyConnected : public Module
	{
	public:
		FullyConnected(int64_t input_features, int64_t output_features, bool use_bias = false, dtype data_type = FLOAT32) : input_features_(input_features), output_features_(output_features), weight_ptr_(nullptr), bias_ptr_(nullptr), bias_multiplier_(nullptr), use_bias_(use_bias) {}
		~FullyConnected()
		{
			delete weight_ptr_;
			delete bias_ptr_;
			delete bias_multiplier_;
		}

		bool init();
		Tensor forward(Tensor& input);
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		Tensor* get_bias_multiplier() { return bias_multiplier_; }
		void clear_gradients();
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index = 0);
		bool is_using_bias() { return use_bias_; }


	private:
		uint64_t input_features_;
		uint64_t output_features_;
		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		Tensor* bias_multiplier_;
		bool use_bias_;
		uint64_t max_bias_multiplier_size_ = 0;

	};


	class FullyConnected_q : public Module // quantized FullyConnected
	{
	public:
		FullyConnected_q(int64_t input_features, int64_t output_features, bool use_bias = false, dtype data_type = FLOAT32) : input_features_(input_features), output_features_(output_features), weight_ptr_(nullptr), bias_ptr_(nullptr), workspace_(nullptr), use_bias_(use_bias) {}
		~FullyConnected_q()
		{
			delete weight_ptr_;
			delete bias_ptr_;
			delete workspace_;
		}

		bool init();
		Tensor forward(Tensor& input);
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		void clear_gradients();
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index = 0) { LTEN_ERR("Only CPU device supported"); } // CPU only for now
		bool is_using_bias() { return use_bias_; }

		void set_qparams_in(QuantizationParams qparams_in) { qparams_in_ = qparams_in; }
		void set_qparams_out(QuantizationParams qparams_out) { qparams_out_ = qparams_out; }
		void set_qparams_params(QuantizationParams* qparams, int count);


	private:
		uint64_t input_features_;
		uint64_t output_features_;
		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		Tensor* workspace_;
		bool use_bias_;
		uint64_t max_batch_size_ = 0;
		QuantizationParams qparams_in_;
		QuantizationParams qparams_out_;
		QuantizationParams qparams_wt_;

	};


	class GRU : public Module
	{
	public:
		GRU(const uint64_t input_dim, const uint64_t hidden_dim, bool use_bias, bool bidirectional)
		{
			input_dim_ = input_dim;
			hidden_dim_ = hidden_dim;
			use_bias_ = use_bias;
			bidirectional_ = bidirectional;

			weights_u_ptr_ = nullptr;
			weights_w_ptr_ = nullptr;
			bias_u_ptr_ = nullptr;
			bias_w_ptr_ = nullptr;

			weights_u_rev_ptr_ = nullptr;
			weights_w_rev_ptr_ = nullptr;
			bias_u_rev_ptr_ = nullptr;
			bias_w_rev_ptr_ = nullptr;
		}

		~GRU()
		{
		}

		bool init();
		Tensor forward(Tensor& input);
		virtual std::vector<Tensor*> get_all_weights();
		Tensor* get_weights() { return nullptr; }
		Tensor* get_bias() { return nullptr; }

		void clear_gradients();
		bool is_bidirectional() { return bidirectional_; }
		bool is_using_bias() { return use_bias_; }

		Tensor* get_u_weights() { return weights_u_ptr_; }
		Tensor* get_w_weights() { return weights_w_ptr_; }

		Tensor* get_u_bias() { return bias_u_ptr_; }
		Tensor* get_w_bias() { return bias_w_ptr_; }

		Tensor* get_u_rev_weights() { return weights_u_rev_ptr_; }
		Tensor* get_w_rev_weights() { return weights_w_rev_ptr_; }

		Tensor* get_u_rev_bias() { return bias_u_rev_ptr_; }
		Tensor* get_w_rev_bias() { return bias_w_rev_ptr_; }


		Tensor* get_hidden_array() { return hidden_; }
		Tensor* get_hidden_rev_array() { return hidden_rev_; }

		Tensor* get_z_t_array() { return z_t_; }
		Tensor* get_hc_t_array() { return hc_t_; }
		Tensor* get_w_hc_h_array() { return w_hc_h_; }
		Tensor* get_r_t_array() { return r_t_; }
		Tensor* get_tmp_5_array() { return tmp_5_; }

		Tensor* get_z_t_rev_array() { return z_t_rev_; }
		Tensor* get_hc_t_rev_array() { return hc_t_rev_; }
		Tensor* get_w_hc_t_rev_array() { return w_hc_h_rev_; }
		Tensor* get_r_t_rev_array() { return r_t_rev_; }
		Tensor* get_tmp_5_rev_array() { return tmp_5_rev_; }
		Tensor* get_extra_workspace() { return extra_workspace_; }

		uint64_t get_hidden_dim() { return hidden_dim_; }
		uint64_t get_input_dim() { return input_dim_; }
		void to(device target_device, int target_device_index = 0);
	private:

		uint64_t input_dim_;
		uint64_t hidden_dim_;
		bool use_bias_;
		bool bidirectional_;


		Tensor* weights_u_ptr_;
		Tensor* weights_w_ptr_;
		Tensor* bias_u_ptr_;
		Tensor* bias_w_ptr_;

		Tensor* weights_u_rev_ptr_;
		Tensor* weights_w_rev_ptr_;
		Tensor* bias_u_rev_ptr_;
		Tensor* bias_w_rev_ptr_;


		Tensor hidden_[1024 * 10]; // TODO: change to dynamically allocated arrays
		Tensor hidden_rev_[1024 * 10];

		Tensor tmp_5_[512 / 2 * 10];
		Tensor hc_t_[512 / 2 * 10];
		Tensor z_t_[512 / 2 * 10];
		Tensor r_t_[512 / 2 * 10];
		Tensor w_hc_h_[512 / 2 * 10];

		Tensor tmp_5_rev_[512 / 2 * 10];
		Tensor hc_t_rev_[512 / 2 * 10];
		Tensor z_t_rev_[512 / 2 * 10];
		Tensor r_t_rev_[512 / 2 * 10];
		Tensor w_hc_h_rev_[512 / 2 * 10];


		Tensor scratch_hidden_state_;
		Tensor matmul_0_;
		Tensor matmul_1_;
		Tensor hc_t_x_;


		Tensor extra_workspace_[7]; // scratch memory for backpopagation

	};

#ifdef USE_CUDA
	class GRU_CUDNN : public Module
	{
	public:
		GRU_CUDNN(const uint64_t input_dim, const uint64_t hidden_dim, bool use_bias, bool bidirectional, int expected_batch_size = 0, int expected_max_sequence_len = 0)
		{
			input_dim_ = input_dim;
			hidden_dim_ = hidden_dim;
			use_bias_ = use_bias;
			bidirectional_ = bidirectional;

			max_sequence_len_ = std::max(1, expected_max_sequence_len);
			max_batch_size_ = std::max(1, expected_batch_size);

			weights_ = nullptr;
			bias_ = nullptr;
			initialized_ = false;
		}

		~GRU_CUDNN();

		bool init();
		Tensor forward(Tensor& input, Tensor& h0 = *(lten::MISC_globals::singleton()->get_null_tensor()));
		std::vector<Tensor*> get_all_weights();
		void clear_gradients();
		void to(device target_device, int target_device_index = 0) {}

		cudnnRNNDescriptor_t* get_rnnDesc() { return &rnnDesc_; }
		cudnnTensorDescriptor_t* get_xDesc() { return xDesc_; }
		cudnnTensorDescriptor_t* get_yDesc() { return yDesc_; }
		cudnnTensorDescriptor_t* get_dyDesc() { return dyDesc_; }
		cudnnTensorDescriptor_t* get_dxDesc() { return dxDesc_; }
		cudnnTensorDescriptor_t* get_hyDesc() { return &dhyDesc_; }
		cudnnTensorDescriptor_t* get_dcyDesc() { return &dcyDesc_; }
		cudnnFilterDescriptor_t* get_wDesc() { return &wDesc_; }
		cudnnFilterDescriptor_t* get_dwDesc() { return &dwDesc_; }

		cudnnTensorDescriptor_t* get_hxDesc() { return &hxDesc_; }
		cudnnTensorDescriptor_t* get_cxDesc() { return &cxDesc_; }

		cudnnTensorDescriptor_t* get_dhxDesc() { return &dhxDesc_; }
		cudnnTensorDescriptor_t* get_dcxDesc() { return &dcxDesc_; }

		Tensor* get_h0() { return h0_; }

		void* get_w() { return w_; }
		void* get_dw() { return dw_; }
		void* get_workspace() { return workspace_; }
		void* get_reserveSpace() { return reserveSpace_; }


		size_t get_workSize() { return workSize_; }
		size_t get_reserveSize() { return reserveSize_; }

	private:
		bool initialized_;
		uint64_t input_dim_;
		uint64_t hidden_dim_;
		bool use_bias_;
		bool bidirectional_;

		Tensor** weights_;
		Tensor** bias_;

		unsigned int max_sequence_len_;
		unsigned int max_batch_size_;

		cudnnTensorDescriptor_t* xDesc_;
		cudnnTensorDescriptor_t* yDesc_;

		cudnnTensorDescriptor_t* dxDesc_;
		cudnnTensorDescriptor_t* dyDesc_;

		cudnnTensorDescriptor_t hxDesc_;
		cudnnTensorDescriptor_t cxDesc_;
		cudnnTensorDescriptor_t hyDesc_;
		cudnnTensorDescriptor_t cyDesc_;
		cudnnTensorDescriptor_t dhxDesc_;
		cudnnTensorDescriptor_t dcxDesc_;
		cudnnTensorDescriptor_t dhyDesc_;
		cudnnTensorDescriptor_t dcyDesc_;

		cudnnFilterDescriptor_t wDesc_;
		cudnnFilterDescriptor_t dwDesc_;

		void *w_;
		void *dw_;
		size_t weightsSize_;

		void *workspace_;
		void *reserveSpace_;
		size_t workSize_;
		size_t reserveSize_;

		cudnnDropoutDescriptor_t dropoutDesc_;
		cudnnRNNDescriptor_t rnnDesc_;
		enum { num_linear_layers_ = 6 };

		Tensor* h0_;
	};
#endif


	class Dropout : public Module
	{
	public:
		Dropout(float probability) : probability_(probability), mask_(nullptr)
		{
			distribution_ = new std::uniform_int_distribution<unsigned int>(0);
		}
		~Dropout()
		{
			delete mask_;
		}

		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients() {}
		std::vector<Tensor*> get_all_weights() { std::vector<Tensor*> dud; return dud; }
		void to(device target_device, int target_device_index = 0);

		MultiDimArray<unsigned int>* get_mask() { return mask_; }
		float get_scale() { return scale_; }
		unsigned int get_threshold() { return threshold_; }
	private:
		std::default_random_engine generator_;
		std::uniform_int_distribution<unsigned int>* distribution_;

#ifdef USE_CUDA
		curandGenerator_t cuda_generator_;
#endif

		float probability_;
		float scale_;
		unsigned int threshold_;
		MultiDimArray<unsigned int>* mask_;
	};

#ifdef USE_CUDA
	class softmax_CUDNN : public Module
	{
	public:
		softmax_CUDNN(bool log_mode = false) 
		{
			if (log_mode)
			{
				algo_ = CUDNN_SOFTMAX_LOG;
			}
			else
			{
				algo_ = CUDNN_SOFTMAX_ACCURATE;
			}
		}
		~softmax_CUDNN() {}
		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients() {}
		std::vector<Tensor*> get_all_weights() { std::vector<Tensor*> dud; return dud; }
		void to(device target_device, int target_device_index) {}
		
		cudnnSoftmaxAlgorithm_t get_algo() { return algo_; }
		cudnnTensorDescriptor_t* get_inputDesc() { return &inputDesc_; }
		cudnnTensorDescriptor_t* get_outputDesc() { return &outputDesc_; }

	private:
		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t outputDesc_;
		cudnnSoftmaxAlgorithm_t algo_;
	};
#endif

	class Conv2d : public Module
	{
	public:
		Conv2d(const int channels_in, const int channels_out, bool use_bias, const int kernel_h, const int kernel_w, const int pad_h = 0, const int pad_w = 0, const int stride_h = 1, const int stride_w = 1)
		{
			channels_in_ = channels_in;
			channels_out_ = channels_out;
			kernel_h_ = kernel_h;
			kernel_w_ = kernel_w;
			pad_h_ = pad_h;
			pad_w_ = pad_w;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
			weight_ptr_ = nullptr;
			bias_ptr_ = nullptr;
			col_buffer_ptr_ = nullptr;

			use_bias_ = use_bias;
		}

		~Conv2d()
		{
			delete weight_ptr_;
			delete bias_ptr_;
			delete col_buffer_ptr_;
		}

		uint64_t get_channels_out() { return channels_out_; }
		uint64_t get_channels_in() { return channels_in_; }
		uint64_t get_kernel_h() { return kernel_h_; }
		uint64_t get_kernel_w() { return kernel_w_; }
		uint64_t get_pad_h() { return pad_h_; }
		uint64_t get_pad_w() { return pad_w_; }
		uint64_t get_stride_h() { return stride_h_; }
		uint64_t get_stride_w() { return stride_w_; }
		Tensor* get_col_buffer_ptr() { return col_buffer_ptr_; }

		bool init();
		Tensor forward(Tensor& input);
		virtual std::vector<Tensor*> get_all_weights();
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		void clear_gradients();
		void to(device target_device, int target_device_index = 0);

	private:
		uint64_t channels_in_;
		uint64_t channels_out_;
		uint64_t kernel_h_;
		uint64_t kernel_w_;
		uint64_t pad_h_;
		uint64_t pad_w_;
		uint64_t stride_h_;
		uint64_t stride_w_;
		bool use_bias_;

		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		Tensor* col_buffer_ptr_;
	};

	class Conv2d_q : public Module // quantized Conv2d
	{
	public:
		Conv2d_q(const int channels_in, const int channels_out, bool use_bias, const int kernel_h, const int kernel_w, const int pad_h = 0, const int pad_w = 0, const int stride_h = 1, const int stride_w = 1)
		{
			channels_in_ = channels_in;
			channels_out_ = channels_out;
			kernel_h_ = kernel_h;
			kernel_w_ = kernel_w;
			pad_h_ = pad_h;
			pad_w_ = pad_w;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
			weight_ptr_ = nullptr;
			bias_ptr_ = nullptr;
			col_buffer_ptr_ = nullptr;
			workspace_ = nullptr;

			use_bias_ = use_bias;
		}

		~Conv2d_q()
		{
			delete weight_ptr_;
			delete bias_ptr_;
			delete col_buffer_ptr_;
			delete workspace_;
		}

		uint64_t get_channels_out() { return channels_out_; }
		uint64_t get_channels_in() { return channels_in_; }
		uint64_t get_kernel_h() { return kernel_h_; }
		uint64_t get_kernel_w() { return kernel_w_; }
		uint64_t get_pad_h() { return pad_h_; }
		uint64_t get_pad_w() { return pad_w_; }
		uint64_t get_stride_h() { return stride_h_; }
		uint64_t get_stride_w() { return stride_w_; }
		Tensor* get_col_buffer_ptr() { return col_buffer_ptr_; }

		bool init();
		Tensor forward(Tensor& input);
		virtual std::vector<Tensor*> get_all_weights();
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		void clear_gradients();
		void to(device target_device, int target_device_index = 0) { LTEN_ERR("Only CPU device supported"); } // CPU only for now

		void set_qparams_in(QuantizationParams qparams_in) { qparams_in_ = qparams_in; }
		void set_qparams_out(QuantizationParams qparams_out) { qparams_out_ = qparams_out; }
		void set_qparams_params(QuantizationParams* qparams, int count);

	private:
		uint64_t channels_in_;
		uint64_t channels_out_;
		uint64_t kernel_h_;
		uint64_t kernel_w_;
		uint64_t pad_h_;
		uint64_t pad_w_;
		uint64_t stride_h_;
		uint64_t stride_w_;
		bool use_bias_;
		uint64_t max_input_size_ = 0;

		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		Tensor* col_buffer_ptr_;
		Tensor* workspace_;

		QuantizationParams qparams_in_;
		QuantizationParams qparams_out_;
		QuantizationParams qparams_wt_;
	};

#ifdef USE_CUDA
	class conv_CUDNN : public Module
	{
	public:
		conv_CUDNN(const int expected_batch_size, const int channels_in, const int channels_out, const int ndims, const int* dims, const int* kernel, const int* padding, const int* stride, bool use_bias )
		{
			dims_ = new int[ndims];
			kernel_ = new int[ndims];
			padding_ = new int[ndims];
			stride_ = new int[ndims];
			output_dims_ = new uint64_t[ndims + 2]; //NCHW, NCDHW ...

			memcpy(dims_, dims, sizeof(int) * ndims);
			memcpy(kernel_, kernel, sizeof(int) * ndims);
			memcpy(padding_, padding, sizeof(int) * ndims);
			memcpy(stride_, stride, sizeof(int) * ndims);

			channels_in_ = channels_in;
			channels_out_ = channels_out;

			use_bias_ = use_bias;
			batch_size_ = expected_batch_size;
			ndims_ = ndims;
		}

		~conv_CUDNN();

		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients();
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index) {}
		bool is_using_bias() { return use_bias_; }

		cudnnTensorDescriptor_t get_inputDesc() { return inputDesc_; }
		cudnnTensorDescriptor_t get_outputDesc() { return outputDesc_; }
		cudnnConvolutionDescriptor_t get_convDesc() { return convDesc_; }
		cudnnFilterDescriptor_t get_wtDesc() { return wtDesc_; }
		cudnnTensorDescriptor_t get_biasDesc() { return biasDesc_; }
		cudnnConvolutionBwdFilterAlgo_t get_bwf_algo() { return bwf_algo_; }
		cudnnConvolutionBwdDataAlgo_t get_bwd_algo() { return bwd_algo_; }
		void* get_workspace() { return workspace_; }
		size_t get_workspace_size() { return workspace_size_; }

		void* get_bwf_workspace() { return bwf_workspace_; }
		size_t get_bwf_workspace_size() { return bwf_workspace_size_; }

		void* get_bwd_workspace() { return bwd_workspace_; }
		size_t get_bwd_workspace_size() { return bwd_workspace_size_; }

	private:
		uint32_t batch_size_;		
		uint32_t channels_in_;
		uint32_t channels_out_;
		int* dims_;
		int* kernel_;
		int* padding_;
		int* stride_;
		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		bool use_bias_;
		uint32_t ndims_;
		uint64_t* output_dims_;

		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t outputDesc_;
		cudnnConvolutionDescriptor_t convDesc_;
		cudnnFilterDescriptor_t wtDesc_;
		cudnnTensorDescriptor_t biasDesc_;
		cudnnConvolutionFwdAlgo_t algo_;
		cudnnConvolutionBwdFilterAlgo_t bwf_algo_;
		cudnnConvolutionBwdDataAlgo_t bwd_algo_;

		void* workspace_;
		size_t workspace_size_;

		void* bwf_workspace_;
		size_t bwf_workspace_size_;

		void* bwd_workspace_;
		size_t bwd_workspace_size_;

	};

	class conv2d_CUDNN : public Module
	{
	public:
		conv2d_CUDNN(const int expected_batch_size, const int channels_in, const int height_in, const int width_in, const int channels_out, bool use_bias, const int kernel_h, const int kernel_w, const int pad_h = 0, const int pad_w = 0, const int stride_h = 1, const int stride_w = 1)
		{
			channels_in_ = channels_in;
			channels_out_ = channels_out;
			kernel_h_ = kernel_h;
			kernel_w_ = kernel_w;
			pad_h_ = pad_h;
			pad_w_ = pad_w;
			stride_h_ = stride_h;
			stride_w_ = stride_w;

			use_bias_ = use_bias;
			batch_size_ = expected_batch_size;
			height_in_ = height_in;
			width_in_ = width_in;
		}

		~conv2d_CUDNN() {}
		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients();
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index) {} 
		bool is_using_bias() { return use_bias_; }

		cudnnTensorDescriptor_t get_inputDesc() { return inputDesc_; }
		cudnnTensorDescriptor_t get_outputDesc() { return outputDesc_; }
		cudnnConvolutionDescriptor_t get_convDesc() { return convDesc_; }
		cudnnFilterDescriptor_t get_wtDesc() { return wtDesc_; }
		cudnnTensorDescriptor_t get_biasDesc() { return biasDesc_; }
		cudnnConvolutionBwdFilterAlgo_t get_bwf_algo() { return bwf_algo_; }
		cudnnConvolutionBwdDataAlgo_t get_bwd_algo() { return bwd_algo_; }
		void* get_workspace() { return workspace_; }
		size_t get_workspace_size() { return workspace_size_; }

		void* get_bwf_workspace() { return bwf_workspace_; }
		size_t get_bwf_workspace_size() { return bwf_workspace_size_; }

		void* get_bwd_workspace() { return bwd_workspace_; }
		size_t get_bwd_workspace_size() { return bwd_workspace_size_; }



	private:
		uint64_t batch_size_;
		uint64_t height_in_;
		uint64_t width_in_;
		uint64_t channels_in_;
		uint64_t channels_out_;
		uint64_t kernel_h_;
		uint64_t kernel_w_;
		uint64_t pad_h_;
		uint64_t pad_w_;
		uint64_t stride_h_;
		uint64_t stride_w_;
		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		bool use_bias_;
		uint64_t output_dims_[4]; //NCHW

		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t outputDesc_;
		cudnnConvolutionDescriptor_t convDesc_;
		cudnnFilterDescriptor_t wtDesc_;
		cudnnTensorDescriptor_t biasDesc_;
		cudnnConvolutionFwdAlgo_t algo_;
		cudnnConvolutionBwdFilterAlgo_t bwf_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
		cudnnConvolutionBwdDataAlgo_t bwd_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

		void* workspace_;
		size_t workspace_size_;

		void* bwf_workspace_;
		size_t bwf_workspace_size_;

		void* bwd_workspace_;
		size_t bwd_workspace_size_;
	};



	class conv3d_CUDNN : public Module
	{
		enum { conv_dims = 3};
	public:
		conv3d_CUDNN(const int expected_batch_size, const int channels_in, const int depth_in, const int height_in, const int width_in, const int channels_out, bool use_bias, const int kernel_h, const int kernel_w, const int kernel_c, const int pad_h = 0, const int pad_w = 0, const int pad_c = 0, const int stride_h = 1, const int stride_w = 1, const int stride_c = 1)
		{
			channels_in_ = channels_in;
			channels_out_ = channels_out;
			kernel_h_ = kernel_h;
			kernel_w_ = kernel_w;
			kernel_c_ = kernel_c;
			pad_h_ = pad_h;
			pad_w_ = pad_w;
			pad_c_ = pad_c;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
			stride_c_ = stride_c;

			use_bias_ = use_bias;
			batch_size_ = expected_batch_size;
			depth_in_ = depth_in;
			height_in_ = height_in;
			width_in_ = width_in;
		}

		~conv3d_CUDNN();
		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients();
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index) {}
		bool is_using_bias() { return use_bias_; }

		cudnnTensorDescriptor_t get_inputDesc() { return inputDesc_; }
		cudnnTensorDescriptor_t get_outputDesc() { return outputDesc_; }
		cudnnConvolutionDescriptor_t get_convDesc() { return convDesc_; }
		cudnnFilterDescriptor_t get_wtDesc() { return wtDesc_; }
		cudnnTensorDescriptor_t get_biasDesc() { return biasDesc_; }
		cudnnConvolutionBwdFilterAlgo_t get_bwf_algo() { return bwf_algo_; }
		cudnnConvolutionBwdDataAlgo_t get_bwd_algo() { return bwd_algo_; }
		void* get_workspace() { return workspace_; }
		size_t get_workspace_size() { return workspace_size_; }

		void* get_bwf_workspace() { return bwf_workspace_; }
		size_t get_bwf_workspace_size() { return bwf_workspace_size_; }

		void* get_bwd_workspace() { return bwd_workspace_; }
		size_t get_bwd_workspace_size() { return bwd_workspace_size_; }



	private:
		uint32_t batch_size_;
		uint32_t height_in_;
		uint32_t width_in_;
		uint32_t channels_in_;
		uint32_t channels_out_;
		uint32_t kernel_h_;
		uint32_t kernel_w_;
		uint32_t kernel_c_;
		uint32_t pad_h_;
		uint32_t pad_w_;
		uint32_t pad_c_;
		uint32_t depth_in_;
		uint32_t stride_h_;
		uint32_t stride_w_;
		uint32_t stride_c_;
		Tensor* weight_ptr_;
		Tensor* bias_ptr_;
		bool use_bias_;
		uint64_t output_dims_[conv_dims + 2]; //NCDHW

		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t outputDesc_;
		cudnnConvolutionDescriptor_t convDesc_;
		cudnnFilterDescriptor_t wtDesc_;
		cudnnTensorDescriptor_t biasDesc_;
		cudnnConvolutionFwdAlgo_t algo_;
		cudnnConvolutionBwdFilterAlgo_t bwf_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
		cudnnConvolutionBwdDataAlgo_t bwd_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

		void* workspace_;
		size_t workspace_size_;

		void* bwf_workspace_;
		size_t bwf_workspace_size_;

		void* bwd_workspace_;
		size_t bwd_workspace_size_;
	};
#endif


	class BatchNorm : public Module
	{
	public:
		BatchNorm(uint64_t num_features) : num_features_(num_features) {}
		~BatchNorm() {}

		bool init();
		Tensor forward(Tensor& input);
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		void clear_gradients();
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index = 0);


	private:
		uint64_t num_features_;
		Tensor* weight_ptr_; // gamma
		Tensor* bias_ptr_; // beta
		Tensor* mu_;
		Tensor* sigma_;
		Tensor* ones_vector_;
		float mo_ = 0.1f; // momentum for mu and sigma updates
		float epsilon_ = 1e-5f;
		uint64_t max_ones_vector_size_ = 0;
	};

	class LayerNorm : public Module
	{
	public:
		LayerNorm(uint64_t num_features, bool affine = true) : num_features_(num_features), affine_(affine) {}
		~LayerNorm() {}

		bool init();
		Tensor forward(Tensor& input);
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		Tensor* get_ln() { return &ln_; }
		Tensor* get_mu() { return &mu_; }
		Tensor* get_sd() { return &sd_; }
		Tensor* get_temp1() { return &temp1_; }
		Tensor* get_feeder_gradient() { return &feeder_gradient_; }
		void clear_gradients();
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index = 0);
		bool is_affine() { return affine_; }

	private:
		uint64_t num_features_;
		Tensor* weight_ptr_; // gamma
		Tensor* bias_ptr_; // beta
		Tensor eps_;
		Tensor ln_;
		Tensor sd_;
		Tensor feeder_gradient_; // gradient flowing back from gamma if affine, ignored othersize (simply use top_gradient)
		Tensor mu_;
		Tensor temp1_;
		Tensor temp2_;
		Tensor temp3_;
		float epsilon_ = 1e-5f;
		uint64_t max_ones_vector_size_ = 0;
		bool affine_;
		int naxes_;
		uint32_t axes_[MAX_DIMS];
	};

#ifdef USE_CUDA
	class BatchNorm_CUDNN : public Module
	{
	public:
		BatchNorm_CUDNN(uint64_t num_features) : num_features_(num_features), weight_ptr_(nullptr), bias_ptr_(nullptr), mu_(nullptr), sigma_(nullptr) {}
		~BatchNorm_CUDNN() 
		{
			delete weight_ptr_;
			delete bias_ptr_;
			delete mu_;
			delete sigma_;
		}

		bool init();
		Tensor forward(Tensor& input);
		Tensor* get_weights() { return weight_ptr_; }
		Tensor* get_bias() { return bias_ptr_; }
		void clear_gradients();
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index) {}

		cudnnTensorDescriptor_t get_inputDesc() { return inputDesc_; }
		cudnnTensorDescriptor_t get_scale_bias_Desc() { return scale_bias_mean_var_Desc_; }
		cudnnBatchNormMode_t get_mode() { return mode_; }
		double get_epsilon() { return epsilon_; }
	private:
		uint64_t num_features_;
		Tensor* weight_ptr_; // gamma
		Tensor* bias_ptr_; // beta
		Tensor* mu_;
		Tensor* sigma_;
		double  mo_; // momentum for mu and sigma updates
		double epsilon_;

		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t scale_bias_mean_var_Desc_;
		cudnnBatchNormMode_t mode_;

	};
#endif

#ifdef USE_CUDA
	class pooling_CUDNN : public Module
	{
	public:
		pooling_CUDNN(int mode, const int kernel_h, const int kernel_w, const int pad_h = 0, const int pad_w = 0, const int stride_h = 1, const int stride_w = 1)
		{
			mode_ = mode;
			kernel_h_ = kernel_h;
			kernel_w_ = kernel_w;
			pad_h_ = pad_h;
			pad_w_ = pad_w;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
		}

		~pooling_CUDNN() 
		{
			cudnnErrCheck(cudnnDestroyPoolingDescriptor(poolingDesc_));
			cudnnErrCheck(cudnnDestroyTensorDescriptor(inputDesc_));
			cudnnErrCheck(cudnnDestroyTensorDescriptor(outputDesc_));
		}

		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients() {}
		std::vector<Tensor*> get_all_weights() { std::vector<Tensor*> dud; return dud; }
		void to(device target_device, int target_device_index) {}


		cudnnPoolingDescriptor_t get_poolingDesc() { return poolingDesc_; }
		cudnnTensorDescriptor_t get_inputDesc() { return inputDesc_; }
		cudnnTensorDescriptor_t get_outputDesc() { return outputDesc_; }

	private:
		cudnnPoolingDescriptor_t poolingDesc_;
		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t outputDesc_;
		int mode_;
		int kernel_h_;
		int kernel_w_;
		int pad_h_;
		int pad_w_;
		int stride_h_;
		int stride_w_;
		int output_dims_[4]; //NCHW
	};

	class pooling3d_CUDNN : public Module
	{
	public:
		pooling3d_CUDNN(int mode, const int kernel_h, const int kernel_w, const int kernel_c, const int pad_h = 0, const int pad_w = 0, const int pad_c = 0, const int stride_h = 1, const int stride_w = 1, const int stride_c = 1)
		{
			mode_ = mode;
			kernel_h_ = kernel_h;
			kernel_w_ = kernel_w;
			kernel_c_ = kernel_c;
			pad_h_ = pad_h;
			pad_w_ = pad_w;
			pad_c_ = pad_c;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
			stride_c_ = stride_c;
		}

		~pooling3d_CUDNN()
		{
			cudnnErrCheck(cudnnDestroyPoolingDescriptor(poolingDesc_));
			cudnnErrCheck(cudnnDestroyTensorDescriptor(inputDesc_));
			cudnnErrCheck(cudnnDestroyTensorDescriptor(outputDesc_));
		}

		bool init();
		Tensor forward(Tensor& input);
		void clear_gradients() {}
		std::vector<Tensor*> get_all_weights() { std::vector<Tensor*> dud; return dud; }
		void to(device target_device, int target_device_index) {}


		cudnnPoolingDescriptor_t get_poolingDesc() { return poolingDesc_; }
		cudnnTensorDescriptor_t get_inputDesc() { return inputDesc_; }
		cudnnTensorDescriptor_t get_outputDesc() { return outputDesc_; }

	private:
		cudnnPoolingDescriptor_t poolingDesc_;
		cudnnTensorDescriptor_t inputDesc_;
		cudnnTensorDescriptor_t outputDesc_;
		int mode_;
		int kernel_h_;
		int kernel_w_;
		int kernel_c_;
		int pad_h_;
		int pad_w_;
		int pad_c_;
		int stride_h_;
		int stride_w_;
		int stride_c_;
		int output_dims_[5]; //NCDHW
	};
#endif


	class Embedding : public Module
	{
	public:
		Embedding(unsigned int num_embeddings, unsigned int embedding_dim) : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim), weight_ptr_(nullptr) {}
		~Embedding()
		{
			delete weight_ptr_;
		}

		bool init();
		Tensor forward(Tensor& input);
		Tensor* get_weights() { return weight_ptr_; }
		void clear_gradients();
		std::vector<Tensor*> get_all_weights();
		void to(device target_device, int target_device_index = 0);

	private:
		unsigned int num_embeddings_;
		unsigned int embedding_dim_;
		Tensor* weight_ptr_;
		Tensor input_indices_;
	};


	class Pseudo_Einsum_1 : public Module
	{
	public:
		Pseudo_Einsum_1()
		{
			pa_.buffer = nullptr;
			oa_.buffer = nullptr;

			ndims_ = 0;
			numels_ = 0;
		}

		~Pseudo_Einsum_1() {}

		bool init();
		Tensor forward(Tensor& A, Tensor& B);
		void clear_gradients() {}
		std::vector<Tensor*> get_all_weights() { std::vector<Tensor*> dud; return dud; }
		void to(device target_device, int target_device_index) {}

	private:
		int ndims_;
		uint64_t numels_;
		POINTER_ARRAYS pa_;
		OFFSET_ARRAYS oa_;
	};

	class Pseudo_Einsum_2 : public Module
	{
	public:
		Pseudo_Einsum_2()
		{
			scratch_buffer_ = nullptr;
		}

		~Pseudo_Einsum_2() {}

		bool init() { return true; }
		Tensor forward(Tensor& A, Tensor& B);
		void clear_gradients() {}
		std::vector<Tensor*> get_all_weights() { std::vector<Tensor*> dud; return dud; }
		void to(device target_device, int target_device_index) {}

	private:
		void* scratch_buffer_;

	};

	Tensor relu(Tensor& input);
	Tensor softmax(Tensor& input, int dim = 1);
	Tensor log_softmax(Tensor& input, int dim = 1);
	Tensor mse_loss(Tensor& input, Tensor& target);
	Tensor nll_loss(Tensor& input, Tensor& target);
} // namespace lten

#endif //LAYERS_H