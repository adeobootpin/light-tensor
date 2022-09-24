#ifndef TENSORIMPL_H
#define TENSORIMPL_H
/*
MIT License

Copyright (c) 2021 adeobootpin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
enum { MAX_DIMS = 16 };

#include <string.h>
#include "shared_pointer.h"
#include "error.h"
#include "threadpool2.h"
#include "memorypool.h"
#include "utils.h"
#include "error.h"
#include "globals.h"
#include "md_array.h"

#ifdef USE_CUDA
#include "cublas_v2.h"
#include "cudnn.h"
#include "md_array_cuda.h"
#endif


namespace lten {

	enum device { CPU, GPU };
	enum dtype { FLOAT32, INT32, UINT8 };

	struct TensorOps
	{
		TensorOps()
		{
			data_type = FLOAT32;
			device_type = CPU;
			device_index = 0;
			alloc_gradient_buffer = false;
		}

		dtype data_type;
		device device_type;
		int device_index;
		bool alloc_gradient_buffer;
	};

	struct TensorImplBase
	{
	public:
		/*these should never get called but can't make them pure virtual*/
		virtual dtype get_data_type() { assert(0);  return FLOAT32; }
		virtual void set_data_type(dtype data_type) { assert(0); }
		virtual void* get_data_ptr() { assert(0); return nullptr; }
		virtual void* get_grad_ptr() { assert(0); return nullptr; }
		virtual const uint64_t* get_sizes() { assert(0); return nullptr; }
		virtual const uint64_t* get_strides() { assert(0); return nullptr; }
		virtual int get_ndims() const { assert(0); return 0; }
		virtual uint64_t get_numels() { return 0; }
		virtual void set_autograd(bool setting) { assert(0); }
		virtual void set_parameter(bool setting) { assert(0); }
		virtual bool autograd_on() { assert(0); return false; }
		virtual device get_device() { assert(0); return CPU; }
		virtual int get_device_index() { assert(0);  return 0; }
		virtual void backward(MultiDimArray<float>* top_gradient_ptr = nullptr) { assert(0); }
		virtual void clear_gradients() { assert(0); }

	public:
		TensorImplBase() : ref_count_(0) {}

		virtual ~TensorImplBase()
		{
			assert(ref_count_.load() == 0);
		}

		int add_ref()
		{
			ref_count_++;
			return ref_count_;
		}

		int release()
		{
			ref_count_--;
			return ref_count_;
		}

		virtual void release_resources() {}

	private:
		mutable std::atomic<int> ref_count_;
	};

	template<typename Dtype>
	struct TensorImpl : public TensorImplBase
	{
	public:
		TensorImpl()
		{
			reset();
		}

		~TensorImpl()
		{
		}

#ifdef USE_MEMORYPOOL
		void* operator new(size_t size)
		{
			return lten::MISC_globals::singleton()->get_cpu_memorypool()->AllocateMemory(size);
		}


		void operator delete(void* memory)
		{
			return lten::MISC_globals::singleton()->get_cpu_memorypool()->FreeMemory(memory);
		}
#endif
		virtual int get_ndims() const { return md_array_base_->GetNDims(); }
		virtual const uint64_t* get_sizes() { return md_array_base_->GetSizes(); }
		virtual const uint64_t* get_strides() { return md_array_base_->GetStrides(); }
		virtual uint64_t get_numels() { return md_array_base_->GetNumels(); }
		virtual void* get_data_ptr() { return md_array_base_->GetDataPtr(); }
		virtual void set_autograd(bool setting) { autograd_on_ = setting; }
		virtual void set_parameter(bool setting) { is_parameter_ = setting; }
		bool autograd_on() { return autograd_on_; }
		void set_grad_fn(void(*grad_fn)(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, TensorImpl** children_ptr_array, int child_index, TensorImpl* parent_ptr)) { grad_fn_ = grad_fn; }
		virtual void* get_grad_ptr() { if (!gradient_ptr_) return nullptr;  return gradient_ptr_->GetDataPtr(); }
		virtual device get_device() { return device_; }
		virtual int get_device_index() { return device_index_; }
		virtual dtype get_data_type() { return data_type_; }
		virtual void set_data_type(dtype data_type) { data_type_ = data_type; }
		virtual void to(TensorImpl& operand1, device target_device, int target_device_index);

		int allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_memory = false, TensorOps* options_ptr = nullptr);
		int allocate_from_buffer(const std::initializer_list<uint64_t>& dims, void* data_ptr, bool own_data_memory, void* gradient_ptr, bool own_gradient_memory, TensorOps* options_ptr = nullptr);
		int allocate_from_buffer(const uint64_t* dims_ptr, int ndims, void* data_ptr, bool own_memory = false, TensorOps* options_ptr = nullptr);

		int allocate(const std::initializer_list<uint64_t>& dims, TensorOps* options_ptr = nullptr);
		int allocate(const uint64_t* dims_ptr, int ndims, TensorOps* options_ptr = nullptr);


		virtual void release_resources();

		MultiDimArray<Dtype>* get_mdarray()
		{
			return md_array_base_;
		}

		MultiDimArray<Dtype>* get_gradients_mdarray()
		{
			return gradient_ptr_;
		}

		void add_child(TensorImpl&);
		virtual void backward(MultiDimArray<Dtype>* top_gradient_ptr = nullptr);
		void do_backward(MultiDimArray<Dtype>* top_gradient_ptr);
		virtual void clear_gradients();
		void set_graph_ref_count();

		void matmul(TensorImpl& operand1, TensorImpl& operand2);
		void add(TensorImpl& operand1, TensorImpl& operand2);
		void sub(TensorImpl& operand1, TensorImpl& operand2);
		void mul(TensorImpl& operand1, TensorImpl& operand2);
		void div(TensorImpl& operand1, TensorImpl& operand2);
		void cat(TensorImpl& operand1, TensorImpl& operand2, int dim);
		void exp(TensorImpl& operand1);
		void log(TensorImpl& operand1);
		void sig(TensorImpl& operand1);
		void tanh(TensorImpl& operand1);
		void sqrt(TensorImpl& operand1);
		void scalar_mul(TensorImpl& operand1, double scalar);
		void max(TensorImpl& operand1);
		void max(TensorImpl& operand1, int dim);
		void sum(TensorImpl& operand1);
		void sum(TensorImpl& operand1, int dim);
		void mean(TensorImpl& operand1);
		void mean(TensorImpl& operand1, const uint32_t* axes, int naxes);
		void var(TensorImpl& operand1, const uint32_t* axes, int naxes);
		void std(TensorImpl& operand1, int dim);
		void sub_array(TensorImpl& operand1, int index); // initialize TensorImpl as a sub array of another (for [][][]... array indexing)
		void reshape(TensorImpl& operand1, const uint64_t* dims, int ndims);
		void squeeze(TensorImpl& operand1, int dim);
		void unsqueeze(TensorImpl& operand1, int dim);
		void transpose(TensorImpl& operand1, int dim1, int dim2);
		void masked_fill(TensorImpl& operand1, TensorImpl& mask, double value);
		void index(TensorImpl& operand1, TensorImpl<int>& index_operand);
		void repeat(TensorImpl& operand1, const uint32_t* repeats, int nrepeats);
		void repeat_interleave(TensorImpl& operand1, const uint32_t* repeats, int nrepeats, int dim, uint32_t* scratch);
		void permute(TensorImpl& operand1, const uint32_t* permutations, int npermutations);

		void(*grad_fn_)(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, TensorImpl** children_ptr_array, int child_index, TensorImpl* parent_ptr);

		//void set_name(const char* name) { strcpy_s(name_, 100, name); }
		//char name_[100];

		uint64_t misc1_; // can be used to store anything (e.g. dimension index for use in back propagation)
		double misc2_; // can be used to store anything (e.g. a scalar for use in back propagation)
		void* misc_ptr1_; // can be used to store anything that requires more than an int
		void* misc_ptr2_; // can be used to store anything that requires more than an int
		bool own_misc_ptr1_; // pointer onwnership 
		bool own_misc_ptr2_;
		TensorImpl* misc_tensor_; // store anything (of type Dtype) that is best kept in tensor form
		TensorImpl<int>* misc_int_tensor_; // store anything (of type int) that is best kept in tensor form
		int graph_ref_count_;

	private:
		MultiDimArray<Dtype>* md_array_base_ = nullptr;
		MultiDimArray<Dtype>* gradient_ptr_ = nullptr;
		bool autograd_on_;
		bool is_parameter_; // network params (i.e. weights/biases)
		device device_;
		int device_index_;
		dtype data_type_;


		enum { MAX_CHILDREN = 2 };
		int num_children_;
		bool accumulate_gradients_;
		TensorImpl* children_[MAX_CHILDREN];
		intrusive_ptr<TensorImpl>* children_lock_[MAX_CHILDREN];

		int64_t multiplier_numel_;

		void reset()
		{
			autograd_on_ = false;
			is_parameter_ = false;
			data_type_ = FLOAT32;
			device_ = CPU;
			device_index_ = 0;
			num_children_ = 0;
			graph_ref_count_ = 0;
			accumulate_gradients_ = false;
			gradient_ptr_ = nullptr;
			grad_fn_ = nullptr;
			//name_[0] = '\0';
			misc1_ = 0;
			misc2_ = 0;
			misc_ptr1_ = nullptr;
			misc_ptr2_ = nullptr;
			own_misc_ptr1_ = false;
			own_misc_ptr2_ = false;
			misc_tensor_ = nullptr;
			misc_int_tensor_ = nullptr;
		}

	};

} // namespace lten

template<typename Dtype>
void matmul_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void sub_array_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void add_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void sub_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void mul_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void div_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void cat_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void max_backward1(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void sum_backward1(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void max_backward2(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void sum_backward2(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void squeeze_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void unsqueeze_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void reshape_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void exp_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void log_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void sig_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void tanh_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void sqrt_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void scalar_mul_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void conv2_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void conv2_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void conv3_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void conv_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void fc_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void relu_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void dropout_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void softmax_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void gru_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void gru_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void transpose_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void nll_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void bn_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void pooling_cudnn_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void embedding_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void mean_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void var_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void std_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void layernorm_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void masked_fill_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void repeat_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void repeat_interleave_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void index_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void permute_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void pseudo_einsum1_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
template<typename Dtype>
void pseudo_einsum2_backward(MultiDimArray<Dtype>* bottom_gradient_ptr, MultiDimArray<Dtype>* top_gradient_ptr, lten::TensorImpl<Dtype>** children_ptr_array, int child_index, lten::TensorImpl<Dtype>* parent_ptr);
#endif // TENSORIMPL_H
