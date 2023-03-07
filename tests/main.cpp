#include "tensor.h"
#include "tests.h"

#include "layers.h"
#include <conio.h>

void simple_binary_op_test()
{
	lten::Tensor x;
	lten::Tensor y;
	lten::Tensor z;
	lten::Tensor top_gradient;

	float error = 0.0001f;

	srand(0);
	x = lten::RandomTensor({ 2, 1, 1, 3 });
	y = lten::RandomTensor({ 1, 1, 3, 1 });

	//*****************************************************************************************
	float x_vals[] = { 2, 3, 4, 2, 6, 8 };
	float y_vals[] = { 5, 3, 2 };

	x = lten::TensorFromBuffer({ 2, 1, 1, 3 }, x_vals, false);
	y = lten::TensorFromBuffer({ 1, 1, 3, 1 }, y_vals, false);
	//*****************************************************************************************


	for (int i = 0; i < y.get_numels(); i++)
	{
		//((float*)y.get_data_ptr())[i] += 0.0001f;
	}

	x.set_accumulate_gradients(true);
	y.set_accumulate_gradients(true);


	lten::Tensor x_anchor = x;
	lten::Tensor y_anchor = y;

	x = x.to(lten::GPU);
	y = y.to(lten::GPU);

	z = x.matmul(y);

	top_gradient = lten::RandomTensor(z.get_sizes(), z.get_ndims());
	std::cout << top_gradient << "\n";

	top_gradient = top_gradient.to(lten::GPU);
	z.backward(top_gradient.get_mdarray<float>());

	z = z.to(lten::CPU);
	x = x.to(lten::CPU);
	y = y.to(lten::CPU);

	float* ptr = (float*)y.get_grad_ptr();


}

void PoorMansArange(lten::Tensor* x)
{
	uint64_t numels;
	uint64_t i;
	int* data;

	numels = x->get_numels();
	data = (int*)x->get_data_ptr();

	for (i = 0; i < numels; i++)
	{
		data[i] = (int)i;
	}
}

struct OffsetCalc_reverse_broadcast2 // for generating output dim from operand dim that involves broadcast
{
	OffsetCalc_reverse_broadcast2(const int num_dims, const uint64_t* op1_dims, const uint64_t* op2_dims, const uint64_t* output_dims, const uint64_t* op1_strides, const uint64_t* op2_strides, const uint64_t* output_strides)
	{
		uint32_t divisor;
		uint32_t shift;
		uint32_t broadcast_dims[MAX_DIMS];
		uint32_t numels;
		uint32_t i;
		int j;

		ndims_ = num_dims;
		op1_broadcast_ndims_ = 0;

		assert(MAX_DIMS <= sizeof(uint16_t) * 8); // increase size of bitmask if this changes
		op1_bitmask_ = 0;
		op2_bitmask_ = 0;

		for (i = 0; i < ndims_; i++)
		{
			divisor = (uint32_t)op1_strides[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_[i].magic = (uint32_t)magic;
			div_[i].shift = shift;
			div_[i].divisor = divisor;

			op1_strides_[i] = (uint32_t)op1_strides[i];
			op2_strides_[i] = (uint32_t)op2_strides[i];
			output_strides_[i] = (uint32_t)output_strides[i];

			if (op1_dims[i] != output_dims[i])
			{
				assert(op1_dims[i] == 1);
				broadcast_dims[op1_broadcast_ndims_] = (uint32_t)output_dims[i];
				op1_broadcast_ndims_++;
				op1_bitmask_ |= (1 << i); // save the broadcast positions
			}

			if (op2_dims[i] != output_dims[i])
			{
				op2_bitmask_ |= (1 << i); // save the broadcast positions
			}
		}


		//
		// create divsors etc. for broadcast dims
		//
		numels = 1;
		for (j = op1_broadcast_ndims_ - 1; j >= 0; j--)
		{
			broadcast_strides_[j] = numels;
			numels *= broadcast_dims[j];
		}

		for (i = 0; i < op1_broadcast_ndims_; i++)
		{
			divisor = (uint32_t)broadcast_strides_[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			broadcast_div_[i].magic = (uint32_t)magic;
			broadcast_div_[i].shift = shift;
			broadcast_div_[i].divisor = divisor;
		}

	}

	LTEN_HOST_DEVICE void GetParialOffsets(uint32_t operand1_index, uint32_t* operand2_offset, uint32_t* output_offset)
	{
		uint32_t i;
		uint32_t out_offset;
		uint32_t coordinate;

		out_offset = 0;

		for (i = 0; i < MAX_DIMS; ++i)
		{
			if (i == ndims_)
			{
				break;
			}

			coordinate = ((((uint64_t)operand1_index * div_[i].magic) >> 32) + operand1_index) >> div_[i].shift;
			operand1_index = operand1_index - coordinate * div_[i].divisor;

			out_offset += coordinate * output_strides_[i];
		}
		*output_offset = out_offset;
	}

	Divider div_[MAX_DIMS];
	Divider broadcast_div_[MAX_DIMS];
	uint32_t op1_strides_[MAX_DIMS];
	uint32_t op2_strides_[MAX_DIMS];
	uint32_t output_strides_[MAX_DIMS];
	uint32_t broadcast_strides_[MAX_DIMS];
	uint32_t ndims_;
	uint32_t op1_broadcast_ndims_;
	uint16_t op1_bitmask_;
	uint16_t op2_bitmask_;
};



template<typename Dtype>
int Compare(Dtype* A, Dtype* B, uint64_t len, Dtype error);

int main(int argc, char* argv[])
{
	/*
	scalar_mul(true);

	lten::Tensor x;
	lten::Tensor y;
	lten::Tensor z;
	lten::Tensor top_gradient;

	x = lten::RandomTensor({ 16, 1, 64, 1 });
	y = lten::RandomTensor({ 16, 64, 1, 96 });
	x.set_accumulate_gradients(true);
	y.set_accumulate_gradients(true);

	x = x.to(lten::GPU);
	y = y.to(lten::GPU);

	z = x * y;

	top_gradient = lten::RandomTensor(z.get_sizes(), z.get_ndims());
	top_gradient = top_gradient.to(lten::GPU);

	z.backward(top_gradient.get_mdarray<float>());

	return 0;
	*/
	//int xx = matmul_test(true);
	/*
	lten::Tensor a;
	lten::Tensor b;
	lten::Tensor c;
	int rett;
	int ndims;

	lten::Tensor grad;
	srand(0);
	a = lten::RandomTensor({ 4, 1, 64, 32 });
	b = lten::RandomTensor({ 4, 64, 32, 96 });
	a.set_accumulate_gradients(true);
	b.set_accumulate_gradients(true);

	a = a.to(lten::GPU);
	b = b.to(lten::GPU);

	c = a.matmul(b);

	lten::Tensor top_gradient = lten::RandomTensor(c.get_sizes(), c.get_ndims());
	top_gradient = top_gradient.to(lten::GPU);

	c.backward(top_gradient.get_mdarray<float>());


	a = a.to(lten::CPU);
	b = b.to(lten::CPU);
	top_gradient = top_gradient.to(lten::CPU);

	ndims = a.get_ndims();

	grad = top_gradient.matmul(b.transpose(ndims-2, ndims-1));
	grad = grad.sum(1);
	rett = Compare((float*)grad.get_data_ptr(), (float*)a.get_grad_ptr(), a.get_numels(), 0.0001f);

	grad = (a.transpose(ndims-2, ndims-1)).matmul(top_gradient);

	rett = Compare((float*)grad.get_data_ptr(), (float*)b.get_grad_ptr(), b.get_numels(), 0.0001f);
	
	rett = rett;
	*/

/*
	lten::Tensor a;
	lten::Tensor b;
	lten::Tensor c;
	lten::Tensor d;


	if(true)
	{
		lten::Tensor x;
		lten::Tensor y;

		x = lten::RandomTensor({ 16, 64, 64 });
		y = lten::RandomTensor({ 16, 64, 64 });

		x.set_autograd(true);
		y.set_autograd(true);

		d = x + y;
	}

	lten::Tensor A;
	lten::Tensor B;
	lten::Tensor C;
	lten::Tensor D;

	A = lten::RandomTensor({ 16, 64, 64 });
	B = lten::RandomTensor({ 16, 64, 64 });
	C = lten::RandomTensor({ 16, 64, 64 });
	A.set_autograd(true);
	B.set_autograd(true);
	C.set_autograd(true);

	//A.set_name("A");
	//B.set_name("B");
	//C.set_name("C");


	//D = A + (B + C);
	//D = (A + B) + C;
	
	return 0;
	*/
	/*
	lten::Tensor A;
	lten::Tensor B;
	lten::Tensor C;

	lten::Pseudo_Einsum_1 pe;
	A = lten::RandomTensor({ 2, 1, 8, 56, 56, 96 });
	B = lten::RandomTensor({ 56, 7, 96 });
	A = A.to(lten::GPU);
	B = B.to(lten::GPU);
	C = pe.forward(A, B);




	//mul_test();
*/
/*
	lten::Tensor temp_l;
	lten::Tensor temp_r;
	lten::TensorOps ops;
	ops.data_type = lten::INT32;
	lten::Tensor dist_h;
	uint64_t q_h = 56;
	uint64_t k_h = 7;

	float q_h_ratio = 1.0f;
	float k_h_ratio = 8.0f;



	temp_l = lten::AllocateTensor({ q_h, 1 }, &ops);
	temp_r = lten::AllocateTensor({ 1, k_h }, &ops);
	PoorMansArange(&temp_l);
	PoorMansArange(&temp_r);

	temp_l = temp_l.to(lten::GPU);
	temp_r = temp_r.to(lten::GPU);


	dist_h = temp_l - temp_r;


	//dist_h = temp_l * q_h_ratio - temp_r * k_h_ratio;


	dist_h = dist_h.to(lten::CPU);

	std::cout << dist_h << "\n";

	return 0;
	*/
	

	int ret;
	int total_tests;
	int total_tests_passed;
	bool run_on_gpu = false;
	const char* MNIST_training_images;
	const char* MNIST_training_labels;
	const char* MNIST_test_images;
	const char* MNIST_test_labels;

	total_tests = 0;
	total_tests_passed = 0;

	if (argc < 5)
	{
		printf("Usage: test path_to_MNIST_training_images path_to_MNIST_training_labels path_to_MNIST_test_images path_to_MNIST_test_labels [-gpu]\n");
		return -1;
	}
	else
	{
		MNIST_training_images = argv[1];
		MNIST_training_labels = argv[2];
		MNIST_test_images = argv[3];
		MNIST_test_labels = argv[4];

		if (argc > 5)
		{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
			if (!_strcmpi(argv[5], "-gpu"))
#else
			if (!strcasecmp(argv[5], "-GPU"))
#endif
			{
				run_on_gpu = true;
			}
		}
	}

	//goto mist_test;
	//run_on_gpu = false;
	//simple_binary_op_test(); return 0;
	//matmul_test(true);
	//neural_network_test();
	//run_on_gpu = false;
	if (run_on_gpu)
	{
		printf("running unit tests on GPU\n");
	}
	else
	{
		printf("running unit tests on CPU\n");
	}

	ret = add_test_1(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor addition test 1 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor addition test 1 passed\n");
	}

	ret = add_test_2(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor addition test 2 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor addition test 2 passed\n");
	}

	ret = sub_test_1(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor subtraction test 1 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor subtraction test 1 passed\n");
	}

	ret = sub_test_1(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor subtraction test 2 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor subtraction test 2 passed\n");
	}

	ret = sub_test_1_uint8();
	total_tests++;
	if (ret)
	{
		printf("tensor uint8 subtraction test 1 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor uint8 subtraction test 1 passed\n");
	}

	ret = sub_test_2_uint8();
	total_tests++;
	if (ret)
	{
		printf("tensor uint8 subtraction test 2 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor uint8 subtraction test 2 passed\n");
	}

	ret = mul_test_1(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor element-wise multiplication test 1 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor element-wise multiplication test 1 passed\n");
	}

	ret = mul_test_2(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor element-wise multiplication test 2 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor element-wise multiplication test 2 passed\n");
	}

	ret = mul_test_1_int32();
	total_tests++;
	if (ret)
	{
		printf("tensor int32 element-wise multiplication test 2 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor int32 element-wise multiplication test 2 passed\n");
	}

	ret = div_test_1(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor division test 1 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor division test 1 passed\n");
	}

	ret = div_test_2(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor division test 2 failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor division test 2 passed\n");
	}

	ret = matmul_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor matrix-multiplication test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor matrix-multiplication test passed\n");
	}

	ret = exp_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor exp test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor exp test passed\n");
	}

	ret = log_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor log test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor log test passed\n");
	}

	ret = sig_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor sigmoid test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor sigmoid test passed\n");
	}

	ret = tanh_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor tanh test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor tanh test passed\n");
	}

	ret = scalar_mul(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor scalar multiplication test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor scalar multiplication test passed\n");
	}

	ret = sum_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor sum test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor sum test passed\n");
	}


	ret = fc_test(run_on_gpu);
	total_tests++;
	if (ret)
	{
		printf("tensor fully-connected layer test failed\n");
	}
	else
	{
		total_tests_passed++;
		printf("tensor fully-connected layer test passed\n");
	}

	
	if (!run_on_gpu)
	{
		/*
		ret = neural_network_test();
		total_tests++;
		if (ret)
		{
			printf("training regression model failed\n");
		}
		else
		{
			total_tests_passed++;
			printf("training regression model passed\n");
		}
		*/

		ret = MNIST_test(MNIST_training_images, MNIST_training_labels, MNIST_test_images, MNIST_test_labels);
		total_tests++;
		if (ret)
		{
			printf("training MNIST model failed\n");
		}
		else
		{
			total_tests_passed++;
			printf("training MNIST model passed\n");
		}
	}

	if (run_on_gpu)
	{
		ret = MNIST_test_gpu(MNIST_training_images, MNIST_training_labels, MNIST_test_images, MNIST_test_labels);
		total_tests++;
		if (ret)
		{
			printf("training MNIST model failed\n");
		}
		else
		{
			total_tests_passed++;
			printf("training MNIST model passed\n");
		}
	}

	if (!run_on_gpu)
	{
		ret = quantized_MNIST_test(MNIST_training_images, MNIST_training_labels, MNIST_test_images, MNIST_test_labels);
		total_tests++;
		if (ret)
		{
			printf("quantization test failed\n");
		}
		else
		{
			total_tests_passed++;
			printf("quantization test passed\n");
		}
	}


	printf("\n------------\nTotal : %d\nPassed: %d\nFailed: %d\n------------\nPress any key to exit.\n", total_tests, total_tests_passed, total_tests - total_tests_passed);
	_getch();
	return 0;
}

