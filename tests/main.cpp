#include "tensor.h"
#include "tests.h"

int main(int argc, char* argv[])
{
	int i;
	uint64_t dims[MAX_DIMS];

	float A[] = { 0.6731, 0.4073, 0.7380, 0.7182, 0.1972, 0.2759, 0.8343, 0.4945, 0.2314, 0.8393, 0.9310, 0.2316, 0.3009, 0.8914, 0.5058, 0.6000, 0.1968, 0.7433, 0.2606, 0.8798, 0.4279, 0.7799, 0.2283, 0.2157 };

	lten::Tensor a = lten::TensorFromBuffer({ 4, 2, 3 }, A);
	uint32_t permutaions[] = { 0, 2, 1 };

	for (i = 0; i < a.get_ndims(); i++)
	{
		dims[i] = a.get_sizes()[permutaions[i]];
	}

	//lten::Tensor ap = lten::RandomTensor({ a.get_sizes()[permutaions[0]], a.get_sizes()[permutaions[1]], a.get_sizes()[permutaions[2]]});
	lten::Tensor ap = lten::RandomTensor(dims, a.get_ndims());
	
	OffsetCalc_permutaion ofs(ap.get_strides(), a.get_strides(), permutaions, a.get_ndims());
	float* src = (float*)a.get_data_ptr();
	float* dst = (float*)ap.get_data_ptr();

	for (i = 0; i < a.get_numels(); i++)
	{
		uint32_t offset;
		offset = ofs.GetOffset(i);

		dst[offset] = src[i];
	}

	std::cout << ap << "\n\n\n";

	a = a.to(lten::GPU);
	lten::Tensor b = a.permute(permutaions, a.get_ndims());
	b = b.to(lten::CPU);
	std::cout << b << "\n\n\n";


	lten::Tensor app = lten::RandomTensor({ a.get_sizes()[permutaions[0]], a.get_sizes()[permutaions[1]], a.get_sizes()[permutaions[2]] });

	a = a.to(lten::GPU);
	app = app.to(lten::GPU);

	gpu_permute((float*)app.get_data_ptr(), (float*)a.get_data_ptr(), a.get_ndims(), a.get_numels(), ap.get_strides(), a.get_strides(), permutaions);

	app = app.to(lten::CPU);
	std::cout << app << "\n";

	embedding_test(); return 0;
	//repeat_interleave_test(); return 0;
	//transpose_test();
	//transpose_backwards_test();

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

  if(run_on_gpu)
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
		printf("tensor ddition test 2 failed\n");
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


	printf("\n------------\nTotal : %d\nPassed: %d\nFailed: %d\n------------\n", total_tests, total_tests_passed, total_tests - total_tests_passed);
	return 0;
}

