#include "tensor.h"
#include "tests.h"

int main(int argc, char* argv[])
{
	repeat_interleave_backward_test(); return 0;

	lten::Tensor aa = lten::RandomTensor({ 2, 1, 8, 56, 56, 96 });
	aa = aa.to(lten::GPU);
	aa = aa.permute({ 3, 0, 1, 2, 4, 5 });


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

