#include <iostream>
#include <chrono>
#include <float.h>
#include "tensor.h"
#include "layers.h"
#include "net.h"
#include "optimizer.h"


int LoadMNISTImages(const char* filename, float** pp_images, int* total_images);
int LoadMNISTLabels(const char* filename, char** pp_labels);


int MNIST_lten(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels, int epochs)
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int batch_size, int channels_in, int channels_out, int kernel_h, int kernel_w, int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1)
		{
			conv1 = register_module("conv1", lten::conv2d_CUDNN(batch_size, channels_in, 28, 28, channels_out, true, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			fc1 = register_module("fc1", lten::FullyConnected(5760, 50));
			drop1 = register_module("drop1", lten::Dropout(0.5f));
			fc2 = register_module("fc2", lten::FullyConnected(50, 10));

			sm = register_module("sm1", lten::softmax_CUDNN(true));
		}

		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor output;

			output = conv1->forward(input);
			output = relu(output);
			output = output.reshape({ output.get_sizes()[0], 1, 1, 5760 });
			output = fc1->forward(output);
			output = drop1->forward(output);
			output = fc2->forward(output);
			output = sm->forward(output);

			return output;
		}
		
	private:
		lten::conv2d_CUDNN* conv1;
		lten::FullyConnected* fc1;
		lten::Dropout* drop1;
		lten::FullyConnected* fc2;
		lten::softmax_CUDNN* sm;
	};

	int ret;
	float* training_images;
	char* training_labels;
	int total_training_examples;
	float* test_images;
	char* test_labels;
	int total_test_examples;
	int i;
	int j;
	int index;
	lten::Tensor input;
	lten::Tensor input_gpu;
	lten::Tensor label;
	lten::Tensor label_gpu;
	lten::Tensor output;
	lten::Tensor loss;
	uint64_t img_dim = 28;
	uint64_t data_len = img_dim * img_dim;
	uint64_t label_len = 10;
	float label_array[10];
	float lr = 0.001f;
	int batch_size;
	float val;
	lten::Tensor temp;

	ret = LoadMNISTImages(MNIST_training_images, &training_images, &total_training_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_training_images << " file failed to load]" << std::endl;
		return -1;
	}

	ret = LoadMNISTLabels(MNIST_training_labels, &training_labels);
	if (ret)
	{
		std::cout << "unable to load MNIST labels [" << MNIST_training_labels << " file failed to load]" << std::endl;
		return -1;
	}


	ret = LoadMNISTImages(MNIST_test_images, &test_images, &total_test_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_test_images << " file failed to load]" << std::endl;
		return -1;
	}

	ret = LoadMNISTLabels(MNIST_test_labels, &test_labels);
	if (ret)
	{
		std::cout << "unable to load MNIST labels [" << MNIST_test_labels << " file failed to load]" << std::endl;
		return -1;
	}

	
	for (i = 0; i < total_training_examples * data_len; i++)
	{
		training_images[i] = training_images[i] / 255.0f;
	}

	for (i = 0; i < total_test_examples * data_len; i++)
	{
		test_images[i] = test_images[i] / 255.0f;
	}

	batch_size = 64;
	Net net(batch_size, 1, 10, 5, 5);

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	input = lten::AllocateTensor({ (uint64_t)batch_size, 1, img_dim, img_dim }, nullptr);
	label = lten::AllocateTensor({ (uint64_t)batch_size, 1, 1, label_len }, nullptr);
	input.set_autograd(true);

	lten::AdamOptimizer optimizer;
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.to(lten::GPU);

	srand(10);
	index = 0;

	int training_iterations = epochs * ceil(total_training_examples / (float)batch_size);

	for (i = 0; i < training_iterations; i++)
	{
		if (i == 1) // allow for any lazy initialization
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		//--------------------------------------------------------------------------------------
		for (j = 0; j < batch_size; j++)
		{
			float* data_ptr;
			float* label_ptr;

			data_ptr = (float*)input.get_data_ptr();
			label_ptr = (float*)label.get_data_ptr();

			int example_idx = (i * batch_size + j) % total_training_examples;
			memcpy(data_ptr + data_len * j, training_images + example_idx * data_len, sizeof(float) * data_len);

			memset(label_array, 0, sizeof(label_array));
			label_array[training_labels[example_idx]] = 1.0f;
			memcpy(label_ptr + label_len * j, label_array, sizeof(label_array));

		}
		//--------------------------------------------------------------------------------------
		input_gpu = input.to(lten::GPU);
		label_gpu = label.to(lten::GPU);

		output = net.forward(input_gpu);

		loss = nll_loss(output, label_gpu);

		if ((index++ % 1000) == 0)
		{
			temp = loss.to(lten::CPU);
			val = *((float*)temp.get_data_ptr());
			printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();

	}

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	temp = loss.to(lten::CPU);
	val = *((float*)temp.get_data_ptr());
	printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);
	printf("training completed [duration: %f sec]\nrunning inference...\n", nseconds);

	net.train(false);
	input = lten::AllocateTensor({ 1, 1, img_dim, img_dim }, nullptr);
	
	int total_correct = 0;
	for (i = 0; i < total_test_examples; i++)
	{
		if (i == 1) // allow for any lazy initialization
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		int example_idx = i;
		memcpy(input.get_data_ptr(), test_images + example_idx * data_len, sizeof(float) * data_len);

		input_gpu = input.to(lten::GPU);

		output = net.forward(input_gpu);
		output = output.to(lten::CPU);
		float* output_ptr = (float*)output.get_data_ptr();

		int label = 0;
		float max = output_ptr[0];
		for (j = 1; j < 10; j++)
		{
			if (output_ptr[j] > max)
			{
				max = output_ptr[j];
				label = j;
			}
		}

		if (label == test_labels[example_idx])
		{
			total_correct++;
		}
	}

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	printf("inference completed [duration: %f sec]\nscore: %f%% [%d/%d]\n\n", nseconds, (100.0f * total_correct) / total_test_examples, total_correct, total_test_examples);
	
	return 0;
}

