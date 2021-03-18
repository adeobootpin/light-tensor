#include <iostream>
#include <chrono>
#include <float.h>
#include "tensor.h"
#include "layers.h"
#include "net.h"
#include "optimizer.h"

#include "benchmark_speech_commands.h"

int speech_commands_lten(DATASET_DATA* training_set, DATASET_DATA* test_set, int epochs)
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int input_dim, int hidden_dim_1, int hidden_dim_2, int sequence_len, int label_len)
		{
			gru1 = register_module("gru1", lten::GRU_CUDNN(input_dim, hidden_dim_1, true, false, 64, sequence_len));
			gru2 = register_module("gru1", lten::GRU_CUDNN(hidden_dim_1, hidden_dim_2, true, false, 64, sequence_len));
			fc1 = register_module("fc1", lten::FullyConnected(hidden_dim_2 * sequence_len, label_len));
			sm = register_module("sm1", lten::softmax_CUDNN(true));
		}

		~Net() {}


		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor output;


			input = input.transpose(0, 1); // transpose because we are using GRU_CUDNN (otherwise dont)

			output = gru1->forward(input);

			output = gru2->forward(output);

			output = output.transpose(0, 1); // transpose because we are using GRU_CUDNN (otherwise dont)


			output = output.reshape({ output.get_sizes()[0], 1, 1, output.get_sizes()[1] * output.get_sizes()[2]});

			output = fc1->forward(output);

			output = sm->forward(output);
			return output;
		}

		lten::Conv2d* cnn1;
		lten::GRU_CUDNN* gru1{ nullptr };
		lten::GRU_CUDNN* gru2{ nullptr };
		lten::FullyConnected* fc1{ nullptr };
		lten::softmax_CUDNN* sm;
		lten::Dropout* dropout1;
	};

	int total_training_examples;
	int total_test_examples;
	int i;
	int j;
	int index;
	int input_dim = 80;
	int hidden_dim_1 = 256;
	int hidden_dim_2 = 128;
	int sequence_len = 40;
	lten::Tensor input;
	lten::Tensor input_gpu;
	lten::Tensor label;
	lten::Tensor label_gpu;
	lten::Tensor output;
	lten::Tensor loss;
	uint64_t data_len = sequence_len * input_dim;
	const uint64_t label_len = 30;  // number of classes
	float label_array[label_len];
	float learning_rate = 0.001f;
	int batch_size;
	float val;
	lten::Tensor temp;

	batch_size = 64;
	Net net(input_dim, hidden_dim_1, hidden_dim_2, sequence_len, label_len);

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	input = lten::AllocateTensor({ (uint64_t)batch_size, (uint64_t)sequence_len, (uint64_t)input_dim }, nullptr);
	label = lten::AllocateTensor({ (uint64_t)batch_size, 1, 1, (uint64_t)label_len }, nullptr);
	input.set_autograd(true);

	lten::AdamOptimizer optimizer;
	optimizer.attach_network(net);
	optimizer.set_learning_rate(learning_rate);

	net.to(lten::GPU);

	index = 0;
	total_training_examples = training_set->total_examples;

	int training_iterations = (int)(epochs * ceil(total_training_examples / (float)batch_size));

	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, total_training_examples - 1);

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
			//unsigned int rnd;

			data_ptr = (float*)input.get_data_ptr();
			label_ptr = (float*)label.get_data_ptr();

			//rnd = rand() + (rand() << 15);
			//int example_idx = rnd % training_set->total_examples;
			int example_idx = distribution(generator);
			memcpy(data_ptr + data_len * j, training_set->audio_data[example_idx], sizeof(float) * data_len);

			memset(label_array, 0, sizeof(label_array));
			label_array[training_set->labels[example_idx]] = 1.0f;
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
	input = lten::AllocateTensor({ 1, (uint64_t)sequence_len, (uint64_t)input_dim }, nullptr);

	total_test_examples = test_set->total_examples;

	int total_correct = 0;
	for (i = 0; i < total_test_examples; i++)
	{
		if (i == 1) // allow for any lazy initialization
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		int example_idx = i;
		memcpy(input.get_data_ptr(), test_set->audio_data[example_idx], sizeof(float) * data_len);

		input_gpu = input.to(lten::GPU);

		output = net.forward(input_gpu);
		output = output.to(lten::CPU);
		float* output_ptr = (float*)output.get_data_ptr();

		int label = -1;
		float max = output_ptr[0];
		for (j = 1; j < label_len; j++)
		{
			if (output_ptr[j] > max)
			{
				max = output_ptr[j];
				label = j;
			}
		}

		if (label == test_set->labels[example_idx])
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


