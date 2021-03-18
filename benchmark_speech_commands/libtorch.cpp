#include <torch/torch.h>
#include <float.h>

#include "benchmark_speech_commands.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif



class speech_commands_dataset : public torch::data::Dataset<speech_commands_dataset>
{
public:

	speech_commands_dataset(DATASET_DATA* data_set, int sequence_len, int input_dim);

	torch::data::Example<> get(size_t index) override;

	torch::optional<size_t> size() const override
	{
		return data_set_->total_examples;
	};

private:
	DATASET_DATA* data_set_;
	int sequence_len_;
	int input_dim_;
};

speech_commands_dataset::speech_commands_dataset(DATASET_DATA* data_set, int sequence_len, int input_dim)
{
	data_set_ = data_set;
	sequence_len_ = sequence_len;
	input_dim_ = input_dim;
}

torch::data::Example<> speech_commands_dataset::get(size_t index)
{
	torch::Tensor audio, labels;

	audio = torch::from_blob(data_set_->audio_data[index], { sequence_len_, input_dim_ }, torch::requires_grad());

	labels = torch::from_blob(&data_set_->labels[index], { 1 }, torch::kLong);

	labels = labels.squeeze(0);

	return { audio, labels };
}



class Net : public torch::nn::Module
{
public:
	Net(int input_dim, int hidden_dim_1, int hidden_dim_2, int sequence_len, int label_len)
	{
		gru1 = register_module("gru1", torch::nn::GRU(torch::nn::GRUOptions(input_dim, hidden_dim_1).num_layers(1).bias(false).batch_first(true).bidirectional(false)));
		gru2 = register_module("gru2", torch::nn::GRU(torch::nn::GRUOptions(hidden_dim_1, hidden_dim_2).num_layers(1).bias(false).batch_first(true).bidirectional(false)));
		fc1 = register_module("fc1", torch::nn::Linear(hidden_dim_2 * sequence_len, label_len));
	}

	~Net(){}

	torch::Tensor forward(torch::Tensor input)
	{
		torch::Tensor output;

		auto tuple = gru1(input);
		output = std::get<0>(tuple);

		tuple = gru2(output);
		output = std::get<0>(tuple);

		output = output.reshape({ output.size(0), output.size(1) * output.size(2)});
		
		output = fc1->forward(output);

		return torch::log_softmax(output, 1);
	}


	torch::nn::GRU gru1{ nullptr };
	torch::nn::GRU gru2{ nullptr };
	torch::nn::Linear fc1{ nullptr };
};


int speech_commands_libtorch(DATASET_DATA* training_set, DATASET_DATA* testing_set, int epochs)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	HMODULE hMod = LoadLibraryA("torch_cuda.dll");
#else
	void* handle = dlopen("libtorch_cuda.so", RTLD_LAZY);
#endif

	if (torch::cuda::is_available())
	{
		if (!torch::cuda::cudnn_is_available())
		{
			std::cout << "CUDNN unavailable" << std::endl;
			return -1;
		}
	}
	else
	{
		std::cout << "CUDA unavailable" << std::endl;
		return -1;
	}


	torch::Device device(torch::kCUDA);

	int batch_size = 32;
	int test_batch_size = 1;
	int input_dim = 80;
	int hidden_dim_1 = 256;
	int hidden_dim_2 = 128;
	int sequence_len = 40;
	int label_len = 30;
	float learning_rate = 0.001f;
	int iteration;
	int training_epochs;
	int training_iterations;
	torch::Tensor loss;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	auto net = std::make_shared<Net>(input_dim, hidden_dim_1, hidden_dim_2, sequence_len, label_len);

	auto data_set = speech_commands_dataset(training_set, sequence_len, input_dim).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(data_set), batch_size);

	auto test_set = speech_commands_dataset(testing_set, sequence_len, input_dim).map(torch::data::transforms::Stack<>());
	auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), test_batch_size);

	net->to(device);

	torch::optim::Adam optimizer(net->parameters(), learning_rate);

	torch::optional<size_t> size = data_set.size();

	training_epochs = epochs;
	training_iterations = training_epochs * ceil(size.value() / (float)batch_size);
	iteration = 0;
	for (int epoch = 0; epoch < training_epochs; epoch++)
	{
		for (auto& batch : *data_loader)
		{
			if (iteration == 1) // allow for any lazy initialization
			{
				clock_begin = std::chrono::steady_clock::now();
			}

			torch::Tensor speech_data = batch.data.to(device);
			torch::Tensor speech_labels = batch.target.to(device);

			optimizer.zero_grad();

			torch::Tensor prediction = net->forward(speech_data);

			loss = torch::nll_loss(prediction, speech_labels);
			loss.backward();
			optimizer.step();

			if (iteration % 1000 == 0)
			{
				printf("  loss: %f [%d%% completed]\n", loss.item<float>(), iteration * 100 / training_iterations);
			}
			iteration++;
		}
	}

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	printf("  loss: %f [%d%% completed]\n", loss.item<float>(), iteration * 100 / training_iterations);
	printf("training completed [duration: %f sec]\nrunning inference...\n", nseconds);


	net->train(false);

	int total_correct = 0;
	int total_test_examples = 0;
	int* prediction_indices = new int[test_batch_size];

	iteration = 0;
	for (auto& batch : *test_data_loader)
	{
		if (iteration == 1) // allow for any lazy initialization
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		torch::Tensor speech_data = batch.data.to(device);
		torch::Tensor speech_labels = batch.target.to(device);

		optimizer.zero_grad();

		torch::Tensor prediction = net->forward(speech_data);

		torch::Tensor contig = prediction.contiguous();
		contig = contig.to(torch::kCPU);
		float* pred = (float*)contig.data_ptr();


		int64_t label = -1;
		float max = pred[0];
		for (int j = 1; j < label_len; j++)
		{
			if (pred[j] > max)
			{
				max = pred[j];
				label = j;
			}
		}

		if (label == speech_labels.item<int64_t>())
		{
			total_correct++;
		}

		total_test_examples++;

		iteration++;
	}

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	printf("inference completed [duration: %f sec]\nscore: %f%% [%d/%d]\n\n", nseconds, total_correct * 100.0f / total_test_examples, total_correct, total_test_examples);

	return 0;
}
