#include <torch/torch.h>
#include <float.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

int LoadMNISTImages(const char* filename, float** pp_images, int* total_images);
int LoadMNISTLabels(const char* filename, char** pp_labels);

class MNISTDataset : public torch::data::Dataset<MNISTDataset>
{
private:
	torch::Tensor images_, labels_;
	int total_examples_;
	float* images_raw_;
	char* labels_raw_;

public:

	MNISTDataset(const char* MNIST_images, const char* MNIST_labels);

	torch::data::Example<> get(size_t index) override;

	torch::optional<size_t> size() const override
	{
		return total_examples_;
	};
};

MNISTDataset::MNISTDataset(const char* MNIST_images, const char* MNIST_labels)
{
	int i;
	int j;
	int index;
	int64_t* labels_ptr;
	int ret;

	ret = LoadMNISTImages(MNIST_images, &images_raw_, &total_examples_);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_images << " file failed to load]" << std::endl;
		return;
	}

	ret = LoadMNISTLabels(MNIST_labels, &labels_raw_);
	if (ret)
	{
		std::cout << "unable to load MNIST labels [" << MNIST_labels << " file failed to load]" << std::endl;
		return;
	}


	labels_ptr = new int64_t[total_examples_];
	for (i = 0; i < total_examples_; i++)
	{
		labels_ptr[i] = (int64_t)labels_raw_[i];
	}
	index = 0;
	for (i = 0; i < total_examples_; i++)
	{
		for (j = 0; j < 28 * 28; j++)
		{
			images_raw_[index++] /= 255.0f;
		}
	}

	images_ = torch::from_blob(images_raw_, { total_examples_, 28 * 28 });
	labels_ = torch::from_blob(labels_ptr, { total_examples_ }, torch::kLong);

}

torch::data::Example<> MNISTDataset::get(size_t index)
{
	return { images_[index], labels_[index] };
}


class Net : public torch::nn::Module
{
public:
	Net()
	{
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)));
		fc1 = register_module("fc1", torch::nn::Linear(5760, 50));
		drop1 = register_module("drop1", torch::nn::Dropout());
		fc2 = register_module("fc2", torch::nn::Linear(50, 10));
	}


	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(conv1->forward(x));
		x = x.reshape({ x.size(0), 5760 });
		x = fc1->forward(x);
		x = torch::dropout(x, 0.5, is_training());
		x = fc2->forward(x);

		return torch::log_softmax(x, 1);
	}

	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Dropout drop1{ nullptr };
	torch::nn::Linear fc1{ nullptr };
	torch::nn::Linear fc2{ nullptr };
};



int MNIST_libtorch(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels, int epochs)
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

	int batch_size = 64;
	int test_batch_size = 1;
	int iteration;
	int training_epochs;
	int training_iterations;
	torch::Tensor loss;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	auto net = std::make_shared<Net>();

	auto data_set = MNISTDataset(MNIST_training_images, MNIST_training_labels).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set), batch_size);

	auto test_set = MNISTDataset(MNIST_test_images, MNIST_test_labels).map(torch::data::transforms::Stack<>());
	auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), test_batch_size);

	net->to(device);

	torch::optim::Adam optimizer(net->parameters(), 0.001);

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

			torch::Tensor digit_images = batch.data.to(device);
			torch::Tensor digit_labels = batch.target.to(device);

			optimizer.zero_grad();

			digit_images = digit_images.reshape({ digit_images.size(0), 1, 28, 28 });
			torch::Tensor prediction = net->forward(digit_images);
			loss = torch::nll_loss(prediction, digit_labels);
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
	int real_batch_size;
	int* prediction_indices = new int[test_batch_size];

	iteration = 0;
	for (auto& batch : *test_data_loader)
	{
		if (iteration == 1) // allow for any lazy initialization
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		torch::Tensor digit_images = batch.data.to(device);
		torch::Tensor digit_labels = batch.target.to(device);


		digit_images = digit_images.reshape({ digit_images.size(0), 1, 28, 28 });
		torch::Tensor prediction = net->forward(digit_images);

		real_batch_size = prediction.sizes()[0];

		torch::Tensor contig = prediction.contiguous();
		contig = contig.to(torch::kCPU);
		float* pred = (float*)contig.data_ptr();


		for (int i = 0; i < real_batch_size; i++)
		{
			float max = -FLT_MAX;
			int max_index;
			for (int j = 0; j < 10; j++)
			{
				if (pred[i * 10 + j] > max)
				{
					max_index = j;
					max = pred[j];
				}
			}

			prediction_indices[i] = max_index;
		}

		contig = digit_labels.contiguous();
		contig = contig.to(torch::kCPU);
		uint64_t* gt = (uint64_t*)contig.data_ptr();

		for (int i = 0; i < real_batch_size; i++)
		{
			if (prediction_indices[i] == gt[i])
			{
				total_correct++;
			}

			total_test_examples++;
		}
		iteration++;
	}

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	printf("inference completed [duration: %f sec]\nscore: %f%% [%d/%d]\n\n", nseconds, total_correct * 100.0f / total_test_examples, total_correct, total_test_examples);

	return 0;

}

