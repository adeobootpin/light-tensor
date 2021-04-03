#include <iostream>
#include <chrono>
#include <float.h>
#include "lten.h"


int LoadMNISTImages(const char* pchFileName, float** ppfImages, int* piTotalImages);
int LoadMNISTLabels(const char* pchFileName, char** ppchLabels);

//-------------------------------------------------------------------------------
// x = [x0, x1, x2, x3], y = [y0 = x0 * 2, y1 = x1 * 4, y2 = x2 * 6, y3 = x3 * 8]
// learn mapping between x and y
// data generated on the fly so no need to download a dataset
//-------------------------------------------------------------------------------
int neural_network_test()
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net()
		{
			fc1 = register_module("fc1", lten::FullyConnected(4, 4, true));
			fc2 = register_module("fc2", lten::FullyConnected(4, 4, true));
			fc3 = register_module("fc3", lten::FullyConnected(4, 4, true));
		}

		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor output;

			output = fc1->forward(input);
			output = fc2->forward(output);
			output = fc3->forward(output);

			return output;
		}

	private:
		lten::FullyConnected* fc1;
		lten::FullyConnected* fc2;
		lten::FullyConnected* fc3;
	};

	int i;
	int j;
	int k;
	float val;
	uint64_t batch_size;
	int64_t numels_per_batch;
	int index;
	lten::Tensor input;
	lten::Tensor target;
	lten::Tensor output;
	lten::Tensor loss;

	Net net;
	float* input_ptr;
	float* target_ptr;
	float lr = 0.0002f;

	batch_size = 32;

	input = lten::AllocateTensor({ batch_size, 1, 1, 4 }, nullptr);
	target = lten::AllocateTensor({ batch_size, 1, 1, 4 }, nullptr);
	input.set_autograd(true);

	input_ptr = (float*)input.get_data_ptr();
	target_ptr = (float*)target.get_data_ptr();

	lten::AdamOptimizer optimizer;
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	numels_per_batch = input.get_numels() / batch_size;
	index = 0;

	printf("\ntraining regression model on CPU...\n");

	for (i = 0; i < 15000; i++)
	{
		for (j = 0; j < batch_size; j++)
		{
			for (k = 0; k < numels_per_batch; k++)
			{
				input_ptr[numels_per_batch * j + k] = rand() % 10 + ((rand() % 100) / 1000.0f);
				if (rand() % 2)
				{
					input_ptr[numels_per_batch * j + k] *= -1.0f;
				}
				target_ptr[numels_per_batch * j + k] = input_ptr[numels_per_batch * j + k] * ((k + 1) * 2);
				target_ptr[numels_per_batch * j + k] += ((rand() % 100) / 1000.0f); // add some noise
			}
		}

		output = net.forward(input);
		loss = mse_loss(output, target);

		if ((index++ % 1000) == 0)
		{
			val = *((float*)loss.get_data_ptr());
			printf("  loss: %f\n", val);
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();
	}

	val = *((float*)loss.get_data_ptr());
  printf("  loss: %f\n", val);
	if (val < 50.0f)
	{
		return 0;
	}

	return -1;
}


//-----------------------------------
// train MNIST data set using the CPU
//-----------------------------------
int MNIST_test(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels)
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int channels_in, int channels_out, int kernel_h, int kernel_w, int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1)
		{
			conv1 = register_module("conv1", lten::Conv2d(channels_in, channels_out, false, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			fc1 = register_module("fc1", lten::FullyConnected(5760, 50));
			drop1 = register_module("drop1", lten::Dropout(0.5f));
			fc2 = register_module("fc2", lten::FullyConnected(50, 10));
		}

		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor output;

			output = conv1->forward(input);
			output = output.reshape({ output.get_sizes()[0], 1, 1, 5760 });
			output = relu(output);
			output = fc1->forward(output);
			output = drop1->forward(output);
			output = fc2->forward(output);
			output = log_softmax(output, 3);

			return output;
		}

	private:
		lten::Conv2d* conv1;
		lten::FullyConnected * fc1;
		lten::Dropout* drop1;
		lten::FullyConnected * fc2;
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
	lten::Tensor label;
	lten::Tensor output;
	lten::Tensor loss;
	uint64_t img_dim = 28;
	uint64_t data_len = img_dim * img_dim;
	uint64_t label_len = 10;
	float label_array[10];
	float lr = 0.001f;
	Net net(1, 10, 5, 5);
	int batch_size;
	int epochs;
	float val;

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

	batch_size = 32;

	input = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, img_dim, img_dim }, nullptr);
	label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, label_len }, nullptr);
	input.set_autograd(true);

	lten::SGDOptimizer optimizer; //lten::AdamOptimizer also works fine
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);


	index = 0;
	epochs = 1;
	int training_iterations = epochs * (total_training_examples / batch_size);

	printf("\ntraining MNIST model on CPU for %d epoch(s)...\n", epochs);

	for (i = 0; i < training_iterations; i++)
	{
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

		output = net.forward(input);

		loss = nll_loss(output, label);

		if ((index++ % 200) == 0)
		{
			val = *((float*)loss.get_data_ptr());
			printf("  loss: %f [%d%% completed]\n",val, i * 100 / training_iterations);
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();

	}

	val = *((float*)loss.get_data_ptr());
	printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);

	printf("running inference...\n");
	net.train(false);
	int total_correct = 0;
	for (i = 0; i < total_test_examples; i++)
	{
		int example_idx =  i;
		memcpy(input.get_data_ptr(), test_images + example_idx * data_len, sizeof(float) * data_len);

		output = net.forward(input);
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

	printf("score: %d/%d [%f%%]\n", total_correct, total_test_examples, (100.0f * total_correct) / total_test_examples);

	if ((100.0f * total_correct) / total_test_examples > 85.0f)
	{
		return 0;
	}

	return -1;
}

//---------------------------------
// train MNIST data set using a GPU
//---------------------------------
int MNIST_test_gpu(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels)
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int channels_in, int channels_out, int kernel_h, int kernel_w, int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1)
		{
			conv1 = register_module("conv1", lten::Conv2d(channels_in, channels_out, false, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			fc1 = register_module("fc1", lten::FullyConnected(5760, 50));
			drop1 = register_module("drop1", lten::Dropout(0.5f));
			fc2 = register_module("fc2", lten::FullyConnected(50, 10));
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
			output = log_softmax(output, 3);

			return output;
		}

	private:
		lten::Conv2d* conv1;
		lten::FullyConnected * fc1;
		lten::Dropout* drop1;
		lten::FullyConnected * fc2;
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
	float lr = 0.0001f;
	Net net(1, 10, 5, 5);
	int batch_size;
	int epochs;
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

	batch_size = 32;

	input = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, img_dim, img_dim }, nullptr);
	label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, label_len }, nullptr);
	input.set_autograd(true);

	lten::AdamOptimizer optimizer; //lten::SGDOptimizer also works fine
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.to(lten::GPU);

	index = 0;
	epochs = 1;
	int training_iterations = epochs * (total_training_examples / batch_size);

	printf("\ntraining MNIST model on GPU for %d epoch(s)...\n", epochs);

	for (i = 0; i < training_iterations; i++)
	{
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

		if ((index++ % 200) == 0)
		{
			temp = loss.to(lten::CPU);
			val = *((float*)temp.get_data_ptr());
			printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();

	}

	temp = loss.to(lten::CPU);
	val = *((float*)temp.get_data_ptr());
	printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);

	printf("running inference...\n");
	net.train(false);
	int total_correct = 0;
	
	for (i = 0; i < total_test_examples; i++)
	{
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

	printf("score: %d/%d [%f%%]\n", total_correct, total_test_examples, (100.0f * total_correct) / total_test_examples);

	if ((100.0f * total_correct) / total_test_examples > 85.0f)
	{
		return 0;
	}

	return -1;
}



//-----------------------------------------------------------------------------------------------
// 1. train an MNIST model using FP32 (train for only a few iterations to make this a quick test)
// 2. run inference using the FP32 model
// 3. quantize the FP32 weights to UINT8
// 4. run inference using the UINT8 model
// 5. compare FP32 and UINT8 results
//-----------------------------------------------------------------------------------------------
int quantized_MNIST_test(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels)
{
	struct blob_statistics // structure for storing data statistics during FP32 training
	{
		blob_statistics()
		{
			min = FLT_MAX;
			max = -FLT_MAX;
		}

		float min;
		float max;
	};

	blob_statistics layer_stats[4]; // input, conv1, fc1, fc2

	class Net : public lten::NeuralNetwork  // FP32 neural network
	{
	public:
		Net(int channels_in, int channels_out, int kernel_h, int kernel_w, int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1)
		{
			conv1 = register_module("conv1", lten::Conv2d(channels_in, channels_out, false, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			fc1 = register_module("fc1", lten::FullyConnected(5760, 50));
			fc2 = register_module("fc2", lten::FullyConnected(50, 10));
		}

		void update_statistics(lten::Tensor* x, blob_statistics* x_stats)
		{
			uint64_t i;
			uint64_t len;
			float min = x_stats->min;
			float max = x_stats->max;
			float* data;

			len = x->get_numels();
			data = (float*)x->get_data_ptr();

			for (i = 0; i < len; i++)
			{
				if (data[i] > max)
				{
					max = data[i];
				}
				if (data[i] < min)
				{
					min = data[i];
				}
			}

			x_stats->min = min;
			x_stats->max = max;
		}

		lten::Tensor forward(lten::Tensor input, blob_statistics* layer_stats)
		{
			lten::Tensor output;

			if (layer_stats)
				update_statistics(&input, &layer_stats[0]);

			output = conv1->forward(input);
			if (layer_stats)
				update_statistics(&output, &layer_stats[1]);

			output = output.reshape({ output.get_sizes()[0], 1, 1, 5760 });
			output = fc1->forward(output);

			if (layer_stats)
				update_statistics(&output, &layer_stats[2]);
			output = fc2->forward(output);

			if (layer_stats)
				update_statistics(&output, &layer_stats[3]);
			output = log_softmax(output, 3);

			return output;
		}

	private:
		lten::Conv2d* conv1;
		lten::FullyConnected * fc1;
		lten::FullyConnected * fc2;
	};

	class QNet : public lten::NeuralNetwork // UINT8 neural network (uses quantized weights from the FP32 network for inference)
	{
	public:
		QNet(int channels_in, int channels_out, int kernel_h, int kernel_w, int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1)
		{
			conv1 = register_module("conv1", lten::Conv2d_q(channels_in, channels_out, false, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			fc1 = register_module("fc1", lten::FullyConnected_q(5760, 50));
			fc2 = register_module("fc2", lten::FullyConnected_q(50, 10));
		}

		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor output;

			output = conv1->forward(input);
			output = output.reshape({ output.get_sizes()[0], 1, 1, 5760 });
			output = fc1->forward(output);
			output = fc2->forward(output);

			return output;
		}

	private:
		lten::Conv2d_q* conv1;
		lten::FullyConnected_q* fc1;
		lten::FullyConnected_q* fc2;
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
	lten::Tensor q_input;
	lten::Tensor label;
	lten::Tensor output;
	lten::Tensor q_output;
	lten::Tensor loss;
	uint64_t img_dim = 28;
	uint64_t data_len = img_dim * img_dim;
	uint64_t label_len = 10;
	float label_array[10];
	float lr = 0.001f;
	Net net(1, 10, 5, 5);
	int batch_size;
	int epochs;
	float val;
	float fp32_score;
	float uint8_score;


	//------------------------------------------------------------------------------------------
	// load MNIST data
	//------------------------------------------------------------------------------------------
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
	//------------------------------------------------------------------------------------------


	batch_size = 32;

	input = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, img_dim, img_dim }, nullptr);
	label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, label_len }, nullptr);
	input.set_autograd(true);

	lten::SGDOptimizer optimizer;
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);


	index = 0;
	epochs = 10;
	int training_iterations = epochs * (total_training_examples / batch_size);
	training_iterations = 1000;  // for quick testing

	//-----------------
	// train FP32 model
	//-----------------
	printf("\nrunning quantization test...\ntraining FP32 MNIST model on CPU for %d iterations\n", training_iterations);
	for (i = 0; i < training_iterations; i++)
	{
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


		output = net.forward(input, layer_stats);

		loss = nll_loss(output, label);

		if ((index++ % 100) == 0)
		{
			val = *((float*)loss.get_data_ptr());
			printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();

	}

	val = *((float*)loss.get_data_ptr());
	printf("  loss: %f [%d%% completed]\n", val, i * 100 / training_iterations);

	printf("\ntraining FP32 MNIST model completed\n");
	
	//-------------------------
	// set up quantized network
	//-------------------------
	printf("quantizing FP32 MNIST model for UINT8 inference\n");
	QNet qnet(1, 10, 5, 5);
	QuantizationParams qp_input;
	QuantizationParams qp_fc1;
	QuantizationParams qp_fc2;
	QuantizationParams qp_fc3;
	QuantizationParams qp_wts[2];
	lten::Module* mod;
	lten::Module* q_mod;
	float minimum;
	float maximum;
	int count;

	ComputeQuantizationParams(layer_stats[0].min, layer_stats[0].max, &qp_input);
	ComputeQuantizationParams(layer_stats[1].min, layer_stats[1].max, &qp_fc1);
	ComputeQuantizationParams(layer_stats[2].min, layer_stats[2].max, &qp_fc2);
	ComputeQuantizationParams(layer_stats[3].min, layer_stats[3].max, &qp_fc3);

	mod = net.get_module(0);
	q_mod = qnet.get_module(0);
	q_mod->set_qparams_in(qp_input);
	q_mod->set_qparams_out(qp_fc1);
	count = 0;
	for (auto wts : mod->get_all_weights())
	{
		// count = 0: weight (uint8_t)
		// count = 1: bias (int32)
		if (count == 0)
		{
			GetMinMax((float*)wts->get_data_ptr(), wts->get_numels(), &minimum, &maximum);
			ComputeQuantizationParams(minimum, maximum, &qp_wts[count]);
			Quantize((float*)wts->get_data_ptr(), (uint8_t*)q_mod->get_all_weights()[count]->get_data_ptr(), wts->get_numels(), &qp_wts[count]);
		}
		else
		{
			qp_wts[count].scale = qp_input.scale * qp_wts[0].scale; // arXiv:1712.05877 Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (see 
			qp_wts[count].zero_point = 0;
			Quantize((float*)wts->get_data_ptr(), (int*)q_mod->get_all_weights()[count]->get_data_ptr(), wts->get_numels(), &qp_wts[count]);
		}

		count++;
	}
	q_mod->set_qparams_params(qp_wts, count);


	mod = net.get_module(1);
	q_mod = qnet.get_module(1);
	q_mod->set_qparams_in(qp_fc1);
	q_mod->set_qparams_out(qp_fc2);
	count = 0;
	for (auto wts : mod->get_all_weights())
	{
		// count = 0: weight (uint8_t)
		// count = 1: bias (int32)
		if (count == 0)
		{
			GetMinMax((float*)wts->get_data_ptr(), wts->get_numels(), &minimum, &maximum);
			ComputeQuantizationParams(minimum, maximum, &qp_wts[count]);
			Quantize((float*)wts->get_data_ptr(), (uint8_t*)q_mod->get_all_weights()[count]->get_data_ptr(), wts->get_numels(), &qp_wts[count]);
		}
		else
		{
			qp_wts[count].scale = qp_fc1.scale * qp_wts[0].scale; // arXiv:1712.05877 Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (see 
			qp_wts[count].zero_point = 0;
			Quantize((float*)wts->get_data_ptr(), (int*)q_mod->get_all_weights()[count]->get_data_ptr(), wts->get_numels(), &qp_wts[count]);
		}

		count++;
	}
	q_mod->set_qparams_params(qp_wts, count);

	mod = net.get_module(2);
	q_mod = qnet.get_module(2);
	q_mod->set_qparams_in(qp_fc2);
	q_mod->set_qparams_out(qp_fc3);
	count = 0;
	for (auto wts : mod->get_all_weights())
	{
		// count = 0: weight (uint8_t)
		// count = 1: bias (int32)
		if (count == 0)
		{
			GetMinMax((float*)wts->get_data_ptr(), wts->get_numels(), &minimum, &maximum);
			ComputeQuantizationParams(minimum, maximum, &qp_wts[count]);
			Quantize((float*)wts->get_data_ptr(), (uint8_t*)q_mod->get_all_weights()[count]->get_data_ptr(), wts->get_numels(), &qp_wts[count]);
		}
		else
		{
			qp_wts[count].scale = qp_fc2.scale * qp_wts[0].scale; // arXiv:1712.05877 Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (see 
			qp_wts[count].zero_point = 0;
			Quantize((float*)wts->get_data_ptr(), (int*)q_mod->get_all_weights()[count]->get_data_ptr(), wts->get_numels(), &qp_wts[count]);
		}

		count++;
	}
	q_mod->set_qparams_params(qp_wts, count);


	lten::TensorOps options;
	options.data_type = lten::UINT8;


	input = lten::AllocateTensor({ 1, 1, img_dim, img_dim }, nullptr);
	q_input = lten::AllocateTensor({ 1, 1, img_dim, img_dim }, &options);


	//------------------------------
	// run inference with FP32 model
	//------------------------------
	printf("running inference...\n");
	input.set_autograd(false);
	int total_correct = 0;

	for (i = 0; i < total_test_examples; i++)
	{
		int example_idx = i;
		memcpy(input.get_data_ptr(), test_images + example_idx * data_len, sizeof(float) * data_len);

		output = net.forward(input, nullptr);
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

	fp32_score = (100.0f * total_correct) / total_test_examples;
	printf("score [fp32]: %d/%d [%f%%]\n", total_correct, total_test_examples, fp32_score);
	
	//-------------------------------
	// run inference with UINT8 model
	//-------------------------------
	output.set_autograd(false);
	total_correct = 0;
	for (i = 0; i < total_test_examples; i++)
	{
		int example_idx = i;
		memcpy(input.get_data_ptr(), test_images + example_idx * data_len, sizeof(float) * data_len);

		Quantize((float*)input.get_data_ptr(), (uint8_t*)q_input.get_data_ptr(), q_input.get_numels(), &qp_input);
		q_output = qnet.forward(q_input);
		Dequantize((float*)output.get_data_ptr(), (uint8_t*)q_output.get_data_ptr(), q_output.get_numels(), &qp_fc3);
		output = log_softmax(output, 3);

	
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

	uint8_score = (100.0f * total_correct) / total_test_examples;
	printf("score [uint8]: %d/%d [%f%%]\n", total_correct, total_test_examples, uint8_score);

	if (uint8_score / fp32_score > 0.95f)
	{
		return 0;
	}

	return -1;
}

void ReverseBytes(unsigned char* bytes, int byte_count)
{
	int i;
	unsigned char* temp;

	temp = new unsigned char[byte_count];
	memcpy(temp, bytes, byte_count);

	for (i = 0; i < byte_count; i++)
	{
		bytes[byte_count - i - 1] = temp[i];
	}

	delete temp;
}


int LoadMNISTImages(const char* filename, float** pp_images, int* total_images)
{
	int ret;
	int i;
	int j;
	int k;
	unsigned char* data;
	size_t data_size;
	int magic;
	int image_count;
	int height;
	int width;
	unsigned char* char_index;
	unsigned char* char_images;
	float* float_images;
	int index;

	ret = ReadDataFromFile(filename, (void**)&data, &data_size);
	if (ret)
	{
		printf("Failed to load MNIST images [%s]\n", filename);
		return ret;
	}

	char_index = data;
	ReverseBytes(char_index, sizeof(int));
	memcpy(&magic, char_index, sizeof(int));

	char_index += sizeof(int);
	ReverseBytes(char_index, sizeof(int));
	memcpy(&image_count, char_index, sizeof(int));

	char_index += sizeof(int);
	ReverseBytes(char_index, sizeof(int));
	memcpy(&height, char_index, sizeof(int));

	char_index += sizeof(int);
	ReverseBytes(char_index, sizeof(int));
	memcpy(&width, char_index, sizeof(int));

	char_index += sizeof(int);

	char_images = new unsigned char[height * width];
	float_images = new float[image_count * height * width];

	index = 0;
	for (i = 0; i < image_count; i++)
	{
		for (j = 0; j < height; j++)
		{
			for (k = 0; k < width; k++)
			{
				float_images[index++] = *char_index++;
			}
		}
	}

	delete data;

	*total_images = image_count;
	*pp_images = float_images;

	return 0;
}


int LoadMNISTLabels(const char* filename, char** pp_labels)
{
	int ret;
	int i;
	unsigned char* data;
	size_t data_size;
	int magic;
	int image_count;
	unsigned char* pch;
	char* labels;

	ret = ReadDataFromFile(filename, (void**)&data, &data_size);
	if (ret)
	{
		printf("Failed to load MNIST labels [%s]\n", filename);
		return ret;
	}

	pch = data;
	ReverseBytes(pch, sizeof(int));
	memcpy(&magic, pch, sizeof(int));

	pch += sizeof(int);
	ReverseBytes(pch, sizeof(int));
	memcpy(&image_count, pch, sizeof(int));

	pch += sizeof(int);

	labels = new char[image_count];

	for (i = 0; i < image_count; i++)
	{
		double dd = (double)(*pch);
		labels[i] = (*pch++);
	}

	delete data;

	*pp_labels = labels;

	return 0;
}
