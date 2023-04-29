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
	input.set_accumulate_gradients(true);
	target.set_accumulate_gradients(true);
	//input.set_autograd(true);

	input_ptr = (float*)input.get_data_ptr();
	target_ptr = (float*)target.get_data_ptr();

	lten::AdamOptimizer optimizer;
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.train(true);

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
	float lr = 0.0001f;
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
	//label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, label_len }, nullptr);
	label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, 1 }, nullptr);
	input.set_autograd(true);

	lten::AdamOptimizer optimizer; //lten::SGDOptimizer also works fine
	//lten::SGDOptimizer optimizer; 
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.train(true);

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

			//memset(label_array, 0, sizeof(label_array));
			//label_array[training_labels[example_idx]] = 1.0f;
			//memcpy(label_ptr + label_len * j, label_array, sizeof(label_array));

			label_ptr[j] = training_labels[example_idx];
		}
		//--------------------------------------------------------------------------------------

		output = net.forward(input);

		loss = nll_loss(output, label, 3);

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
	//label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, label_len }, nullptr);
	label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, 1 }, nullptr);
	input.set_autograd(true);

	lten::AdamOptimizer optimizer; //lten::SGDOptimizer also works fine
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.to(lten::GPU);
	net.train(true);

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

			//memset(label_array, 0, sizeof(label_array));
			//label_array[training_labels[example_idx]] = 1.0f;
			//memcpy(label_ptr + label_len * j, label_array, sizeof(label_array));

			label_ptr[j] = training_labels[example_idx];
		}
		//--------------------------------------------------------------------------------------
		input_gpu = input.to(lten::GPU);
		label_gpu = label.to(lten::GPU);

		output = net.forward(input_gpu);

		loss = nll_loss(output, label_gpu, 3);

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
	label = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, 1, 1 }, nullptr);
	input.set_autograd(true);


	lten::AdamOptimizer optimizer;
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.train(true);

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


			//memset(label_array, 0, sizeof(label_array));
			//label_array[training_labels[example_idx]] = 1.0f;
			//memcpy(label_ptr + label_len * j, label_array, sizeof(label_array));

			label_ptr[j] = training_labels[example_idx];
		}
		//--------------------------------------------------------------------------------------


		output = net.forward(input, layer_stats);

		loss = nll_loss(output, label, 3);

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

//---------------------------------
// train simple PixelCNN model
//---------------------------------
int PixelCNN_test(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels)
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int channels, int kernel_h, int kernel_w)
		{
			int pad_h = kernel_h / 2;
			int pad_w = kernel_h / 2;
			int stride_h = 1;
			int stride_w = 1;
			int i;
			int j;
			float* mask_a_data;
			float* mask_b_data;
			bool use_bias = true;

			conv1 = register_module("conv1", lten::Conv2d(1, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv2 = register_module("conv2", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv3 = register_module("conv3", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv4 = register_module("conv4", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv5 = register_module("conv5", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv6 = register_module("conv6", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv7 = register_module("conv7", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv8 = register_module("conv8", lten::Conv2d(channels, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			output = register_module("output", lten::Conv2d(channels, 256, use_bias, 1, 1, 0, 0, stride_h, stride_w));

			mask_a = lten::AllocateTensor({ 1, 1, (uint64_t)kernel_h * (uint64_t)kernel_w });
			mask_b = lten::AllocateTensor({ 1, 1, (uint64_t)kernel_h * (uint64_t)kernel_w });

			mask_a_data = (float*)mask_a.get_data_ptr();
			mask_b_data = (float*)mask_b.get_data_ptr();

			for (i = 0; i < kernel_h; i++)
			{
				for (j = 0; j < kernel_w; j++)
				{	
					if (i > kernel_h / 2)
					{
						mask_a_data[i * kernel_w + j] = 0;
						mask_b_data[i * kernel_w + j] = 0;
					}
					else
					{
						mask_a_data[i * kernel_w + j] = 1;
						mask_b_data[i * kernel_w + j] = 1;
					}

					if ((i == kernel_h / 2) && (j >= kernel_w / 2))
					{
						mask_a_data[i * kernel_w + j] = 0;
					}

					if ((i == kernel_h / 2) && (j > kernel_w / 2))
					{
						mask_b_data[i * kernel_w + j] = 0;
					}
				}
			}	
		}

		void mask_weights(lten::Tensor* weight, lten::Tensor* mask)
		{
			lten::Tensor temp;
			lten::TensorOps options;

			options.data_type = weight->get_data_type();
			options.device_type = weight->get_device();
			options.device_index = weight->get_device_index();

			temp = lten::TensorFromBuffer(weight->get_sizes(), weight->get_ndims(), weight->get_data_ptr(), false, &options);
			temp = temp * (*mask);

			if (options.device_type == lten::GPU)
			{
				GPUToGPUCopy(weight->get_data_ptr(), temp.get_data_ptr(), sizeof(float) * temp.get_numels());
			}
			else
			{
				memcpy(weight->get_data_ptr(), temp.get_data_ptr(), sizeof(float) * temp.get_numels());
			}

		}

		void to( lten::device target_device, int target_device_index = 0)
		{
			mask_a = mask_a.to(target_device, target_device_index);
			mask_b = mask_b.to(target_device, target_device_index);

			NeuralNetwork::to(target_device, target_device_index);
		}

		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor x;

			mask_weights(conv1->get_weights(), &mask_a);
			x = conv1->forward(input);
			x = relu(x);

			mask_weights(conv2->get_weights(), &mask_b);
			x = conv2->forward(x);
			x = relu(x);

			mask_weights(conv3->get_weights(), &mask_b);
			x = conv3->forward(x);
			x = relu(x);

			mask_weights(conv4->get_weights(), &mask_b);
			x = conv4->forward(x);
			x = relu(x);
			
			mask_weights(conv5->get_weights(), &mask_b);
			x = conv5->forward(x);
			x = relu(x);

			mask_weights(conv6->get_weights(), &mask_b);
			x = conv6->forward(x);
			x = relu(x);

			mask_weights(conv7->get_weights(), &mask_b);
			x = conv7->forward(x);
			x = relu(x);

			mask_weights(conv8->get_weights(), &mask_b);
			x = conv8->forward(x);
			x = relu(x);

			x = output->forward(x);
	
			return x;
		}

	private:
		lten::Conv2d* conv1;
		lten::Conv2d* conv2;
		lten::Conv2d* conv3;
		lten::Conv2d* conv4;
		lten::Conv2d* conv5;
		lten::Conv2d* conv6;
		lten::Conv2d* conv7;
		lten::Conv2d* conv8;
		lten::Conv2d* output;

		lten::Tensor mask_a;
		lten::Tensor mask_b;
	};

	int i;
	int ret;
	float* training_images;
	int total_training_examples;
	float* test_images;
	int total_test_examples;
	int img_dim = 28;
	int data_len = img_dim * img_dim;
	lten::Tensor input;
	lten::Tensor input_to_net;
	lten::Tensor output;
	lten::Tensor loss;
	lten::Tensor gt;
	lten::Tensor gt_to_net;
	int batch_size = 64;
	float lr = 1e-3f;

	Net net(64, 7, 7);

	//-------------------------------------------------------------------
	/*
	srand(101);
	int j;
	int k;
	int pixel;
	float max;
	uint64_t sample_size = 24;
	net.train(false);
	ret = net.load_checkpoint("f:\\xfer\\pixel_cnn_with_bias_checkpoint_3.bin");
	lten::Tensor sample = lten::AllocateTensor({ sample_size, 1, (uint64_t)img_dim, (uint64_t)img_dim }, nullptr);
	lten::Tensor probs = lten::AllocateTensor({ sample_size, 256 }, nullptr);
	memset(sample.get_data_ptr(), 0, sample.get_numels() * sizeof(float));
	for (i = 0; i < img_dim; i++)
	{
		for (j = 0; j < img_dim; j++)
		{
			output = net.forward(sample);
			output = lten::softmax(output);

			for (int s = 0; s < sample_size; s++)
			{
				//probs = probs.index({ torch::indexing::Slice(), torch::indexing::Slice(), i, j });
				for (k = 0; k < 256; k++)
				{
					((float*)probs.get_data_ptr())[s * 256 + k] = ((float*)output.get_data_ptr())[s * 256 * img_dim * img_dim + k * img_dim * img_dim + i * img_dim + j];
				}
			}

			lten::Tensor val = lten::Multinomial(probs, 1);

			for (int s = 0; s < sample_size; s++)
			{
				//sample.index({ torch::indexing::Slice(), torch::indexing::Slice(), i, j }) = val / 255.0;
				((float*)sample.get_data_ptr())[s * img_dim * img_dim + i * img_dim + j] = ((float*)val.get_data_ptr())[s] / 255.0;

				std::cout << val << "\n";
				//printf("%f\n", ((float*)val.get_data_ptr())[s]);
			}
			
		}
	}
	sample = sample * 255.0f;
	WriteDataToFile("f:\\xfer\\pixel_cnn_image4.bin", sample.get_data_ptr(), sample.get_numels() * sizeof(float));
	*/
	//-------------------------------------------------------------------


	ret = LoadMNISTImages(MNIST_training_images, &training_images, &total_training_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_training_images << " file failed to load]" << std::endl;
		return -1;
	}

	ret = LoadMNISTImages(MNIST_test_images, &test_images, &total_test_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_test_images << " file failed to load]" << std::endl;
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

	batch_size = 144;

	input = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, (uint64_t)img_dim, (uint64_t)img_dim }, nullptr);
	input.set_autograd(true);

	gt = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, (uint64_t)img_dim, (uint64_t)img_dim }, nullptr);

	lten::AdamOptimizer optimizer; //lten::SGDOptimizer also works fine
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.to(lten::GPU);
	net.train(true);

	float loss_val = 0;
	int epochs = 25;
	int training_iterations = epochs * (total_training_examples / batch_size);

	for (i = 0; i < training_iterations; i++)
	{
		//--------------------------------------------------------------------------------------
		float* data_ptr;
		data_ptr = (float*)input.get_data_ptr();
		for (int j = 0; j < batch_size; j++)
		{
			int example_idx = (i * batch_size + j) % total_training_examples;
			memcpy(data_ptr + data_len * j, training_images + example_idx * data_len, sizeof(float) * data_len);
		}


		float* gt_ptr;
		gt_ptr = (float*)gt.get_data_ptr();
		int len = gt.get_numels();
		for (int j = 0; j < len; j++)
		{
			gt_ptr[j] = data_ptr[j] * 255.0f;
		}
		//--------------------------------------------------------------------------------------
		
		input_to_net = input.to(lten::GPU);
		gt_to_net = gt.to(lten::GPU);

		input_to_net.set_autograd(true);
		input_to_net.set_accumulate_gradients(true);

		output = net.forward(input_to_net);

		output = lten::log_softmax(output);

		loss = nll_loss(output, gt_to_net);

		lten::Tensor temp = loss.to(lten::CPU);
		loss_val +=(*((float*)temp.get_data_ptr()));

		if (i % 10 == 0)
		{
			char message[100];
			float epch = (i + 1.0f) / ((float)total_training_examples / batch_size);
			sprintf_s(message, "Epoch: %2d Loss: %f [%f]\n", (int)epch, loss_val / (i + 1), (*((float*)temp.get_data_ptr())));
			printf(message);
			//std::cout << "Loss: " << loss_val / (iteration + 1) << "\n";
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();
	}

	ret = net.save_checkpoint("f:\\xfer\\pixel_cnn_with_bias_checkpoint_3.bin");
	return 0;
}


//-----------------------------------------
// train simple PixelCNN model using CUDNN
//-----------------------------------------
int PixelCNN_CUDNN_test(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels)
{
	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int batch_size, int channels, int height, int width, int kernel_h, int kernel_w)
		{
			int pad_h = kernel_h / 2;
			int pad_w = kernel_h / 2;
			int stride_h = 1;
			int stride_w = 1;
			int i;
			int j;
			float* mask_a_data;
			float* mask_b_data;
			bool use_bias = true;

			conv1 = register_module("conv1", lten::conv2d_CUDNN(batch_size, 1, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv2 = register_module("conv2", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv3 = register_module("conv3", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv4 = register_module("conv4", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			conv5 = register_module("conv5", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			//conv6 = register_module("conv6", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			//conv7 = register_module("conv7", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			//conv8 = register_module("conv8", lten::conv2d_CUDNN(batch_size, channels, height, width, channels, use_bias, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
			output = register_module("output", lten::conv2d_CUDNN(batch_size, channels, height, width, 256, use_bias, 1, 1, pad_h, pad_w, stride_h, stride_w));

			sm = register_module("sm1", lten::softmax_CUDNN(true));

			mask_a = lten::AllocateTensor({ 1, 1, (uint64_t)kernel_h, (uint64_t)kernel_w });
			mask_b = lten::AllocateTensor({ 1, 1, (uint64_t)kernel_h, (uint64_t)kernel_w });

			mask_a_data = (float*)mask_a.get_data_ptr();
			mask_b_data = (float*)mask_b.get_data_ptr();

			for (i = 0; i < kernel_h; i++)
			{
				for (j = 0; j < kernel_w; j++)
				{
					if (i > kernel_h / 2)
					{
						mask_a_data[i * kernel_w + j] = 0;
						mask_b_data[i * kernel_w + j] = 0;
					}
					else
					{
						mask_a_data[i * kernel_w + j] = 1;
						mask_b_data[i * kernel_w + j] = 1;
					}

					if ((i == kernel_h / 2) && (j >= kernel_w / 2))
					{
						mask_a_data[i * kernel_w + j] = 0;
					}

					if ((i == kernel_h / 2) && (j > kernel_w / 2))
					{
						mask_b_data[i * kernel_w + j] = 0;
					}
				}
			}
		}

		void mask_weights(lten::Tensor* weight, lten::Tensor* mask)
		{
			lten::Tensor temp;
			lten::TensorOps options;

			options.data_type = weight->get_data_type();
			options.device_type = weight->get_device();
			options.device_index = weight->get_device_index();

			temp = lten::TensorFromBuffer(weight->get_sizes(), weight->get_ndims(), weight->get_data_ptr(), false, &options);
			temp = temp * (*mask);

			if (options.device_type == lten::GPU)
			{
				GPUToGPUCopy(weight->get_data_ptr(), temp.get_data_ptr(), sizeof(float) * temp.get_numels());
			}
			else
			{
				memcpy(weight->get_data_ptr(), temp.get_data_ptr(), sizeof(float) * temp.get_numels());
			}

		}

		void to(lten::device target_device, int target_device_index = 0)
		{
			mask_a = mask_a.to(target_device, target_device_index);
			mask_b = mask_b.to(target_device, target_device_index);

			NeuralNetwork::to(target_device, target_device_index);
		}

		lten::Tensor forward(lten::Tensor input)
		{
			lten::Tensor x;

			mask_weights(conv1->get_weights(), &mask_a);
			x = conv1->forward(input);
			x = relu(x);

			mask_weights(conv2->get_weights(), &mask_b);
			x = conv2->forward(x);
			x = relu(x);

			mask_weights(conv3->get_weights(), &mask_b);
			x = conv3->forward(x);
			x = relu(x);

			mask_weights(conv4->get_weights(), &mask_b);
			x = conv4->forward(x);
			x = relu(x);
			/*
			mask_weights(conv5->get_weights(), &mask_b);
			x = conv5->forward(x);
			x = relu(x);
			
			mask_weights(conv6->get_weights(), &mask_b);
			x = conv6->forward(x);
			x = relu(x);
			
			mask_weights(conv7->get_weights(), &mask_b);
			x = conv7->forward(x);
			x = relu(x);

			mask_weights(conv8->get_weights(), &mask_b);
			x = conv8->forward(x);
			x = relu(x);
			*/
			x = output->forward(x);
			//x = sm->forward(x);
			x = lten::log_softmax(x);

			//WriteDataToFile("e:\\xfer\\output.bin", x.get_data_ptr(), x.get_numels() * sizeof(float));

			return x;
		}

	private:
		lten::conv2d_CUDNN* conv1;
		lten::conv2d_CUDNN* conv2;
		lten::conv2d_CUDNN* conv3;
		lten::conv2d_CUDNN* conv4;
		lten::conv2d_CUDNN* conv5;
		lten::conv2d_CUDNN* conv6;
		lten::conv2d_CUDNN* conv7;
		lten::conv2d_CUDNN* conv8;
		lten::conv2d_CUDNN* output;

		lten::softmax_CUDNN* sm;

		lten::Tensor mask_a;
		lten::Tensor mask_b;
	};

	int i;
	int ret;
	float* training_images;
	int total_training_examples;
	float* test_images;
	int total_test_examples;
	int img_dim = 28;
	int data_len = img_dim * img_dim;
	lten::Tensor input;
	lten::Tensor input_to_net;
	lten::Tensor output;
	lten::Tensor loss;
	lten::Tensor gt;
	lten::Tensor gt_to_net;
	int batch_size = 64;
	float lr = 1e-3f;
	//lr *= 0.5f;
	Net net(batch_size, 64, 28, 28, 7, 7);

	ret = LoadMNISTImages(MNIST_training_images, &training_images, &total_training_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_training_images << " file failed to load]" << std::endl;
		return -1;
	}

	ret = LoadMNISTImages(MNIST_test_images, &test_images, &total_test_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_test_images << " file failed to load]" << std::endl;
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

	input = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, (uint64_t)img_dim, (uint64_t)img_dim }, nullptr);
	input.set_autograd(true);

	gt = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, (uint64_t)img_dim, (uint64_t)img_dim }, nullptr);

	lten::AdamOptimizer optimizer; //lten::SGDOptimizer also works fine
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	net.to(lten::GPU);
	net.train(true);

	float loss_val = 0;
	int epochs = 25;
	int training_iterations = epochs * (total_training_examples / batch_size);

	for (i = 0; i < training_iterations; i++)
	{
		//--------------------------------------------------------------------------------------
		float* data_ptr;
		data_ptr = (float*)input.get_data_ptr();
		for (int j = 0; j < batch_size; j++)
		{
			int example_idx = (i * batch_size + j) % total_training_examples;
			memcpy(data_ptr + data_len * j, training_images + example_idx * data_len, sizeof(float) * data_len);
		}


		float* gt_ptr;
		gt_ptr = (float*)gt.get_data_ptr();
		int len = gt.get_numels();
		for (int j = 0; j < len; j++)
		{
			gt_ptr[j] = data_ptr[j] * 255.0f;
		}
		//--------------------------------------------------------------------------------------

		input_to_net = input.to(lten::GPU);
		gt_to_net = gt.to(lten::GPU);

		input_to_net.set_autograd(true);
		input_to_net.set_accumulate_gradients(true);

		output = net.forward(input_to_net);

		loss = nll_loss(output, gt_to_net);

		lten::Tensor temp = loss.to(lten::CPU);
		loss_val += (*((float*)temp.get_data_ptr()));


		if (i % 10 == 0)
		{
			char message[100];
			float epch = (i + 1.0f) / ((float)total_training_examples / batch_size);
			sprintf_s(message, "Epoch: %2d Loss: %f\n", (int)epch, loss_val / (i + 1));
			printf(message);
		}

		loss.backward();

		optimizer.step();
		optimizer.zero_grad();
	}

	return 0;
}

//------------------------
// train simple VAE model 
//------------------------
int vae_test(const char* MNIST_training_images, const char* MNIST_test_images)
{

	class Net : public lten::NeuralNetwork
	{
	public:
		Net(int channels, int latent_dims)
		{
			int kernel = 4;
			int stride = 2;
			int pad = 1;
			bool use_bias = false;
			int c;

			is_training_ = false;
			channels_ = channels;
			latent_dims_ = latent_dims;
			c = channels;

			enc_conv1 = register_module("enc_conv1", lten::Conv2d(1, c, use_bias, kernel, kernel, pad, pad, stride, stride));
			enc_conv2 = register_module("enc_conv2", lten::Conv2d(c, c * 2, use_bias, kernel, kernel, pad, pad, stride, stride));
			fc_mu = register_module("fc_mu", lten::FullyConnected(c * 2 * 7 * 7, latent_dims));
			fc_logvar = register_module("fc_logvar", lten::FullyConnected(c * 2 * 7 * 7, latent_dims));


			fc = register_module("fc", lten::FullyConnected(latent_dims, c * 2 * 7 * 7));
			fc2 = register_module("fc2", lten::FullyConnected(c * 2 * 7 * 7, 784));
			dec_conv3 = register_module("dec_conv3", lten::Conv2d(1, c, use_bias, kernel, kernel, pad, pad, stride, stride));
			fc3 = register_module("fc3", lten::FullyConnected(c * 14 * 14, 784));


#ifdef USE_CUDA
			curandCreateGenerator(&cuda_generator_, CURAND_RNG_PSEUDO_DEFAULT);
#endif
			
		}


		void to(lten::device target_device, int target_device_index = 0)
		{
			enc_conv1->to(target_device, target_device_index);
			enc_conv2->to(target_device, target_device_index);
			fc_mu->to(target_device, target_device_index);
			fc_logvar->to(target_device, target_device_index);

			dec_conv3->to(target_device, target_device_index);
			fc->to(target_device, target_device_index);
			fc2->to(target_device, target_device_index);
			fc3->to(target_device, target_device_index);

			NeuralNetwork::to(target_device, target_device_index);
		}

		void train(bool is_training)
		{
			is_training_ = is_training;

			NeuralNetwork::train(is_training);
		}

		std::tuple<lten::Tensor, lten::Tensor> encode(lten::Tensor x)
		{
			x = enc_conv1->forward(x);
			x = lten::relu(x);

			x = enc_conv2->forward(x);
			x = lten::relu(x);

			x = x.reshape({ x.get_sizes()[0], 1, (uint64_t)(channels_ * 2 * 7 * 7) });

			lten::Tensor x_mu = fc_mu->forward(x);
			lten::Tensor x_logvar = fc_logvar->forward(x);

			return std::make_tuple(x_mu, x_logvar);
		}


		// poor man's decoder: transpose convolutions not implemented so use fc layers to expand the features
		// throw in a convolution to help keep the 'imageness' of the decoding (does not seem to work without it)
		lten::Tensor decode(lten::Tensor x)  
		{
			int c;

			c = channels_;

			x = fc->forward(x);
			x = fc2->forward(x);
			x = x.reshape({ x.get_sizes()[0], 1, 28, 28 });

			x = dec_conv3->forward(x);
			x = lten::relu(x);

			x = x.reshape({ x.get_sizes()[0], 1, (uint64_t)(c * 14 * 14) });
			x = fc3->forward(x);

			x = x.reshape({ x.get_sizes()[0], 1, 28, 28 });
			x = x.sig();

			return x;
		}


		lten::Tensor sample(lten::Tensor mu, lten::Tensor log_var)
		{
			if (is_training_)
			{
				lten::TensorOps options;
				options.data_type = mu.get_data_type();
				options.device_type = mu.get_device();
				options.device_index = mu.get_device_index();

				lten::Tensor std = (log_var * 0.5).exp();
				lten::Tensor eps = lten::AllocateTensor(std.get_sizes(), std.get_ndims(), &options);

				if (options.device_type == lten::GPU)
				{
#ifdef USE_CUDA
					curandGenerateNormal(cuda_generator_, (float*)eps.get_data_ptr(), eps.get_numels(), 0, 1.0f);
#endif
				}
				else
				{
					//--------------------------------
					int index = 0;
					for (int i = 0; i < eps.get_sizes()[0]; i++)
					{
						for (int j = 0; j < eps.get_numels() / eps.get_sizes()[0]; j++)
						{
							((float*)eps.get_data_ptr())[index++] = distribution_(generator_);
						}
					}
					//--------------------------------

				}
				return eps * std + mu;

			}
			else
			{
				return mu;
			}
		}


		std::tuple<lten::Tensor, lten::Tensor, lten::Tensor> forward(lten::Tensor x)
		{
			lten::Tensor latent_mu;
			lten::Tensor latent_logvar;
			lten::Tensor latent;
			lten::Tensor x_recon;

			std::tie(latent_mu, latent_logvar) = encode(x);

			latent = sample(latent_mu, latent_logvar);

			x_recon = decode(latent);

			return std::make_tuple(x_recon, latent_mu, latent_logvar);
		}
		/*
		lten::Tensor vae_loss(lten::Tensor image_recon, lten::Tensor image, lten::Tensor mu, lten::Tensor logvar)
		{
			lten::Tensor recon_loss;
			lten::Tensor kldivergence;
			float beta = 1.0f;

			//recon_loss = torch::mse_loss(image_recon.view({ -1, 784 }), image.view({ -1, 784 }), torch::Reduction::Sum);
			//kldivergence = -0.5 * torch::sum(1 + logvar - mu.pow(2) - logvar.exp());

			recon_loss = lten::mse_loss(image_recon, image);
			recon_loss = recon_loss.reshape({ 1, 1, 1 });

			static lten::Tensor one;
			one = lten::AllocateTensor({ 1,1,1 });
			((float*)one.get_data_ptr())[0] = 1.0f;

			//kldivergence = logvar - (mu * mu) - logvar.exp();
			//kldivergence = one + kldivergence;
			//kldivergence = kldivergence.sum();

			kldivergence = logvar - (mu * mu) - logvar.exp();
			//kldivergence = one + kldivergence;
			kldivergence = (kldivergence.sum()) * -0.5f;

			//return recon_loss + beta * kldivergence;
			return recon_loss + kldivergence;
		}
		*/


		lten::Tensor vae_loss(lten::Tensor image_recon, lten::Tensor image, lten::Tensor mu, lten::Tensor logvar)
		{
			lten::Tensor recon_loss;
			lten::Tensor kldivergence;
			float beta = 1.0f;

			//recon_loss = torch::mse_loss(image_recon.view({ -1, 784 }), image.view({ -1, 784 }), torch::Reduction::Sum);
			//kldivergence = -0.5 * torch::sum(1 + logvar - mu.pow(2) - logvar.exp());

			recon_loss = lten::mse_loss(image_recon, image, false);
			recon_loss = recon_loss.reshape({ 1, 1, 1 });


			lten::TensorOps options;
			options.data_type = mu.get_data_type();
			options.device_type = mu.get_device();
			options.device_index = mu.get_device_index();

			lten::Tensor one;
			one = lten::AllocateTensor({ 1,1,1 }, &options);

			if (options.device_type == lten::GPU)
			{
				float number_1 = 1.0f;
				CopyDataToGPU(one.get_data_ptr(), &number_1, sizeof(float));
			}
			else
			{
				((float*)one.get_data_ptr())[0] = 1.0f;
			}
			

			kldivergence = ((one + logvar - (mu * mu) - logvar.exp()).sum()) * (-0.5f);

			return recon_loss + kldivergence;
		}

	private:
		lten::Conv2d* enc_conv1;
		lten::Conv2d* enc_conv2;
		lten::FullyConnected* fc_mu;
		lten::FullyConnected* fc_logvar;

		lten::Conv2d* dec_conv3;
		lten::FullyConnected* fc;
		lten::FullyConnected* fc2;
		lten::FullyConnected* fc3;

		int channels_;
		int latent_dims_;
		bool is_training_;

#ifdef USE_CUDA
		curandGenerator_t cuda_generator_;
#endif
		std::default_random_engine generator_;
		std::normal_distribution<float> distribution_{ 0, 1 };


	};

	int i;
	int ret;
	float* training_images;
	int total_training_examples;
	float* test_images;
	int total_test_examples;
	int img_dim = 28;
	int data_len = img_dim * img_dim;
	lten::Tensor input;
	lten::Tensor input_to_net;
	lten::Tensor output;
	lten::Tensor loss;
	int batch_size = 64;
	float lr = 1e-3f;
	int latent_dims = 2;

	Net net(64, latent_dims);

	//-------------------------------------------------------------------
	
	int j;
	int k;
	int pixel;
	float max;
	net.train(false);
	ret = net.load_checkpoint("f:\\xfer\\vae_lten.bin");

	///////////////////////////////////////
	std::default_random_engine generator;
	std::normal_distribution<float> distribution{ 0, 1 };
	lten::Tensor latent = lten::AllocateTensor({ 24, 1, (uint64_t)latent_dims });
	int index = 0;
	for (int i = 0; i < latent.get_sizes()[0]; i++)
	{
		distribution(generator);
		for (int j = 0; j < latent.get_numels() / latent.get_sizes()[0]; j++)
		{
			((float*)latent.get_data_ptr())[index++] = distribution(generator);
		}
	}
	///////////////////////////////////////
	lten::Tensor img_recon = net.decode(latent);

	img_recon = img_recon * 255.0f;
	WriteDataToFile("f:\\xfer\\vae_lten_imagexx.bin", img_recon.get_data_ptr(), img_recon.get_numels() * sizeof(float));
	
	//-------------------------------------------------------------------



	ret = LoadMNISTImages(MNIST_training_images, &training_images, &total_training_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_training_images << " file failed to load]" << std::endl;
		return -1;
	}

	ret = LoadMNISTImages(MNIST_test_images, &test_images, &total_test_examples);
	if (ret)
	{
		std::cout << "unable to load MNIST images [" << MNIST_test_images << " file failed to load]" << std::endl;
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

	batch_size = 128;

	input = lten::AllocateTensor({ static_cast<uint64_t>(batch_size), 1, (uint64_t)img_dim, (uint64_t)img_dim }, nullptr);
	input.set_autograd(true);

	lten::AdamOptimizer optimizer; 
	optimizer.attach_network(net);
	optimizer.set_learning_rate(lr);

	lten::device dev = lten::GPU;

	net.to(dev);
	net.train(true);

	lten::Tensor image_recon;
	lten::Tensor latent_mu;
	lten::Tensor latent_logvar;


	float loss_val = 0;
	int epochs = 100;
	int training_iterations = epochs * (total_training_examples / batch_size);
	int epoch_iterations = training_iterations / epochs; printf("epoch_iterations: %d\n", epoch_iterations);


	lten::TensorOps options;
	options.data_type = lten::FLOAT32;
	options.device_type = dev;
	options.device_index = 0;

	
	for (i = 0; i < training_iterations; i++)
	{
		//--------------------------------------------------------------------------------------
		float* data_ptr;
		data_ptr = (float*)input.get_data_ptr();
		for (int j = 0; j < batch_size; j++)
		{
			int example_idx = (i * batch_size + j) % total_training_examples;
			memcpy(data_ptr + data_len * j, training_images + example_idx * data_len, sizeof(float) * data_len);
		}
		//--------------------------------------------------------------------------------------

		input_to_net = input.to(dev);
		input_to_net = input_to_net.reshape({ input_to_net.get_sizes()[0], 1, 28, 28 });

		input_to_net.set_autograd(true);
		input_to_net.set_accumulate_gradients(true);

		std::tie(image_recon, latent_mu, latent_logvar) = net.forward(input_to_net);

		lten::Tensor loss = net.vae_loss(image_recon, input_to_net, latent_mu, latent_logvar);

		lten::Tensor temp = loss.to(lten::CPU);
		loss_val += (*((float*)temp.get_data_ptr()));

		if (i % 10 == 0)
		{
			char message[100];
			float epch = (i + 1.0f) / ((float)total_training_examples / batch_size);
			sprintf_s(message, "Epoch: %2d Loss: %f [%f]\n", (int)epch, loss_val / (i + 1), (*((float*)temp.get_data_ptr())));
			printf(message);
		}



		lten::Tensor top_gradient;
		top_gradient = lten::AllocateTensor(loss.get_sizes(), loss.get_ndims(), &options);
		if (dev == lten::GPU)
		{
			float number_1 = 1.0f;
			CopyDataToGPU(top_gradient.get_data_ptr(), &number_1, sizeof(float));

		}
		else
		{
			((float*)top_gradient.get_data_ptr())[0] = 1.0f;
		}

		loss.backward(top_gradient.get_mdarray<float>());
		//loss.backward();

		optimizer.step();
		optimizer.zero_grad();


		if (!(i % epoch_iterations))
		{
			printf("saving checkpoint...\n");
			net.save_checkpoint("f:\\xfer\\vae_lten.bin");
		}
	}

	ret = net.save_checkpoint("f:\\xfer\\vae_lten.bin");

	return 0;
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
