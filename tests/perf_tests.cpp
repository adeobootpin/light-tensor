#include <iostream>
#include <chrono>
#include "lten.h"


int conv3d_perf_test();
void mean_test()
{
	conv3d_perf_test();


	int i;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	lten::Tensor a;
	lten::Tensor b;

	uint64_t numels = 8 * 1568 * 768;
	float* data;

	data = new float[numels];
	for (i = 0; i < numels; i++)
	{
		data[i] = (rand() % 1000) * 0.0001f;
	}

	a = lten::TensorFromBuffer({ 8, 1568, 768 }, data, false);
	a = a.to(lten::GPU);

	for (i = 0; i < 100000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		b = a.mean();
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten MeanTest [duration: %f sec]\n", nseconds);
}


template<typename Dtype>
int Comparexx(Dtype* A, Dtype* B, uint64_t len, Dtype error = 0)
{
	uint64_t i;

	for (i = 0; i < len; i++)
	{
		if (fabs(A[i] - B[i]) > error)
		{
			return -1;
		}
	}
	return 0;
}

void mean_test2()
{
	int i;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	lten::Tensor a;
	lten::Tensor b;

	a = lten::RandomTensor({ 8, 448, 768 });

	uint64_t len = a.get_numels();
	float* data = (float*)a.get_data_ptr();

	for (uint64_t j = 0; j < len; j++)
	{
		((float*)a.get_data_ptr())[j] = 1.0f;
	}
	
	a = a.to(lten::GPU);

	uint32_t axes[] = { 1 };

	for (i = 0; i < 1000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		b = a.mean(axes, 1);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten MeanTest [duration: %f sec]\n", nseconds);
}


void var_test()
{
	int i;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	lten::Tensor a;
	lten::Tensor b;

	//a = lten::RandomTensor({ 16, 256, 8, 768 });
	a = lten::RandomTensor({ 1, 1, 1, 13 });
	a = a.to(lten::GPU);

	uint32_t axes[] = { 3 };

	//for (i = 0; i < 100000; i++)
	for (i = 0; i < 1; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		b = a.var(axes, 1);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten var test [duration: %f sec]\n", nseconds);
}

int layerNorm_test()
{
	int i;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	lten::Tensor a;
	lten::Tensor b;


	float* data;

	data = new float[16 * 1568 * 768];
	for (i = 0; i < 16 * 1568 * 768; i++)
	{
		data[i] = (rand() % 1000) * 0.0001f;
	}

	a = lten::TensorFromBuffer({ 1, 4, 1568, 768 }, data, false);

	a = a.to(lten::GPU);


	lten::LayerNorm ln(768, true);
	ln.init();
	ln.to(lten::GPU);


	for (i = 0; i < 100000; i++)
	//for (i = 0; i < 1; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		b = ln.forward(a);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten LayerNormTest [duration: %f sec]\n", nseconds);

	return 0;
}

void maxPool3d_test()
{
	int i;

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	lten::Tensor x;
	lten::Tensor y;

	int kernel_h = 2;
	int kernel_w = 2;
	int kernel_c = 2;
	int stride_h = 1;
	int stride_w = 1;
	int stride_c = 1;
	int pad_h = 0;
	int pad_w = 0;
	int pad_c = 0;

	x = lten::RandomTensor({ 2, 3, 16, 224, 224 });

	lten::pooling3d_CUDNN pool(0, kernel_h, kernel_w, kernel_c, pad_h, pad_w, pad_c, stride_h, stride_w, stride_c);
	pool.init();

	x = x.to(lten::GPU);

	for (i = 0; i < 10000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		y = pool.forward(x);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten maxPool3d_test [duration: %f sec]\n", nseconds);

}

void repeat_test()
{
	int i;
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;


	lten::Tensor a;
	lten::Tensor b;

	uint32_t repeats[MAX_DIMS] = { 1, 8, 1 };

	a = lten::RandomTensor({ 8, 3136, 96 });
	a = a.to(lten::GPU);

	for (i = 0; i < 1000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		b = a.repeat(repeats, 5);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten repeat [duration: %f sec]\n", nseconds);
}

void repeat_interleave_test()
{
	int i;

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	uint32_t repeats[MAX_DIMS] = { 3136, 3136, 3136, 3136, 3136, 3136, 3136, 3136 };
	int nrepeats = 8;
	uint32_t* scratch;
	scratch = new uint32_t[nrepeats + 1];

	lten::Tensor x;
	lten::Tensor y;

	x = lten::RandomTensor({ 1, 8, 96 });
	x = x.to(lten::GPU);


	for (i = 0; i < 100000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}
		y = x.repeat_interleave(repeats, nrepeats, 1, scratch);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten repeat interleave [duration: %f sec]\n", nseconds);

}

void index_test()
{
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	int* indices;
	float* data;
	int i;
	int j;

	lten::Tensor x;
	lten::Tensor y;
	lten::Tensor indexx;
	lten::TensorOps op;

	srand(0);

	data = new float[111 * 96];
	for (i = 0; i < 111 * 96; i++)
	{
		data[i] = (rand() % 1000) * 0.00001f;
	}

	indices = new int[56 * 14];

	for (i = 0; i < 56; i++)
	{
		for (j = 0; j < 14; j++)
		{
			indices[i * 14 + j] = 52 + i - j * 4;
		}
	}

	//x = torch::from_blob(data, { 111, 96 }, torch::requires_grad());
	//indexx = torch::from_blob(indices, { 56, 14 }, torch::kInt32);
	//indexx = indexx.to(torch::kLong);

	//x = x.to(device);
	//indexx = indexx.to(device);

	x = lten::RandomTensor({ 111, 96 });
	x = x.to(lten::GPU);

	op.data_type = lten::INT32;
	indexx = lten::TensorFromBuffer({ 56, 14 }, indices, false, &op);
	indexx = indexx.to(lten::GPU);

	//for (i = 0; i < 250000; i++)
	for (i = 0; i < 1; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		y = x.index({ indexx });
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten::index [duration: %f sec]\n", nseconds);

}

int ReadDataFromFile(const char* file_name, void** pp_data, size_t* data_size);
int conv3d_perf_test()
{
	/*
	lten::Tensor a;
	lten::Tensor b;

	lten::Tensor a_gpu;

	int i;
	uint64_t numels = 1;
	float* a_buffer = new float[numels];
	a = lten::TensorFromBuffer({ numels }, a_buffer, false);
	float sum = 0;
	int isum = 0;

	for (i = 0; i < numels; i++)
	{
		a_buffer[i] = (rand() % 1000) / 1000.0f;
		a_buffer[i] = i + 1.0f;
		sum += a_buffer[i];
		isum += i + 1;
	}

	
	float val[32];
	for (int thread = 0; thread < 32; thread++)
	{
		val[thread] = 0;
		for (i = thread; i < numels; i += 32)
		{
			val[thread] += a_buffer[i];
		}
	}
	float sum2 = 0;
	for (int thread = 0; thread < 32; thread++)
	{
		sum2 += val[thread];
	}
	

	
	b = a.mean(0);
	std::cout << b << std::endl;

	a_gpu = a.to(lten::GPU);
	b = a_gpu.mean();
	b = b.to(lten::CPU);
	std::cout << b << std::endl;
	printf("sum: %f", sum);
	sum /= numels;
	*/

	lten::conv3d_CUDNN* conv3d;

	conv3d = new lten::conv3d_CUDNN(2, 3, 16, 224, 224, 96, true, 7, 7, 3, 3, 3, 1, 4, 4, 2);

	conv3d->init();

	size_t cbBytes;
	void* input_scratch;

	ReadDataFromFile("e:/xfer/input.bin", &input_scratch, &cbBytes);

	lten::Tensor x;
	lten::Tensor y;
	x = lten::TensorFromBuffer({ 2, 3, 16, 224, 224 }, input_scratch, false);

	x = x.to(lten::GPU);

	y = conv3d->forward(x);

	//bool lten::conv3d_CUDNN c3d;
	//conv1 = register_module("conv1", lten::Conv2d(channels_in, channels_out, false, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

	delete conv3d;
	return 0;
}




void transpose_test()
{
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	int i;

	lten::Tensor at;
	lten::Tensor bt;

	at = lten::RandomTensor({ 4, 32, 25088 }, nullptr);
	//at = lten::RandomTensor({ 7, 13, 47 }, nullptr);
	at = at.to(lten::GPU);



	/*
	lten::Tensor dt;
	lten::Tensor ct;

	dt = at.to(lten::CPU);
	bt = at.transpose(1, 2);

	ct = dt.transpose(1, 2);
	bt = bt.to(lten::CPU);

	int ret = Comparexx<float>((float*)bt.get_data_ptr(), (float*)ct.get_data_ptr(), (int)ct.get_numels(), 0);
	if (ret)
	{
		printf("all correct sir!\n");
	}
	return;
	*/


	for (i = 0; i < 1000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}
		bt = at.transpose(1, 2);
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten::transpose [duration: %f sec]\n", nseconds);

}


void transpose_backwards_test()
{
	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;

	int i;

	lten::Tensor at;
	lten::Tensor bt;
	lten::Tensor top_gradient;

	at = lten::RandomTensor({ 2, 128, 256 * 256 }, nullptr);
	//at = lten::RandomTensor({ 256, 128, 64 }, nullptr);
	at.set_autograd(true);
	at = at.to(lten::GPU);


	bt = at.transpose(0, 2);
	top_gradient = lten::RandomTensor(bt.get_sizes(), bt.get_ndims());

	top_gradient = top_gradient.to(lten::GPU);



	for (i = 0; i < 10000; i++)
	{
		if (i == 10)
		{
			clock_begin = std::chrono::steady_clock::now();
		}

		bt.backward(top_gradient.get_mdarray<float>());
	}

	cudaDeviceSynchronize();

	clock_end = std::chrono::steady_clock::now();
	time_span = clock_end - clock_begin;
	nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	printf("lten::transpose backwards [duration: %f sec]\n", nseconds);
}
