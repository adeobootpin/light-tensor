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
	at = at.to(lten::GPU);




	lten::Tensor dt;
	lten::Tensor ct;

	dt = at.to(lten::CPU);
	bt = at.transpose(1, 2);

	ct = dt.transpose(1, 2);
	bt = bt.to(lten::CPU);

	int ret = Comparexx<float>((float*)bt.get_data_ptr(), (float*)ct.get_data_ptr(), (int)ct.get_numels(), 0);
	if (!ret)
	{
		printf("all correct sir!\n");
	}
	return;









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
