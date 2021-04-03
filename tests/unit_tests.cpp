#include <iostream>
#include <chrono>
#include "lten.h"

template<typename Dtype>
int Compare(Dtype* A, Dtype* B, uint64_t len, Dtype error = 0)
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

//--------------------------------------------------
// addition test (tensors with the same dimensions)
//--------------------------------------------------
int add_test_1(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 2, 2, 3 }, w_vals, false);


	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x + w;

	a = z[1][0][1][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}


	float z_vals[] = { 7, 6, 6, 7, 8, 10, 12, 11, 6, 5, 4, 10, 1, 7, 11, 15, 16, 8, 5, 4, 5, 6, 9, 13 };
	float grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 8)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//--------------------------------------------------
// addition test (tensors with different dimensions)
//--------------------------------------------------
int add_test_2(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2.0, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5.0, 3, 2, 5 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 1, 2, 1 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = w + x;

	a = z[0][1][0][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 7, 8, 9, 5, 9, 11, 10, 12, 10, 5, 4, 12, 3, 5, 8, 9, 13, 7,	5, 4, 4, 9, 10, 11 };
	float w_grad_vals[] = { 1, 0, 0, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0,0, 0, 1, 0, 0, 0,	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 10)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}


//----------------------------------------------------
// subtraction test (tensors with the same dimensions)
//----------------------------------------------------
int sub_test_1(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 2, 2, 3 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x - w;

	a = z[0][1][1][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { -3, 0, 2, -3, 4, 6, -2, 3, 4, -1, -2, 8, -1, -1, 1, -7, 0, -4, 1, 0, -1, 2, 1, -1 };
	float w_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 8)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//-----------------------------------------------------
// subtraction test (tensors with different dimensions)
//-----------------------------------------------------
int sub_test_2(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2.0, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5.0, 3, 2, 5 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 1, 2, 1 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x - w;

	a = z[1][1][0][2];
	a.backward();


	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { -3, -2, -1, -1, 3, 5, 0, 2, 0, -1, -2, 6, -1, 1, 4,	-1, 3, -3, 1, 0, 0,	-1, 0, 1, };
	float w_grad_vals[] = { 0, 0, -1, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,	0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 0)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//----------------------------------------------------------
// subtraction test (uint8 tensors with the same dimensions)
//----------------------------------------------------------
int sub_test_1_uint8()
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	lten::TensorOps options;
	options.data_type = lten::UINT8;

	uint8_t x_vals[] = { 12, 13, 24, 32, 46, 58, 56, 27, 95, 12, 31, 19, 50, 63, 86, 24, 48, 72, 38, 29, 200, 46, 59, 86 };
	uint8_t w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false, &options);
	w = lten::TensorFromBuffer({ 2, 2, 2, 3 }, w_vals, false, &options);

	z = x - w;

	a = z[0][1][1][2];

	uint8_t z_vals[] = { 7, 10, 22, 27, 44, 56, 49, 23, 94, 9, 28, 18, 49, 59, 81, 13, 40, 66, 36, 27, 197, 44, 55, 79 };
	uint8_t val;

	val = *((uint8_t*)a.get_data_ptr());

	if (val != 18)
	{
		return -1;
	}

	if (Compare((uint8_t*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	return 0;
}

//-----------------------------------------------------------
// subtraction test (uint8 tensors with different dimensions)
//-----------------------------------------------------------
int sub_test_2_uint8()
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	lten::TensorOps options;
	options.data_type = lten::UINT8;

	uint8_t x_vals[] = { 12, 13, 24, 32, 46, 58, 56, 27, 95, 12, 31, 19, 50, 63, 86, 24, 48, 72, 38, 29, 200, 46, 59, 86 };
	uint8_t w_vals[] = { 5, 3, 2, 5 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false, &options);
	w = lten::TensorFromBuffer({ 2, 1, 2, 1 }, w_vals, false, &options);

	z = x - w;

	a = z[1][1][0][2];

	uint8_t z_vals[] = { 7, 8, 19, 29, 43, 55, 51, 22, 90, 9, 28, 16, 48, 61, 84, 19, 43, 67, 36, 27, 198, 41, 54, 81 };
	uint8_t val;

	val = *((uint8_t*)a.get_data_ptr());

	if (val != 198)
	{
		return -1;
	}

	if (Compare((uint8_t*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	return 0;
}


//--------------------------------------------------------------------
// element-wise multiplication test (tensors with the same dimensions)
//--------------------------------------------------------------------
int mul_test_1(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 2, 2, 3 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x * w;

	a = z[1][0][1][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 10, 9, 8, 10, 12, 16, 35, 28, 5, 6, 3, 9, 0, 12, 30, 44, 64, 12, 6, 4, 6, 8, 20, 42 };
	float w_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0 };
	float val;


	val = *((float*)a.get_data_ptr());

	if (val != 12)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//---------------------------------------------------------------------
// element-wise multiplication test (tensors with different dimensions)
//---------------------------------------------------------------------
int mul_test_2(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 1, 2, 1 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x * w;

	a = z[1][1][0][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 10, 15, 20, 6, 18, 24, 25, 35, 25, 6, 3, 27, 2, 6, 12, 20, 40, 10, 6, 4, 4, 20, 25, 30 };
	float w_grad_vals[] = { 0, 0, 2, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0 };
	float val;


	val = *((float*)a.get_data_ptr());

	if (val != 4)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}


//--------------------------------------------------------------------------
// element-wise multiplication test (int32 tensors with the same dimensions)
//-------------------------------------------------------------------------
int mul_test_1_int32()
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;


	lten::TensorOps options;
	options.data_type = lten::INT32;

	int x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	int w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false, &options);
	w = lten::TensorFromBuffer({ 2, 2, 2, 3 }, w_vals, false, &options);


	z = x * w;

	a = z[1][0][1][2];

	int z_vals[] = { 10, 9, 8, 10, 12, 16, 35, 28, 5, 6, 3, 9, 0, 12, 30, 44, 64, 12, 6, 4, 6, 8, 20, 42 };
	int val;

	val = *((int*)a.get_data_ptr());

	if (val != 12)
	{
		return -1;
	}

	if (Compare((int*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	return 0;
}

//--------------------------------------------------------------
// element-wise division test (tensors with the same dimensions)
//--------------------------------------------------------------
int div_test_1(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 1, 4, 3 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x.div(w);

	a = z[0][0][3][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 0.400000f, 1.000000f, 2.000000f, 0.400000f, 3.000000f, 4.000000f, 0.714285731f, 1.750000f,	5.000000f, 0.666666687f, 0.333333343f, 9.000000f,	0.000000f, 0.750000f, 1.200000f, 0.363636374f, 1.000000f, 0.333333343f, 1.500000f, 1.000000f,	0.666666687f, 2.000000f, 1.250000f, 0.857142866f };
	float w_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 9)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//---------------------------------------------------------------
// element-wise division test (tensors with different dimensions)
//---------------------------------------------------------------
int div_test_2(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;
	float error = 0.000001f;

	float x_vals[] = { 2.0, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5.0, 3, 2, 5 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 2, 1, 2, 1 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x.div(w);

	a = z[1][1][0][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 0.400000006f, 0.600000024f, 0.800000012f, 0.666666687f, 2.000000000f, 2.666666746f,1.000000000f, 1.399999976f, 1.000000000f, 0.666666687f, 0.333333343f, 3.000000000f,0.500000000f, 1.500000000f, 3.000000000f, 0.800000012f, 1.600000024f, 0.400000006f,1.500000000f, 1.000000000f, 1.000000000f, 0.800000012f, 1.000000000f, 1.200000048f };
	float w_grad_vals[] = { 0, 0, -0.5, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,	0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 1)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels(), error))
	{
		return -1;
	}

	return 0;
}

//---------------------------
// matrix-multiplication test
//---------------------------
int matmul_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	float w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);
	w = lten::TensorFromBuffer({ 1, 1, 3, 4 }, w_vals, false);

	x.set_autograd(true);
	w.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
		w = w.to(lten::GPU);
	}

	z = x.matmul(w);

	a = z[1][0][3][1];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		w = w.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 20, 24, 37, 26,	30, 42, 70, 42,	44, 44, 74, 58,	21, 35, 38, 23,	12, 24, 39, 18,	38, 34, 70, 54,	21, 19, 26, 25,	36, 40, 61, 46 };
	float w_grad_vals[] = { 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0 };
	float x_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3 };

	float val;

	val = *((float*)a.get_data_ptr());

	if (val != 40)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	if (Compare((float*)w.get_grad_ptr(), w_grad_vals, w.get_numels()))
	{
		return -1;
	}

	return 0;
}

//---------------------------------
// unit8 matrix-multiplication test
//---------------------------------
int matmul_test_uint8()
{
	lten::Tensor x;
	lten::Tensor w;
	lten::Tensor z;
	lten::Tensor a;

	lten::TensorOps options;
	options.data_type = lten::UINT8;

	uint8_t x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };
	uint8_t w_vals[] = { 5, 3, 2, 5, 2, 2, 7, 4, 1, 3, 3, 1, 1, 4, 5, 11, 8, 6, 2, 2, 3, 2, 4, 7 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false, &options);
	w = lten::TensorFromBuffer({ 1, 1, 3, 4 }, w_vals, false, &options);

	x.set_autograd(true);
	w.set_autograd(true);

	z = x.matmul(w);

	a = z[1][0][3][1];

	uint8_t z_vals[] = { 20, 24, 37, 26,	30, 42, 70, 42,	44, 44, 74, 58,	21, 35, 38, 23,	12, 24, 39, 18,	38, 34, 70, 54,	21, 19, 26, 25,	36, 40, 61, 46 };

	uint8_t val;

	val = *((uint8_t*)a.get_data_ptr());

	if (val != 40)
	{
		return -1;
	}

	if (Compare((uint8_t*)z.get_data_ptr(), z_vals, z.get_numels()))
	{
		return -1;
	}


	return 0;
}



//---------
// exp test
//---------
int exp_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;
	lten::Tensor a;
	float error = 0.001f;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = x.exp();

	a = z[1][0][3][2];
	a.backward();


	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 7.389056f, 20.085537f, 54.598148f,	7.389056f, 403.428802f, 2980.958008f,	148.413162f, 1096.633179f, 148.413162f,7.389056f, 2.718282f, 8103.083984f,1.000000f, 20.085537f, 403.428802f,	54.598148f, 2980.958008f, 7.389056f,		20.085537f, 7.389056f, 7.389056f,54.598148f, 148.413162f, 403.428802f };
	float x_grad_vals[] = { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 403.428802f };
	float val;

	val = *((float*)a.get_data_ptr());

	if (fabs(val - 403.428802) > error)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels(), error))
	{
		return -1;
	}

	return 0;

}


//---------
// log test
//---------
int log_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;
	lten::Tensor a;
	float error = 0.000001f;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = x.log();

	a = z[1][0][3][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 0.693147f, 1.098612f, 1.386294f, 0.693147f, 1.791759f, 2.079442f, 1.609438f, 1.945910f, 1.609438f,0.693147f, 0.000000f, 2.197225f,	0.000000f, 1.098612f, 1.791759f, 1.386294f, 2.079442f, 0.693147f, 1.098612f, 0.693147f, 0.693147f, 1.386294f, 1.609438f, 1.791759f };
	float x_grad_vals[] = { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.166666672f };
	float val;

	val = *((float*)a.get_data_ptr());

	if (fabs(val - 1.79175949) > error)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;

}


//-------------
// sigmoid test
//-------------
int sig_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;
	lten::Tensor a;
	float error = 0.000001f;

	float x_vals[] = { -2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, -3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = x.sig();

	a = z[1][0][3][2];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 0.119203f, 0.952574f, 0.982014f, 0.880797f, 0.997527f, 0.999665f, 0.993307f, 0.999089f, 0.993307f, 0.880797f, 0.731059f, 0.999877f, 0.5f, 0.0474259f, 0.997527f, 0.982014f, 0.999665f, 0.880797f, 0.952574f, 0.880797f, 0.880797f, 0.982014f, 0.993307f, 0.997527f };
	float x_grad_vals[] = { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.00246646581f };
	float val;

	val = *((float*)a.get_data_ptr());

	if (fabs(val - 0.997527421f) > error)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;

}


//----------
// tanh test
//----------
int tanh_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;
	lten::Tensor a;
	float error = 0.000001f;

	float x_vals[] = { -2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, -3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = x.tanh();

	a = z[0][0][0][2];
	a.backward();

	//std::cout << z << std::endl;
	//std::cout << *(x.get_gradients_mdarray<float>()) << "\n";

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { -0.964028f, 0.995055f, 0.999329f, 0.964028f, 0.999988f, 1.0f, 0.999909f, 0.999998f, 0.999909f, 0.964028f, 0.761594f, 1.0f, 0.0f, -0.995055f, 0.999988f, 0.999329f, 1.0f, 0.964028f, 0.995055f, 0.964028f, 0.964028f, 0.999329f, 0.999909f, 0.999988f };
	float x_grad_vals[] = { 0.000000f, 0.000000f, 0.00134086609f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,	0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f };
	float val;

	val = *((float*)a.get_data_ptr());

	if (fabs(val - 0.999329329f) > error)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels(), error))
	{
		return -1;
	}

	return 0;

}


//---------------------------
// scalar multiplication test
//---------------------------
int scalar_mul(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;
	lten::Tensor a;
	float error = 0.000001f;

	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = 87 * x;

	a = z[1][0][2][1];
	a.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 174, 261, 348, 174, 522, 696, 435, 609, 435, 174, 87, 783,	87, 261, 522, 348, 696, 174, 261, 174, 174, 348, 435, 522 };
	float x_grad_vals[] = { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,	0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,	0.000000, 87.000000, 0.000000, 0.000000, 0.000000, 0.000000 };
	float val;

	val = *((float*)a.get_data_ptr());

	if (fabs(val - 174) > error)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;

}


//---------
// max test
//---------
int max_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;

	float x_vals[] = { 9, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 3 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = x.max();

	z.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
	}

	float x_grad_vals[] = { 0.500000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.500000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000 };
	float val;

	val = *((float*)z.get_data_ptr());
	if (val != 9)
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//---------
// sum test
//---------
int sum_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;

	float x_vals[] = { 2.0, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 1, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 2, 2, 3 }, x_vals, false);

	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(lten::GPU);
	}

	z = x.sum();

	z.backward();

	if (run_on_gpu)
	{
		x = x.to(lten::CPU);
		z = z.to(lten::CPU);
	}
	
	float x_grad_vals[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	float val;

	val = *((float*)z.get_data_ptr());
	if (val != 100)
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels()))
	{
		return -1;
	}


	lten::Tensor a;
	float z_vals[] = { 11, 17, 26,	12, 18, 16 };
	float x_grad_vals_2[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals, false);
	x.set_autograd(true);
	z = x.sum(2);

	a = z[0][1][0][1];
	a.backward();

	val = *((float*)a.get_data_ptr());
	if (val != 18)
	{
		return -1;
	}

	if (Compare((float*)x.get_grad_ptr(), x_grad_vals_2, x.get_numels()))
	{
		return -1;
	}

	return 0;
}

//--------------------------
// FullyConnected layer test
//--------------------------
int fc_test(bool run_on_gpu)
{
	lten::Tensor x;
	lten::Tensor z;
	lten::Tensor a;
	lten::TensorOps ops;
	uint64_t len;
	uint64_t i;
	float val;
	float error = 0.001f;
	lten::device device = lten::GPU;

	float test_wts[] = { 0.0900306f, 0.2447285f, 0.6652409f, 0.0021785f, 0.1189432f, 0.8788782f, 0.1065070f, 0.7869860f, 0.1065070f, 0.0009107f, 0.0003350f, 0.9987543f,	0.0023556f, 0.0473142f, 0.9503303f, 0.0179425f, 0.9796292f, 0.0024283f, 0.5761169f, 0.2119416f, 0.2119416f, 0.0900306f, 0.2447285f, 0.6652409f };
	float x_vals[] = { 2, 3, 4, 2, 6, 8, 5, 7, 5, 2, 1, 9, 0, 3, 6, 4, 8, 2, 3, 2, 2, 4, 5, 6 };

	x = lten::TensorFromBuffer({ 2, 1, 4, 3 }, x_vals);
	x.set_autograd(true);

	if (run_on_gpu)
	{
		x = x.to(device);
	}


	lten::FullyConnected fc(3, 8);
	fc.init();
	len = fc.get_weights()->get_numels();

	for (i = 0; i < len; i++)
	{
		((float*)(fc.get_weights()->get_data_ptr()))[i] = test_wts[i];
	}

	if (run_on_gpu)
	{
		fc.to(device);
	}

	z = fc.forward(x);

	a = z[1][0][3][4];
	a.backward();

	if (run_on_gpu)
	{
		z = z.to(lten::CPU);
		a = a.to(lten::CPU);
	}

	float z_vals[] = { 3.57521f, 3.8767f, 3.0f, 3.99784f, 3.94798f, 2.98449f, 2.63583f, 3.57521f, 6.97036f, 7.74904f, 5.78699f, 7.99387f, 7.89124f, 5.93309f, 4.11942f, 6.97036f, 5.48946f, 5.23789f, 6.57397f, 5.00067f, 5.09463f, 6.95926f, 5.42388f, 5.48946f, 6.41196f, 8.0332f, 1.95856f, 8.99094f, 8.605f, 1.03737f, 3.27165f, 6.41196f, 4.72563f, 5.6301f, 3.0f, 5.99353f, 5.84392f, 2.95346f, 1.90747f, 4.72563f, 3.64843f, 2.71802f, 6.93493f, 2.00383f, 2.2886f, 7.91366f, 4.42388f, 3.64843f, 2.09003f, 2.00218f, 2.10651f, 2.00091f, 2.00236f, 2.01794f, 2.57612f, 2.09003f, 5.57521f, 5.8767f, 5.0f, 5.99784f, 5.94798f, 4.98449f, 4.63583f, 5.57521f, };
	float x_grad_vals[] = { 0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0, 0.0023556f, 0.0473142f, 0.95033f };
	float wt_grad_vals[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.0f, 5.0f, 6.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	val = *((float*)a.get_data_ptr());

	if (fabs(val - 5.94797564f) > error)
	{
		return -1;
	}

	if (Compare((float*)z.get_data_ptr(), z_vals, z.get_numels(), error))
	{
		return -1;
	}

	if (run_on_gpu)
	{
		float* x_grad = new float[x.get_numels()];
		CopyDataFromGPU(x_grad, x.get_grad_ptr(), x.get_numels() * sizeof(float));

		if (Compare(x_grad, x_grad_vals, x.get_numels(), error))
		{
			return -1;
		}

		float* wt_grad = new float[fc.get_weights()->get_numels()];
		CopyDataFromGPU(wt_grad, fc.get_weights()->get_grad_ptr(), fc.get_weights()->get_numels() * sizeof(float));

		if (Compare(wt_grad, wt_grad_vals, fc.get_weights()->get_numels(), error))
		{
			return -1;
		}

		delete wt_grad;
		delete x_grad;
	}
	else
	{
		if (Compare((float*)x.get_grad_ptr(), x_grad_vals, x.get_numels(), error))
		{
			return -1;
		}

		if (Compare((float*)fc.get_weights()->get_grad_ptr(), wt_grad_vals, fc.get_weights()->get_numels(), error))
		{
			return -1;
		}
	}

	return 0;
}

