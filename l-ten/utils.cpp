#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include "lten.h"
#include "error.h"
#include "utils.h"

void* BlockRealloc(void* current_block_ptr, int current_size, int new_size)
{
	unsigned char* reallocated_block_ptr;

	reallocated_block_ptr = new unsigned char[new_size];

	memcpy(reallocated_block_ptr, current_block_ptr, current_size);

	delete current_block_ptr;

	return reallocated_block_ptr;
}


int64_t GetFileSize(FILE* stream)
{
	int64_t size;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	_fseeki64(stream, 0, SEEK_END);
	size = _ftelli64(stream);
#else
	fseek(stream, 0, SEEK_END);
	size = ftell(stream);
#endif

	fseek(stream, 0, SEEK_SET);

	return size;
}


int ReadDataFromFile(const char* file_name, void** pp_data, size_t* data_size)
{
	int ret;
	FILE* stream;
	size_t bytes;
	size_t bytes_read;
	size_t file_size;
	char* char_data;
	
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	errno_t err;
	err = fopen_s(&stream, file_name, "rb");
	if (err)
	{
		ret = -1;
		goto Exit;
	}
#else
	stream = fopen(file_name, "rb");
	if (!stream)
	{
		ret = -1;
		goto Exit;
	}
#endif
	file_size = GetFileSize(stream);
	char_data = new char[file_size];
	if (!char_data)
	{
		assert(0);
		ret = -1;
		goto Exit;
	}

	*pp_data = static_cast<void*>(char_data);

	bytes_read = 0;

	while (bytes_read < file_size)
	{
		bytes = fread(char_data, 1, file_size, stream);
		bytes_read += bytes;
		char_data += bytes;
	}

	*data_size = bytes_read;

	fclose(stream);

	ret = 0;

Exit:
	return ret;

}

int WriteDataToFile(const char* file_name, void* data, size_t data_size)
{
	int ret;
	FILE* stream;
	size_t bytes;
	size_t bytes_written;
	char* char_data;
	
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	errno_t err;
	err = fopen_s(&stream, file_name, "wb");
	if (err)
	{
		ret = -1;
		goto Exit;
	}
#else
	stream = fopen(file_name, "wb");
	if (!stream)
	{
		ret = -1;
		goto Exit;
	}
#endif
	bytes_written = 0;
	char_data = (char*)data;

	while (bytes_written < data_size)
	{
		bytes = fwrite(char_data, sizeof(char), data_size - bytes_written, stream);
		bytes_written += bytes;
		char_data += bytes;
	}

	fclose(stream);

	ret = 0;
Exit:

	return ret;

}

void ReshapeDims(const uint64_t* current_dims_ptr, int current_size, uint64_t* new_dims_ptr, int new_size)
{
	int i;

	memmove(&new_dims_ptr[new_size - current_size], current_dims_ptr, sizeof(uint64_t) * current_size);

	for (i = 0; i < new_size - current_size; i++)
	{
		new_dims_ptr[i] = 1;
	}
}


// reshape current_dims by inserting 1's into locations specified by unsqueeze_dims
// e.g. [20, 24, 8] becomes [20, 1, 24, 1, 8]
// if unsqueeze_dims is [1,3]
void UnsqueezeDims(const uint64_t* current_dims, int current_ndims, uint32_t* unsqueeze_dims, int unsqueeze_ndims, uint64_t* new_dims)
{
	uint32_t ndims;
	int i;
	int index;

	ndims = current_ndims + unsqueeze_ndims;
	memset(new_dims, 0, sizeof(uint64_t) * ndims);

	for (i = 0; i < unsqueeze_ndims; i++)
	{
		new_dims[unsqueeze_dims[i]] = 1;
	}

	index = 0;
	for (i = 0; i < ndims; i++)
	{
		if (new_dims[i] == 0)
		{
			new_dims[i] = current_dims[index++];
		}
	}

}


void GetMaxDims(uint64_t* dims_1, uint64_t* dims_2, uint64_t* dims_max, int ndims)
{
	int i;

	for (i = 0; i < ndims; i++)
	{
		dims_max[i] = std::max(dims_1[i], dims_2[i]);
	}
}

void GetPermutationStridesAndeDims(const uint64_t* src_dims, uint64_t* dst_dims, uint64_t* dst_strides, const uint32_t* permutations, int ndims)
{
	int i;
	uint64_t numels;
	uint64_t dst_dims_[MAX_DIMS];
	uint64_t dst_strides_[MAX_DIMS];

	if (!dst_dims)
	{
		dst_dims = dst_dims_;
	}
	if (!dst_strides)
	{
		dst_strides = dst_strides_;
	}

	//
	// generate dst dims
	//
	for (i = 0; i < ndims; i++)
	{
		dst_dims[i] = src_dims[permutations[i]];
	}

	//
	// generate dst strides
	//
	numels = 1;
	for (i = ndims - 1; i >= 0; i--)
	{
		dst_strides[i] = numels;
		numels *= dst_dims[i];
	}

}


void CoordinatesFromIndex(uint64_t index, const uint64_t* dims, const uint64_t* strides, uint64_t* coordinates, int ndims)
{
	int i;
	uint64_t idx;

	if (ndims == 1)
	{
		coordinates[0] = index;
		return;
	}

	idx = index;

	coordinates[0] = idx / strides[0];
	for (i = 1; i < ndims - 1; i++)
	{
		idx = idx % strides[i - 1];
		coordinates[i] = idx / strides[i];
	}
	coordinates[i] = idx % dims[i];


	/*
	//sanity check
	idx = 0;
	for (int j = 0; j < ndims; j++)
	{
		idx += coordinates[j] * strides[j];
	}
	assert(idx == index);
	*/
}


void GetMinMax(float* data, uint64_t len, float* min_val, float* max_val)
{
	uint64_t i;
	float minimum;
	float maximum;

	if (!data || !min_val || !max_val)
	{
		LTEN_ERR("Invalid argument");;
	}

	minimum = data[0];
	maximum = data[0];

	for (i = 0; i < len; i++)
	{
		if (data[i] > maximum)
		{
			maximum = data[i];
		}
		if (data[i] < minimum)
		{
			minimum = data[i];
		}
	}

	*min_val = minimum;
	*max_val = maximum;
}

void Quantize(float* data, int* q_data, uint64_t len, QuantizationParams* qp)
{
	uint64_t i;
	float val;
	int lowest = (std::numeric_limits<int>::min)();
	int highest = (std::numeric_limits<int>::max)();

	for (i = 0; i < len; i++)
	{
		val = qp->zero_point + data[i] / qp->scale;
		val = std::max((float)lowest, std::min((float)highest, val));
		q_data[i] = (int)(round(val));
	}
}

void Quantize(float* data, uint8_t* q_data, uint64_t len, QuantizationParams* qp)
{
	uint64_t i;
	float val;

	for (i = 0; i < len; i++)
	{
		val = qp->zero_point + data[i] / qp->scale;
		val = std::max(0.f, std::min(255.f, val));
		q_data[i] = (uint8_t)(round(val));
	}
}

void Dequantize(float* data, uint8_t* q_data, uint64_t len, QuantizationParams* qp)
{
	uint64_t i;

	for (i = 0; i < len; i++)
	{
		data[i] = qp->scale * (q_data[i] - qp->zero_point);
	}
}

void ComputeQuantizationParams(float min_val, float max_val, QuantizationParams* qp)
{
	float q_min = 0;
	float q_max = 255;
	float zero_point;


	// make sure 0 is included
	min_val = std::min(min_val, 0.f);
	max_val = std::max(max_val, 0.f);


	qp->scale = (max_val - min_val) / (q_max - q_min);

	zero_point = q_min - min_val / qp->scale;

	qp->zero_point = (uint8_t)(std::round(zero_point));

	assert(zero_point >= q_min);
	assert(zero_point <= q_max);
}



template<typename Dtype>
void FillBuffer(Dtype* data_ptr, uint64_t len, Dtype value)
{
	uint64_t i;

	for (i = 0; i < len; i++)
	{
		data_ptr[i] = value;
	}
}

void GetStrides(const int* dims, int* strides, int ndims)
{
	int i;
	int numels;

	numels = 1;
	for (i = ndims - 1; i >= 0; i--)
	{
		strides[i] = numels;
		numels *= dims[i];
	}
}

void GetStrides(const uint32_t* dims, uint32_t* strides, int ndims)
{
	int i;
	uint32_t numels;

	numels = 1;
	for (i = ndims - 1; i >= 0; i--)
	{
		strides[i] = numels;
		numels *= dims[i];
	}
}

void GetStrides(const uint64_t* dims, uint64_t* strides, int ndims)
{
	int i;
	uint64_t numels;

	numels = 1;
	for (i = ndims - 1; i >= 0; i--)
	{
		strides[i] = numels;
		numels *= dims[i];
	}
}


uint32_t GetNextPowerOf2(int number)
{
	float log2;

	log2 = log((float)number) / log(2.0f);
	log2 = ceil(log2);

	return (uint32_t)pow(2.0f, log2);
}

void ltenFail(const char* msg, const char* file, const char* line, const char* func)
{
	throw lten::ExceptionInfo(msg, file, func, line);
}


void* cpu_alloc(uint64_t size)
{
	return ::new char[size];
}

void cpu_free(void* memory)
{
	::delete memory;
}

void ltenErrMsg(const char* msg, const char *file, const char *function, int line)
{
	fprintf(stderr, "[Error] File: %s Function: %s Line:%d Msg: %s\n", file, function, line, msg);
}


bool check_broadcast_required(const uint64_t* dims_a, const uint64_t* dims_b, uint32_t ndims, bool transpose_a, bool transpose_b, uint64_t* dims_result, bool mat_mul_check)
{
	bool broadcast_required;
	uint32_t i;
	uint64_t dims_A[MAX_DIMS];
	uint64_t dims_B[MAX_DIMS];

	const uint64_t* dims_a_ptr;
	const uint64_t* dims_b_ptr;

	uint64_t temp;

	broadcast_required = false;

	if (transpose_a)
	{
		memcpy(dims_A, dims_a, sizeof(uint64_t) * ndims);
		temp = dims_A[ndims - 1];
		dims_A[ndims - 1] = dims_A[ndims - 2];
		dims_A[ndims - 2] = temp;

		dims_a_ptr = dims_A;
	}
	else
	{
		dims_a_ptr = dims_a;
	}


	if (transpose_b)
	{
		memcpy(dims_B, dims_b, sizeof(uint64_t) * ndims);
		temp = dims_B[ndims - 1];
		dims_B[ndims - 1] = dims_B[ndims - 2];
		dims_B[ndims - 2] = temp;

		dims_b_ptr = dims_B;
	}
	else
	{
		dims_b_ptr = dims_b;
	}


	if (mat_mul_check)
	{
		if (dims_a_ptr[ndims - 1] != dims_b_ptr[ndims - 2]) // check matrix dimension compatibility
		{
			LTEN_ERR("Tensors must have compatiple dimensions");
		}

		if (dims_result)
		{
			dims_result[ndims - 1] = dims_b_ptr[ndims - 1];
			dims_result[ndims - 2] = dims_a_ptr[ndims - 2];
		}

		ndims -= 2; // W & H may not match for matrix multiplication
	}

	for (i = 0; i < ndims; i++)
	{
		if (dims_a_ptr[i] != dims_b_ptr[i])
		{
			if (dims_a_ptr[i] != 1 && dims_b_ptr[i] != 1)
			{
				LTEN_ERR("MultiDimArrays must have compatiple dimensions");
			}

			broadcast_required = true;
		}

		if (dims_result)
		{
			dims_result[i] = std::max(dims_a_ptr[i], dims_b_ptr[i]);
		}
	}

	return broadcast_required;
}


template void FillBuffer<float>(float* data_ptr, uint64_t len, float value);
template void FillBuffer<int>(int* data_ptr, uint64_t len, int value);
template void FillBuffer<uint8_t>(uint8_t* data_ptr, uint64_t len, uint8_t value);



