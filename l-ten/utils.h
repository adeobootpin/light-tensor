#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include "math_fns.h"

template<typename Dtype>
void FillBuffer(Dtype* data_ptr, uint64_t len, Dtype value);


void ReshapeDims(const uint64_t* current_dims_ptr, int current_size, uint64_t* new_dims_ptr, int new_size);



void GetMaxDims(uint64_t* dims_1, uint64_t* dims_2, uint64_t* dims_max, int ndims);
void CoordinatesFromIndex(uint64_t index, const uint64_t* dims, const uint64_t* strides, uint64_t* coordinates, int ndims);

void* BlockRealloc(void* current_block_ptr, int current_size, int new_size);

int ReadDataFromFile(const char* file_name, void** pp_data, size_t* data_size);
int WriteDataToFile(const char* file_name, void* data, size_t data_size);

void GetMinMax(float* data, uint64_t len, float* min_val, float* max_val);
void Quantize(float* data, int* q_data, uint64_t len, QuantizationParams* qp);
void Quantize(float* data, uint8_t* q_data, uint64_t len, QuantizationParams* qp);
void Dequantize(float* data, uint8_t* q_data, uint64_t len, QuantizationParams* qp);
void ComputeQuantizationParams(float min_val, float max_val, QuantizationParams* qp);

void* cpu_alloc(uint64_t size);
void cpu_free(void* memory);
void* gpu_alloc(uint64_t size);
void gpu_free(void* memory);


int AllocateMemoryOnGPU(void** memory_ptr_addr, uint64_t size, bool zero_memory);
void ZeroMemoryOnGPU(void* memory, size_t size);
void FreeMemoryOnGPU(void* memory);
int CopyDataToGPU(void* gpu, void* host, size_t size);
int CopyDataFromGPU(void* host, void* gpu, size_t size);
int GPUToGPUCopy(void* dst, void* src, size_t size);
int GetDevice(int* device);
void GetStrides(int* dims, int* strides, int ndims);
int GetNextPowerOf2(int number);

#endif // UTILS_H
