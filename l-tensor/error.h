#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <stdint.h>

namespace lten {
	class ExceptionInfo : public std::exception
	{
	public:
		ExceptionInfo(const char* msg, const char* file, const char* function, const char* line)
		{
			msg_ = std::string("File:") + std::string(file) + std::string(" Line:") + std::string(line) + std::string(" Function:") + std::string(function) + std::string(" Msg:") + msg;
		}

		~ExceptionInfo()
		{

		}

		virtual const char* what() const noexcept
		{
			return msg_.c_str();
		}

	private:
		std::string msg_;
	};
}

void ltenFail(const char* msg, const char* file, const char* line, const char* func);
void ltenErrMsg(const char* msg, const char *file, const char *function, int line);

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

#ifdef USE_CUDA
#include "cudnn.h"
#include "cublas.h"
#include "cublas_v2.h"


void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line);
void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line);

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
#define cublasErrCheck(stat) {cublasErrCheck_((stat), __FILE__, __LINE__); }

#define LTEN_CUBLAS_CHECK(EXPR)\
	cublasStatus_t status = EXPR;\
	cublasErrCheck(status);\
	LTEN_CHECK(status == CUBLAS_STATUS_SUCCESS, "BLAS error calling "#EXPR);

#define LTEN_CUBLAS_CHECK_2(EXPR)\
	status = EXPR;\
	cublasErrCheck(status);\
	LTEN_CHECK(status == CUBLAS_STATUS_SUCCESS, "BLAS error calling "#EXPR);

#define LTEN_CUDA_CHECK(EXPR)\
	cudaError_t status = EXPR;\
	cudaErrCheck(status);\
	LTEN_CHECK(status == cudaSuccess, "CUDA error calling "#EXPR);

#define LTEN_CUDNN_CHECK(EXPR)\
	cudnnStatus_t status = EXPR;\
	cudnnErrCheck(status);\
	LTEN_CHECK(status == CUDNN_STATUS_SUCCESS, "CUDNN error calling "#EXPR);

#define LTEN_CUDNN_CHECK_2(EXPR)\
	status = EXPR;\
	cudnnErrCheck(status);\
	LTEN_CHECK(status == CUDNN_STATUS_SUCCESS, "CUDNN error calling "#EXPR);
#endif


#define LTEN_CHECK(cond, msg)\
	if(!(cond)){ \
		ltenFail(msg, __FILE__, STRINGIFY(__LINE__), __func__); \
	}

#define LTEN_ERR_CHECK(EXPR)\
	int ret_val = EXPR;\
	LTEN_CHECK(ret_val == 0, "lten error calling "#EXPR );

#define LTEN_BOOL_ERR_CHECK(EXPR)\
	bool ret_val = EXPR;\
	LTEN_CHECK(ret_val == true, "lten error calling "#EXPR );

#define LTEN_ERR(msg)\
	ltenErrMsg(msg, __FILE__, __func__, __LINE__);\
	LTEN_CHECK(false, msg );

#endif
