#ifndef CUMAT_H_
#define CUMAT_H_

#include "mat_ones_kernel.h"
#include "mat_mul_elementwise_kernel.h"
#include "mat_mul_plus_elementwise_kernel.h"

#include <iostream>
#include <cmath>
#include <random>
#include <sstream>
#include <map>
#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "/usr/local/cuda-9.1/include/cublas_v2.h"
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

inline int IDX2F(int i, int j, int ld){
    return j * ld + i;
}

inline void FatalError(std::string s){
    std::stringstream _where, _message;
    _where << __FILE__ << ':' << __LINE__;
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;
    std::cerr << _message.str() << "\nAborting...\n";
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

// inline void checkCUDNN(int  status){
//     std::stringstream _error;
//     if(status != CUDNN_STATUS_SUCCESS){
// 	_error << "CUDNN failure\nError:" << cudnnGetErrorString(status);
// 	FatalError(_error.str());
//     }
// }

inline void checkCublasErrors(int status){
    std::stringstream _error;
    if(status != 0){
	_error << "Cublas failure\nError code " << status;
	FatalError(_error.str());
    }
}

class MallocCounter{
private:
    int num_ = 0;
public:
    inline void up(){ num_++;}
    inline void down(){ num_--;}
    inline int get(){ return num_;}
};

extern MallocCounter mallocCounter;

class cuMat{
private:
    friend class boost::serialization::access;
    template<class Archive> void Serialize(Archive &ar, const unsigned int version){
	ar & m_host_array_;
	ar & rows_;
	ar & cols_;
    }

    float *m_device_ = nullptr;
    float *m_host_ = nullptr;
    std::vector<float> m_host_array_;
    int rows_ = 0;
    int cols_ = 0;

    cublasHandle_t cuda_handle_;
    
public:

    //accesser
    inline int row() const { return rows_; }
    
    inline int col() const { return cols_; }

    //    inline float* m_device_ const { return m_device_; }

    //    inline float *m_host_ const { return m_host_;}

    //    inline cublasHandle_t cuda_handle_ const { return cuda_handle_;}
    
    cuMat(){
	rows_ = cols_ = 0;
	cublasCreate(&cuda_handle_);
	cudaThreadSynchronize();
    }

    cuMat(const int rows, const int cols){
	cublasCreate(&cuda_handle_);
	cudaThreadSynchronize();
	new_matrix(rows, cols);
    }

    cuMat(const cuMat &a){
	cublasCreate(&cuda_handle_);
	cudaThreadSynchronize();

	new_matrix(a.rows_, a.cols_);

	cudaError_t error = cudaMemcpy(m_device_, a.m_device_, rows_*cols_*sizeof(m_device_), cudaMemcpyDeviceToDevice);

	if(error != cudaSuccess) FatalError("cuMat copy constructer failed");
    }

    ~cuMat(){
	del_matrix();
	cublasDestroy(cuda_handle_);
    }

    void memMallocHost(){
	m_host_ = (float*)malloc(rows_*cols_*sizeof(*m_host_));
	for(int i=0; i<rows_; i++){
	    for(int j=0; j<cols_; j++){
		m_host_[IDX2F(i, j, rows_)] = 0.0;
	    }
	}
    }

    void memMallocDevice(){
	cudaError_t error = cudaMalloc((void**)&m_device_, rows_*cols_*sizeof(*m_device_));
	if(error != cudaSuccess) FatalError("memMallocDevice failed\n");
	cudaMemset(m_device_, 0x00, rows_*cols_*sizeof(*m_device_));
	cudaThreadSynchronize();
    }
    
    void new_matrix(const int rows, const int cols){
	if(this->rows_ != rows || this->cols_ != cols){
	    if(m_device_ != nullptr || m_host_ != nullptr) del_matrix();
	    this->rows_ = rows;
	    this->cols_ = cols;

	    cudaError_t error;

	    error = cudaMalloc((void**)&m_device_, rows_*cols_*sizeof(*m_device_));
	    if(error != cudaSuccess) FatalError("new_matrix failed\n");
	    cudaMemset(m_device_, 0x00, rows_*cols_*sizeof(*m_device_));
	    cudaThreadSynchronize();
	    mallocCounter.up();
	}
    }

    void del_matrix(){
	if(m_device_ != nullptr){
	    cudaFree(m_device_);
	    m_device_ = nullptr;
	    mallocCounter.down();
	}
	if(m_host_ != nullptr){
	    free(m_host_);
	    m_host_ = nullptr;
	}
	cudaThreadSynchronize();
    }

    void memHostToDevice(){
	cudaError_t error = cudaMemcpy(m_device_, m_host_, rows_*cols_*sizeof(*m_device_),
				       cudaMemcpyHostToDevice);
	if(error != cudaSuccess) FatalError("memHostToDevice failed\n");
    }

    void memDeviceToHost(){
	if(m_host_ == nullptr) this->memMallocHost();
	cudaError_t error = cudaMemcpy(m_host_, m_device_, rows_*cols_*sizeof(*m_device_),
				       cudaMemcpyDeviceToHost);
	if(error != cudaSuccess) FatalError("memDevicetoHost faield\n");
    }

    void memSetHost(int i, int j, float val){
	if(m_host_ == nullptr) this->memMallocHost();
	m_host_[IDX2F(i, j, rows_)] = val;
    }
    
    void memSetHost(float *v){
	if(m_host_ == nullptr) this->memMallocHost();
	if(m_device_ == nullptr) FatalError("memSetHost m_device_ is nullptr");

	cudaError_t error = cudaMemcpy(m_device_, v, rows_*cols_*sizeof(m_device_),
				       cudaMemcpyHostToDevice);
	if(error != cudaSuccess) FatalError("memSetHost(float *v) failed\n");
    }

    void memSetDevice(float *v){
	cudaError_t error = cudaMemcpy(m_device_, v, rows_*cols_*sizeof(*m_device_),
				       cudaMemcpyDeviceToHost);
	if(error != cudaSuccess) FatalError("memSetDevice(float *v) failed\n");
    }

    void memSetDeviceRow(float *v, int row_index){
	cudaError_t error = cudaMemcpy(m_device_ + row_index*cols_, v, cols_*sizeof(float),
				       cudaMemcpyDeviceToDevice);
	if(error != cudaSuccess) FatalError("memSetDeviceRow failed\n");
    }

    void memSetDeviceCol(float *v, int col_index){
	cudaError_t error = cudaMemcpy(m_device_ + col_index*rows_, v, rows_*cols_*sizeof(float), cudaMemcpyDeviceToDevice);
	if(error != cudaSuccess) FatalError("memSetDeviceCol failed\n");
    }

    void toHostArray(){
	if(m_host_ == nullptr) this->memMallocHost();
	memDeviceToHost();

	m_host_array_.resize(rows_*cols_);
	for(int i=0; i<rows_; i++){
	    for(int j=0; j<cols_; j++){
		m_host_array_[IDX2F(i, j, rows_)] = m_host_[IDX2F(i, j, rows_)];
	    }
	}
    }

    void fromHostArray(){
	if(m_host_ == nullptr) this->memMallocHost();
	if(m_device_ == nullptr) this->memMallocDevice();
	for(int i=0; i<rows_; i++){
	    for(int j=0; j<cols_; j++){
		m_host_[IDX2F(i, j, rows_)] = m_host_array_[IDX2F(i, j, cols_)];
	    }
	}

	memHostToDevice();
    }

    cuMat sliceRows(int offset, int len){
	cuMat r(len, this->cols_);

	slice_rows_kernel_exec(m_device_, r.m_device_, cols_, rows_, offset, len);

	return r;
    }

    void joinRows(cuMat &a, int offset, int len){
	join_rows_kernel_exec(a.m_device_, m_device_, cols_, rows_, offset, len);
    }

    cuMat &operator=(const cuMat &a){
	new_matrix(a.rows_, a.cols_);
	cudaError_t error = cudaMemcpy(m_device_, a.m_device_, rows_*cols_*sizeof(*m_device_),
				      cudaMemcpyDeviceToDevice);
	if(error != cudaSuccess) FatalError("cuMat operator=(const cuMat &) failed\n");

	return *this;
    }
    
    float operator()(const int i, const int j){
	if(m_host_ == nullptr){
	    this->memMallocHost();
	}
	this->memDeviceToHost();
	return m_host_[IDX2F(i, j, rows_)];
    }

    friend void printRows(std::ostream &output, const cuMat &a, int i){
	output << "[";
	if(a.cols_ < 11){
	    for(int j=0; j<a.cols_; j++) output << a.m_host_[IDX2F(i, j, a.rows_)] << " ";
	}
	else{
	    for(int j=0; j<3; j++) output << a.m_host_[IDX2F(i, j, a.rows_)] << " ";
	    std::cout << "..., ";
	    for(int j=a.cols_-2; j<a.cols_; j++) output << a.m_host_[IDX2F(i, j, a.rows_)] << " ";
	}
	output << "]";
    }
    /*
     arithmatic manipulation functions
     -plus/minus ... for operator+/-.
       (cuMat + cuMat), (cuMat + float)
     -mul ... cuMat * float, cuMat*cuMat(element_wise)
     -dot ... cuMat@cuMat(multipliction of matricies)
     -div ... cuMat * (1 / float)
     operator+/- .. same as plus/minus
     operator* ... same as mul(doesnot operate @-multiplication)
     operator/ ... same as div
    */

    friend cuMat operator+(const cuMat &);
    void ones(){
	mat_ones_kernel_exec(m_device_, m_device_, cols_, rows_);
    }

    void plus(const cuMat &b, cuMat &r){
	float alpha = 1, beta = 1;
	cublasStatus_t stat = cublasSgeam(r.cuda_handle_, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, m_device_, rows_, &beta, b.m_device_, rows_, r.m_device_, r.rows_);

	if(stat != CUBLAS_STATUS_SUCCESS){
	    FatalError("cannot cublasSgeam");
	}
	cudaThreadSynchronize();
    }

    void plus(const float beta, cuMat &r){
	cuMat i(rows_, cols_);
	i.ones();

	float alpha = 1;
	cublasStatus_t stat = cublasSgeam(r.cuda_handle_, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, m_device_, rows_, &beta, r.m_device_, r.rows_, r.m_device_, r.rows_);
	if(stat != CUBLAS_STATUS_SUCCESS){
	    FatalError("cannot cublasSgeam");
	}
	cudaThreadSynchronize();
    }

    void plus(const float beta, cuMat &i, cuMat &r){
	float alpha = 1;
	cublasStatus_t stat = cublasSgeam(r.cuda_handle_, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, m_device_, rows_, &beta, i.m_device_, i.rows_, r.m_device_, r.rows_);
	if(stat != CUBLAS_STATUS_SUCCESS){
	    FatalError("cannot cublasSgeam");
	}
	cudaThreadSynchronize();
    }

    void minus(const cuMat &b, cuMat &r){
	float alpha = 1;
	float beta = -1;
	cublasStatus_t stat = cublasSgeam(r.cuda_handle_, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, m_device_, rows_, &beta, b.m_device_, rows_, r.m_device_, r.rows_);
	if(stat != CUBLAS_STATUS_SUCCESS){
	    FatalError("cannot cublasSgeam in minus(const cuMat &, const cuMat &)");
	}
	cudaThreadSynchronize();
    }

    void mul(const float alpha, cuMat &r){
	float beta = 0;
	cublasStatus_t stat = cublasSgeam(r.cuda_handle_, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, m_device_, rows_, &beta, r.m_device_, rows_, r.m_device_, r.rows_);
	if(stat !=CUBLAS_STATUS_SUCCESS) FatalError("cannot cublasSgeam in mul(cosnt float, cuMat &)");
	cudaThreadSynchronize();
    }

    void mul(const cuMat &m, const cuMat &r){
	mat_mul_elementwise_kernel_exec(m_device_, m.m_device_, r.m_device_, cols_, rows_);
    }

    void mul_plus(const float alpha, cuMat &r){
	float beta = 1;
	cublasStatus_t stat = cublasSgeam(r.cuda_handle_, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, m_device_, rows_, &beta, r.m_device_, r.rows_, r.m_device_, r.rows_);
	if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot cublasSgeam in mul_plus");
	cudaThreadSynchronize();
    }
    void mul_plus(const cuMat &m, const cuMat &r, float alpha, float beta){
	mat_mul_plus_elementwise_kernel_exec(m_device_, m.m_device_, r.m_device_, alpha, beta, cols_, rows_);
    }
};
#endif
