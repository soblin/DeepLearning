#ifndef CUMAT_H_
#define CUMAT_H_

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

    cublasHandle_t cuda_handle;
    
public:

    inline int row() const { return rows_; }
    
    inline int col() const { return cols_; }

    inline float* m_device() const { return m_device_; }
    
    cuMat(){
	rows_ = cols_ = 0;
	cublasCreate(&cuda_handle);
	cudaThreadSynchronize();
    }

    cuMat(const int rows, const int cols){
	cublasCreate(&cuda_handle);
	cudaThreadSynchronize();
	new_matrix(rows, cols);
    }

    cuMat(const cuMat &a){
	cublasCreate(&cuda_handle);
	cudaThreadSynchronize();

	new_matrix(a.row(), a.col());

	cudaError_t error = cudaMemcpy(m_device_, a.m_device(), rows_*cols_*sizeof(m_device_), cudaMemcpyDeviceToDevice);

	if(error != cudaSuccess) FatalError("cuMat copy constructer failed");
    }

    ~cuMat(){
	del_matrix();
	cublasDestroy(cuda_handle);
    }

    void new_matrix(const int rows, const int cols){
	if(this->row() != rows || this->col() != cols){
	    if(m_device_ != nullptr || m_host_ != nullptr) del_matrix();
	    this->rows_ = rows;
	    this->cols_ = cols;

	    cudaError_t error;
	    cublasStatus_t status;

	    error = cudaMalloc((void**)&m_device_, rows_*cols_*sizeof(*m_device_));
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
};
#endif
