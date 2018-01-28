#ifndef CUMAT_H_
#define CUMAT_H_

#include "mat_ones_kernel.h"
#include "mat_mul_elementwise_kernel.h"
#include "mat_mul_plus_elementwise_kernel.h"
#include "matmod_kernel.h"
#include "slice_rows_kernel.h"
#include "mat_div_kernel.h"
#include "mat_log_kernel.h"
#include "mat_sqrt_kernel.h"
#include "mat_sqrt_d_kernel.h"
#include "mat_cos_kernel.h"
#include "mat_sin_kernel.h"
#include "relu_kernel.h"
#include "relu_d_kernel.h"
#include "prelu_kernel.h"

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

        cudaError_t error = cudaMemcpy(m_device_, a.m_device_, rows_*cols_*sizeof(m_device_),
                                       cudaMemcpyDeviceToDevice);

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
        //サイズが違う場合はリサイズ
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
        cudaError_t error = cudaMemcpy(m_device_ + col_index*rows_, v, rows_*cols_*sizeof(float),
                                       cudaMemcpyDeviceToDevice);
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

    friend std::ostream &operator<<(std::ostream &output, cuMat &a){
        if(a.m_device_ == nullptr){
            FatalError("m_device_ is nullptr so cannot <<");
        }
        if(a.m_host_ == nullptr){
            FatalError("m_host_ is nullptr so cannot <<");
        }

        cudaError_t error = cudaMemcpy(a.m_host_, a.m_device_, a.rows_*a.cols_*sizeof(*m_device_), cudaMemcpyDeviceToHost);
        if(error != cudaSuccess){
            FatalError("cudaMemcpy failed in <<");
        }
        
        output << "A matrix of " << a.rows_ << " X " << a.cols_ << std::endl;
        output << "[";
        if(a.rows_ < 11){
            for(int i=0; i<a.rows_; i++){
                printRows(output, a, i);
                if(i != a.rows_-1) output << std::endl;
                else output << "]" << std::endl;
            }
        }
        else{
            for(int i=0; i<5; i++){
                printRows(output,  a, i);
                output << std::endl;
            }
            output << "...," << std::endl;
            for(int i=a.rows_-5; i<a.rows_; i++){
                printRows(output, a, i);
                if(i != a.rows_-1) output << std::endl;
                else{ output << "]\n";}
            }
        }
        return output;
    }
    /*
      arithmatic manipulation functions
      plus/minus ... for operator+/-.
      (cuMat + cuMat), (cuMat + float)
      mul ... cuMat * float, cuMat*cuMat(element_wise)
      dot ... cuMat@cuMat(multipliction of matricies)
      div ... cuMat * (1 / float)
      operator+/- .. same as plus/minus
      operator* ... same as mul(doesnot operate @-multiplication)
      operator/ ... same as div
    */

    void copy(const cuMat &a){
        if(rows_ != a.rows_ || cols_ != a.cols_){
            FatalError("the size doesnot match in copy.");
        }
        cudaError_t error = cudaMemcpy(m_device_, a.m_device_, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) FatalError("cudaMemcpy failed in copy");
    }

    void ones(){
        mat_ones_kernel_exec(m_device_, m_device_, cols_, rows_);
    }

    //r[i][j] <- this[i][j] + b
    void plus(const cuMat &b, cuMat &r){
        float alpha = 1, beta = 1;
        cublasStatus_t stat = cublasSgeam(r.cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          b.m_device_, rows_,
                                          r.m_device_, r.rows_);

        if(stat != CUBLAS_STATUS_SUCCESS){
            FatalError("cannot cublasSgeam");
        }
        cudaThreadSynchronize();
    }

    //r[i][j] <- this[i][j] + beta
    void plus(const float beta, cuMat &r){
        cuMat i(rows_, cols_);
        i.ones();

        float alpha = 1;
        cublasStatus_t stat = cublasSgeam(r.cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          i.m_device_, i.rows_,
                                          r.m_device_, r.rows_);
        if(stat != CUBLAS_STATUS_SUCCESS){
            FatalError("cannot cublasSgeam");
        }
        cudaThreadSynchronize();
    }

    //r[i][j] <- this[i][j] + beta*i[i][j]
    void plus(const float beta, cuMat &i, cuMat &r){
        float alpha = 1;
        cublasStatus_t stat = cublasSgeam(r.cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          i.m_device_, i.rows_,
                                          r.m_device_, r.rows_);
        if(stat != CUBLAS_STATUS_SUCCESS){
            FatalError("cannot cublasSgeam");
        }
        cudaThreadSynchronize();
    }

    //r[i][j] = this[i][j] - b[i][j]
    void minus(const cuMat &b, cuMat &r){
        float alpha = 1;
        float beta = -1;
        cublasStatus_t stat = cublasSgeam(r.cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          b.m_device_, rows_,
                                          r.m_device_, r.rows_);
        if(stat != CUBLAS_STATUS_SUCCESS){
            FatalError("cannot cublasSgeam in minus(const cuMat &, const cuMat &)");
        }
        cudaThreadSynchronize();
    }

    //r[i][j] <- alpha * this[i][j]
    void mul(const float alpha, cuMat &r){
        float beta = 0;
        cublasStatus_t stat = cublasSgeam(r.cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          r.m_device_, rows_,
                                          r.m_device_, r.rows_);
        if(stat !=CUBLAS_STATUS_SUCCESS) FatalError("cannot cublasSgeam in mul(cosnt float, cuMat &)");
        cudaThreadSynchronize();
    }

    //r[i][j] <- this[i][j] * m[i][j]
    void mul(const cuMat &m, cuMat &r){
        mat_mul_elementwise_kernel_exec(m_device_, m.m_device_, r.m_device_, cols_, rows_);
    }

    //r[i][j] += alpha * this[i][j]
    void mul_plus(const float alpha, cuMat &r){
        float beta = 1;
        cublasStatus_t stat = cublasSgeam(r.cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          r.m_device_, r.rows_,
                                          r.m_device_, r.rows_);
        if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot cublasSgeam in mul_plus");
        cudaThreadSynchronize();
    }

    //r[i][j] += alpha * this[i][j] * beta * m[i][j]
    void mul_plus(const cuMat &m, cuMat &r, float alpha, float beta){
        mat_mul_plus_elementwise_kernel_exec(m_device_, m.m_device_, r.m_device_, alpha, beta, cols_, rows_);
    }

    //r[i][j] <- this[i][j] / p
    void div(const float p, cuMat &r){
        matmod_kernel_exec(m_device_, r.m_device_, cols_, rows_, p);
    }

    //r[i][j] <-  this[i][j] / b[i][j]
    void div(const cuMat &b, const cuMat &r){
        mat_div_kernel_exec(m_device_, b.m_device_, r.m_device_, cols_, rows_);
    }

    //A.dot(B) returns A@B
    cuMat dot(const cuMat &b){
        cuMat r(this->rows_, b.cols_);
        dot(b, r);
        return r;
    }

    // r <- this@b where this(rows_ X cols_) and b(rows_ X)
    void dot(const cuMat &b, cuMat &r){
        float alpha = 0;
        float beta = 0;
        cublasStatus_t stat = cublasSgemm(cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_,b.cols_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          b.m_device_, b.rows_,
                                          &beta,
                                          r.m_device_, r.rows_);
        checkCublasErrors(stat);
        if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot cublasSgemm dot");
        cudaThreadSynchronize();
    }

    //r += this@b
    void dot_plus(const cuMat &b, cuMat &r){
        float alpha = 1;
        float beta = 1;

        cublasStatus_t stat = cublasSgemm(cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, b.cols_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          b.m_device_, b.rows_,
                                          &beta,
                                          r.m_device_, r.rows_
                                          );
        checkCublasErrors(stat);
        if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot dot_plus cublasSgemm");
        cudaThreadSynchronize();
    }

    //r += Trans(this)@b
    void transpose_dot_plus(const cuMat &b, cuMat &r){
        float alpha = 1;
        float beta = 1;

        cublasStatus_t stat = cublasSgemm(cuda_handle_,
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          cols_, b.cols_, rows_,
                                          &alpha,
                                          m_device_, rows_,
                                          b.m_device_, b.rows_,
                                          &beta,
                                          r.m_device_, r.rows_);
        checkCublasErrors(stat);
        if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot transpose_dot_plus cublasSgemm");
        cudaThreadSynchronize();
    }

    //
    void dot_transpose_plus(const cuMat &b, cuMat &r){
        float alpha = 1;
        float beta = 1;
        cublasStatus_t stat = cublasSgemm(cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_T,
                                          rows_, b.rows_, cols_,
                                          &alpha,
                                          m_device_, rows_,
                                          b.m_device_, b.rows_,
                                          &beta,
                                          r.m_device_, r.rows_);
        checkCublasErrors(stat);
        if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot dot_transpose_plus cublasSgemm");
        cudaThreadSynchronize();
    }

    cuMat transpose(){
        cuMat r(cols_, rows_);
        transpose(r);
        return r;
    }
    
    void transpose(cuMat &r){
        float alpha = 1;
        float beta = 0;
        cublasStatus_t stat = cublasSgeam(cuda_handle_,
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          cols_, rows_,
                                          &alpha,
                                          m_device_, rows_,
                                          &beta,
                                          r.m_device_, cols_,
                                          r.m_device_, cols_);
        checkCublasErrors(stat);
        if(stat != CUBLAS_STATUS_SUCCESS){
            FatalError("cannnot transpose cublasSgem");
        }
        cudaThreadSynchronize();
    }
    
    void plus_util(float alpha, float beta, cuMat &b, cuMat &r){
        cublasStatus_t stat = cublasSgeam(cuda_handle_,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          rows_, cols_,
                                          &alpha, m_device_, rows_,
                                          &beta, b.m_device_, rows_,
                                          r.m_device_, rows_);
        if(stat != CUBLAS_STATUS_SUCCESS) FatalError("cannot plus_util cublasSgeam");
        cudaThreadSynchronize();
    }

    void log(cuMat &r, float alpha){
        mat_log_kernel_exec(m_device_, r.m_device_, cols_, rows_, alpha);
    }

    cuMat log(){
        cuMat r(rows_, cols_);
        log(r, 0.0);
        return r;
    }

    cuMat sqrt(){
        cuMat r(rows_, cols_);
        sqrt(r, 1e-8);
        return r;
    }

    void sqrt(cuMat &r, float alpha){
        mat_sqrt_kernel_exec(m_device_, r.m_device_, cols_, rows_, alpha);
    }

    cuMat sqrt_d(){
        cuMat r(rows_, cols_);
        sqrt_d(r, 1e-8);
        return r;
    }

    void sqrt_d(cuMat &r, float alpha){
        mat_sqrt_d_kernel_exec(m_device_, r.m_device_, cols_, rows_, alpha);
    }

    cuMat  sin(){
        cuMat r(rows_, cols_);
        sin(r);
        return r;
    }

    void sin(cuMat &r){
        mat_sin_kernel_exec(m_device_, r.m_device_, cols_, rows_, 0);
    }

    cuMat cos(){
        cuMat r(rows_, cols_);
        cos(r);
        return r;
    }

    void cos(cuMat &r){
        mat_cos_kernel_exec(m_device_, r.m_device_, cols_, rows_, 0);
    }

    cuMat relu(){
        cuMat r(rows_, cols_);
        relu(r);
        return r;
    }

    void relu(cuMat &r){
        relu_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    cuMat relu_d(){
        cuMat r(rows_, cols_);
        relu_d(r);
        return r;
    }

    void relu_d(cuMat &r){
        relu_d_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    cuMat prelu(cuMat &a){
        cuMat r(rows_, cols_);
        prelu(a, r);
        return r;
    }

    void prelu(cuMat &a, cuMat &r){
        prelu_kernel_exec(m_device_, a.m_device_, r.m_device_, cols_, rows_);
    }


    cuMat prelu_d(cuMat &a, cuMat &da){
        cuMat r(rows_, cols_);
        prelu_d(a, r, da);
        return r;
    }
    
    void prelu_d(cuMat &a, cuMat &r, cuMat &da){
        prelu_d_kernel_exec(m_device_, a.m_device_, r.m_device_, cols_, rows_);
    }
};
#endif
