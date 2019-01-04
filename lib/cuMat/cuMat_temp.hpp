/*!
  @file cuMat.hpp
  @brief header file for libcumat.so
  @author soblin
  @date 3/10
 */

#ifndef CUMAT_HPP_
#define CUMAT_HPP_

#include "DeepLearning/cuMat/include/mat_ones_kernel.h"
#include "DeepLearning/cuMat/include/mat_mul_elementwise_kernel.h"
#include "DeepLearning/cuMat/include/mat_mul_plus_elementwise_kernel.h"
#include "DeepLearning/cuMat/include/mat_mod_kernel.h"
#include "DeepLearning/cuMat/include/slice_rows_kernel.h"
#include "DeepLearning/cuMat/include/mat_div_kernel.h"
#include "DeepLearning/cuMat/include/mat_log_kernel.h"
#include "DeepLearning/cuMat/include/mat_sqrt_kernel.h"
#include "DeepLearning/cuMat/include/mat_sqrt_d_kernel.h"
#include "DeepLearning/cuMat/include/mat_cos_kernel.h"
#include "DeepLearning/cuMat/include/mat_sin_kernel.h"
#include "DeepLearning/cuMat/include/relu_kernel.h"
#include "DeepLearning/cuMat/include/relu_d_kernel.h"
#include "DeepLearning/cuMat/include/prelu_kernel.h"
#include "DeepLearning/cuMat/include/prelu_d_kernel.h"
#include "DeepLearning/cuMat/include/sigmoid_kernel.h"
#include "DeepLearning/cuMat/include/sigmoid_d_kernel.h"
#include "DeepLearning/cuMat/include/tanh_kernel.h"
#include "DeepLearning/cuMat/include/tanh_d_kernel.h"
#include "DeepLearning/cuMat/include/softmax_kernel.h"
#include "DeepLearning/cuMat/include/mat_l2_kernel.h"
#include "DeepLearning/cuMat/include/mat_exp_kernel.h"
#include "DeepLearning/cuMat/include/mat_inverse_kernel.h"
#include "DeepLearning/cuMat/include/mat_inverse_d_kernel.h"

#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <map>
#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <cmath>
#include <cstdlib>

#include "DeepLearning/cuMat/cublas_v2.h"
#include "DeepLearning/cuMat/cuda_runtime.h"

/*!
   @brief convert 2D-index to 1D-array index(column major)
   @param i rows
   @param j cols
   @return j * width_of_row + i
 */
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

/*! 
  @class cuMat
  @brief cuMat is a matrix class with arithmatic operators and fuctions
 */
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

    /*!
      @brief This function returns the row number.
     */
    inline int row() const { return rows_; }

    /*!
      @brief This function returns the column number.
     */
    inline int col() const { return cols_; }


    /*!
      @brief The matrix will initialized to size 0.
     */
    cuMat(){
        rows_ = cols_ = 0;
        cublasCreate(&cuda_handle_);
        cudaThreadSynchronize();
    }

    /*!
      @brief The matrix will be initialized to the designated size
     */
    cuMat(const int rows, const int cols){
        cublasCreate(&cuda_handle_);
        cudaThreadSynchronize();
        new_matrix(rows, cols);
    }


    /*!
      @brief This is copy constructor. It uses new_matrix() inside.
      @sa new_matrix 
     */
    cuMat(const cuMat &a){
        cublasCreate(&cuda_handle_);
        cudaThreadSynchronize();

        new_matrix(a.rows_, a.cols_);

        // 3/12 modified from sizeof(m_device_) to sizeof(*m_device_)
        cudaError_t error = cudaMemcpy(m_device_, a.m_device_, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyDeviceToDevice);

        if(error != cudaSuccess) FatalError("cuMat copy constructer failed");
    }

    /*!
      @brief This is the destructor. It uses del_matrix inside.
     */
    ~cuMat(){
        del_matrix();
        cublasDestroy(cuda_handle_);
    }

    /*!
      @brief This function malloc memory for host device(main memory) of size [*this->rows_]x[*this->cols_]
     */
    void memMallocHost(){
        m_host_ = (float*)malloc(rows_*cols_*sizeof(*m_host_));
        //if(!m_host_) FatalError("memMallocHost failed");
        for(int i=0; i<rows_; i++){
            for(int j=0; j<cols_; j++){
                m_host_[IDX2F(i, j, rows_)] = 0.0;
            }
        }
    }

    /*!
      @brief This funciton malloc memory in device and 0-clear
     */
    void memMallocDevice(){
        cudaError_t error = cudaMalloc((void**)&m_device_, rows_*cols_*sizeof(*m_device_));
        if(error != cudaSuccess) FatalError("memMallocDevice failed\n");
        cudaMemset(m_device_, 0x00, rows_*cols_*sizeof(*m_device_));
        cudaThreadSynchronize();
    }

    /*!
      @brief new_matrix malloc new memory in device. If the current size of *this and designated size is different, *this will be resized. 
      @sa del_matrix
    */
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

    /*!
      @brief del_matrix frees both main memory and device memory.
    */
    
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

    /*!
      @brief This function copies host memory to device memory of *this (HostToDevice).
     */
    void memHostToDevice(){
        cudaError_t error = cudaMemcpy(m_device_/*dst*/, m_host_/*src*/, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyHostToDevice);
        if(error != cudaSuccess) FatalError("memHostToDevice failed\n");
    }

    /*!
      @brief This function copies device memory of *this to host memory of *this (DeviceToHost)
      @sa memMallocHost
     */
    void memDeviceToHost(){
        if(m_host_ == nullptr) this->memMallocHost();
        cudaError_t error = cudaMemcpy(m_host_/*dst*/, m_device_/*src*/, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyDeviceToHost);
        if(error != cudaSuccess) FatalError("memDevicetoHost faield\n");
    }

    /*!
      @brief This function set host memory[i][j] = val
     */
    void memSetHost(int i, int j, const float val){
        if(m_host_ == nullptr) this->memMallocHost();
        m_host_[IDX2F(i, j, rows_)] = val;
    }

    /*!
      @brief This function copies the host/device array *v to device_memory of *this (HostToDevice).
    */
    void memSetHost(const float *v){
        if(m_host_ == nullptr) this->memMallocHost();
        if(m_device_ == nullptr){
            std::cout << "memSetHost m_device_ is nullptr" << std::endl;
        }
        cudaError_t error = cudaMemcpy(m_device_/*dst*/, v/*src*/, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyHostToDevice);
        if(error != cudaSuccess) FatalError("memSetDevice(float *v) failed\n");
    }

    /*!
      @brief This function copies the host/device array *v to device memory of *this (DeviceToDevice)
      @param float *v This is the src array to copy to device memory
     */
    void memSetDevice(const float *v){
        cudaError_t error = cudaMemcpy(m_device_/*dst*/, v/*src*/, rows_*cols_*sizeof(m_device_),
                                       cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) FatalError("memSetHost(float *v) failed\n");
    }


    /*!
      @brief This function copies array v to the row of row_index (DeviceToDevice)
      @param v This is the src array
      @param row_index This is the index of column
     */
    void memSetDeviceRow(const float *v, int row_index){
        cudaError_t error = cudaMemcpy(m_device_ + row_index*cols_, v, cols_*sizeof(float),
                                       cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) FatalError("memSetDeviceRow failed\n");
    }

    /*!
      @brief This function copies array v to the column of col_index (DeviceToDevice).
      @param v This is the src array.
      @param col_index This is the index of column
     */
    void memSetDeviceCol(const float *v, int col_index){
        cudaError_t error = cudaMemcpy(m_device_ + col_index*rows_, v, rows_*cols_*sizeof(float),
                                       cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) FatalError("memSetDeviceCol failed\n");
    }

    /*!
      @brief This function copies the device memory to host array(std::vector<float> m_host_array)
      @sa memMallocHost
      @sa memDeviceToHost
     */
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

    /*!
      @brief This function copies host array to device memory.
     */
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

    /*!
      @brief This is the substition constructor(<=> copy constructor). Copies device memory of a to that of *this.
      @param a This is the rvalue of operator=
      @sa cuMat(const cuMat &a)
     */
    cuMat &operator=(const cuMat &a){
        new_matrix(a.rows_, a.cols_);
        cudaError_t error = cudaMemcpy(m_device_/*dst*/, a.m_device_/*src*/, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) FatalError("cuMat operator=(const cuMat &) failed\n");

        return *this;
    }

    /*!
      @brief This returns the current value of device memroy by copying device memory to host memory with memDeviceToHost.
      @sa memDeviceToHost
     */
    float operator()(const int i, const int j){
        if(m_host_ == nullptr){
            this->memMallocHost();
        }
        this->memDeviceToHost();
        return m_host_[IDX2F(i, j, rows_)];
    }

    /*!
      @brief This function displays the i-th row of *this. If the column is ls leq10, print all. Else print only the 2elements from both start and end.
     */
    friend void printRows(std::ostream &output, const cuMat &a, int i){
        output.setf(std::ios::left, std::ios::adjustfield);
        output << "[";
        if(a.cols_ < 7){
            for(int j=0; j<a.cols_; j++) output << std::setw(7) << a.m_host_[IDX2F(i, j, a.rows_)] << " ";
        }
        else{
            for(int j=0; j<3; j++) output << std::setw(7) <<a.m_host_[IDX2F(i, j, a.rows_)] << " ";
            std::cout << "..., ";
            for(int j=a.cols_-2; j<a.cols_; j++) output << a.m_host_[IDX2F(i, j, a.rows_)] << " ";
        }
        output << "]";
    }

    /*!
      @brief This function prints the matrix. If the size is smaller than 11x11, print all.
     */
    friend std::ostream &operator<<(std::ostream &output, cuMat &a){
        if(a.m_device_ == nullptr){
            std::cout << "cuMat operator<< a.m_device_ is nullptr" << std::endl;
            if(a.m_host_ == nullptr){
                std::cout << "also cuMat operator<< a.m_host_ is nullptr" << std::endl;
            }
        }

        if(a.m_host_ == nullptr) a.memMallocHost();
        
        cudaError_t error = cudaMemcpy(a.m_host_/*dst*/, a.m_device_/*src*/, a.rows_*a.cols_*sizeof(*m_device_), cudaMemcpyDeviceToHost);
        if(error != cudaSuccess){
            FatalError("cudaMemcpy failed in <<");
        }
        
        output << "A matrix of " << a.rows_ << " X " << a.cols_ << std::endl;
        output << "[";
        if(a.rows_ < 7){
            for(int i=0; i<a.rows_; i++){
                if(i != 0) output << " ";
                printRows(output, a, i);
                if(i != a.rows_-1) output << "\n";
                else output << "]" << std::endl;
            }
        }
        else{
            for(int i=0; i<5; i++){
                if(i != 0) output << " ";
                printRows(output,  a, i);
                output << "\n";
            }
            output << " ...," << std::endl;
            for(int i=a.rows_-5; i<a.rows_; i++){
                if(i != 0) output << " ";
                printRows(output, a, i);
                if(i != a.rows_-1) output << "\n";
                else{ output << "]" << std::endl;}
            }
        }
        std::cout << std::endl;
        return output;
    }
    /*
      operator+/- .. same as plus/minus
      operator* ... same as mul(doesnot operate @-multiplication)
      operator/ ... same as div
    */

    /*!
      @brief This function copies device memory of a to that of this.
     */
    void copy(const cuMat &a){
        if(rows_ != a.rows_ || cols_ != a.cols_){
            FatalError("the size doesnot match in copy.");
        }
        cudaError_t error = cudaMemcpy(m_device_/*dst*/, a.m_device_/*src*/, rows_*cols_*sizeof(*m_device_),
                                       cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) FatalError("cudaMemcpy failed in copy");
    }

    /*!
      @brief This function sets all the elements of *this to 1.
      @sa mat_ones_kernel_exec
     */
    void ones(){
        mat_ones_kernel_exec(m_host_, m_device_, cols_, rows_);
    }

    /*!
      @brief This is the addition of matrix.
      @sa plus
     */
    friend cuMat operator+(const cuMat &a, const cuMat &b){
        cuMat r = a; //copy constructor.
        r.plus(b, r);

        return r;
    }

    /*!
      @brief This is the addition of matrix and real number. Each element is added by a
      @sa plus
     */
    friend cuMat operator+(const float a, const cuMat &b){
        cuMat r = b;
        r.plus(a, r);

        return r;
    }
    
    /*!
      @brief This is the addition of matrix and real number. Each element is added by a
      @sa plus
     */
    friend cuMat operator+(const cuMat &a, const float b){
        cuMat r = a;
        r.plus(b, r);

        return r;
    }

    /*!
      @brief This is the subtraction of matrix.
      @sa minus
     */
    friend cuMat operator-(const cuMat &a, const cuMat &b){
        cuMat r = a;
        //r <= r(==a) - b
        r.minus(b, r);

        return r;
    }

    /*!
      @brief This is the elemnetwise product of matrix. Please notice.
      @sa mul
      @sa dot
     */
    friend cuMat operator*(const cuMat &a, const cuMat &b){
        cuMat r = a;
        r.mul(b, r);

        return r;
    }

    /*!
      @brief This operator returns a matrix that is multiplied real number to all element of matrix.
      @sa mul
     */
    friend cuMat operator*(const float a, const cuMat &b){
        cuMat r = b;
        r.mul(a, r);

        return r;
    }

    /*!
      @brief This operator returns a matrix that is multiplied real number to all element of matrix.
      @sa mul
     */
    friend cuMat operator*(const cuMat &a, const float b){
        cuMat r = a;
        r.mul(b, r);

        return r;
    }

    /*!
      @brief This operator returns a matrix that is multiplied 1/real number to a matrix.
     */
    friend cuMat operator/(float p, cuMat &b){
        cuMat r = b;
        b.div(p, r);

        return r;
    }

    /*!
      @brief This operator returns a matrix that is multiplied 1/real number to a matrix.
     */
    friend cuMat operator/(const cuMat &a, float b){
        cuMat r = a;
        r.mul(1.0/b, r);

        return r;
    }

    /*!
      @brief This is the elemnetwise division of matricies.
     */
    friend cuMat operator/(const cuMat &a, const cuMat &b){
        cuMat r = a;
        r.div(b, r);

        return r;
    }

    cuMat &operator+=(const cuMat &a){
        plus(a, *this);
        return *this;
    }

    cuMat &operator+=(const float a){
        plus(a, *this);
        return *this;
    }

    cuMat &operator-=(const cuMat &b){
        minus(b, *this);
        return *this;
    }

    cuMat &operator-=(const float b){
        plus(-b, *this);
        return *this;
    }

    cuMat &operator*=(const cuMat &a){
        mul(a, *this);
        return *this;
    }

    cuMat &operator*=(const float a){
        mul(a, *this);
        return *this;
    }

private:

    /*!
      @brief This function calculates r[i][j] <= this[i][j] + b[i][j]
     */
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
    /*!
      @brief This function calculates r[i][j] <= this[i][j] + beta
     */
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

    /*!
      @brief This function caluculates r[i][j] <= this[i][j] + beta*i[i][j]
     */
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

    /*!
      @brief This function calculates r[i][j] <= this[i][j] - b[i][j]
    */
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

    /*!
      @brief This function caluculates r[i][j] <= alpha * this[i][j]
    */
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

    /*!
      @brief This function calculates elementwise product r[i][j] <= this[i][j] * m[i][j]. 
    */
    void mul(const cuMat &m, cuMat &r){
        mat_mul_elementwise_kernel_exec(m_device_, m.m_device_, r.m_device_, cols_, rows_);
    }

    //r[i][j] += alpha * this[i][j]
    /*!
      @brief This operates r[i][j] += alpha*this[i][j]
     */
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

    /*!
      @brief This function operates r[i][j] += alpha * this[i][j] * beta * m[i][]j]
     */
    void mul_plus(const cuMat &m, cuMat &r, float alpha, float beta){
        mat_mul_plus_elementwise_kernel_exec(m_device_, m.m_device_, r.m_device_, alpha, beta, cols_, rows_);
    }

    /*!
      @brief This operates r[i][j] <= this[i][j] / p
     */
    void div(const float p, cuMat &r){
        mat_mod_kernel_exec(m_device_, r.m_device_, cols_, rows_, p);
    }

    /*!
      @brief This function operates r[i][j] <= this[i][j] / b[i][j]
     */
    void div(const cuMat &b, const cuMat &r){
        mat_div_kernel_exec(m_device_, b.m_device_, r.m_device_, cols_, rows_);
    }

public:
    /*!
      @brief This returns a matrix product with b. like C = A.dot(B)
      @param b This is the matrix to multiply to *this
     */
    cuMat dot(const cuMat &b){
        cuMat r(this->rows_, b.cols_);
        dot(b, r);
        return r;
    }

    // r <- this@b where this(rows_ X cols_) and b(rows_ X)
    /*!
      @brief This operates r <= *this @ B
      @param b This is the operand of *this
      @param r This is the matrix to store the result
     */
    void dot(const cuMat &b, cuMat &r){
        float alpha = 1;
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

    /*!
      @brief This operates r += *this @ b
      @param b This is the operand
      @param r This is the matrix to store the result
     */
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

    /*!
      @brief This operates r += t(*this) @ b
      @param b This is the operand
      @param r This is the matrix to store the result
     */
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

    /*!
     */
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

    /*!
      @brief This returns transposed matrix
     */
    cuMat transpose(){
        cuMat r(cols_, rows_);
        transpose(r);
        return r;
    }

    /*!
     */
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

public:
    //followiings are methematical functions.
    cuMat log(){
        cuMat r(rows_, cols_);
        log(r, 1e-8);
        return r;
    }
    
    cuMat sqrt(){
        cuMat r(rows_, cols_);
        sqrt(r, 1e-8);
        return r;
    }

    /*!
      @brief This returns d/dx{sqrt(this)}
      @sa mat_sqrt_d_kernel_exec
     */
     cuMat sqrt_d(){
        cuMat r(rows_, cols_);
        sqrt_d(r, 1e-8);
        return r;
    }

    cuMat  sin(){
        cuMat r(rows_, cols_);
        sin(r);
        return r;
    }

    cuMat cos(){
        cuMat r(rows_, cols_);
        cos(r);
        return r;
    }

    /*!
      @sa relu_kernel_exec
     */
    cuMat relu(){
        cuMat r(rows_, cols_);
        relu(r);
        return r;
    }

    /*!
      @sa prelu_kernel_exec
     */
    cuMat prelu(cuMat &a){
        cuMat r(rows_, cols_);
        prelu(a, r);
        return r;
    }

    /*!
      @sa relu_d_kernel_exec
     */
    cuMat relu_d(){
        cuMat r(rows_, cols_);
        relu_d(r);
        return r;
    }

    /*!
      @sa prelu_d_kernel_exec
     */
    cuMat prelu_d(cuMat &a, cuMat &da){
        cuMat r(rows_, cols_);
        prelu_d(a, r, da);
        return r;
    }

    /*!
      @sa sigmoid_kernel_exec
     */
    cuMat sigmoid(){
        cuMat r(rows_, cols_);
        sigmoid(r);
        return r;
    }

    /*!
      @sa sigmoid_d_kernel_exec
     */
    cuMat sigmoid_d(){
        cuMat r(rows_, cols_);
        sigmoid_d(r);
        return r;;
    }

    /*!
      @sa tanh_kernel_exec
     */
    cuMat tanh(){
        cuMat r(rows_, cols_);
        tanh(r);
        return r;
    }

    /*!
      @sa tanh_d_kernel_exec
     */
    cuMat tanh_d(){
        cuMat r(rows_, cols_);
        tanh_d(r);
        return r;
    }

    /*!
      @sa softmax_kernel_exec
     */
    cuMat softmax(){
        cuMat r(rows_, cols_);
        softmax(r);
        return r;
    }

    float l2(){
        float *sum_d;
        float sum_h = 0;
        cudaError_t error = cudaMalloc((void**)&sum_d, sizeof(*sum_d));
        if(error != cudaSuccess) FatalError("cudaMalloc failed in l2");
        cudaThreadSynchronize();
        cudaMemset(sum_d, 0x00, sizeof(*sum_d));
        mat_l2_kernel_exec(m_device_, sum_d, cols_, rows_);

        error = cudaMemcpy(&sum_h, sum_d, sizeof(*sum_d), cudaMemcpyDeviceToHost);
        if(error != cudaSuccess) FatalError("cudaMemcpy in l2");
        cudaFree(sum_d);

        return std::sqrt(sum_h);
    }

    void fill(float a){
        this->ones();
        this->mul(a, *this);
    }

    /*!
      @brief This returns elemnentwise exponeintial of *this
      @sa mat_exp_kernel_exec
     */
    cuMat exp(){
        cuMat r(rows_, cols_);
        exp(r);
        return r;
    }

    /*!
      @brief This returns a elementwise inverse matrix of *this.
      @sa mat_inverse_kernel_exec
    */
    cuMat inverse(){
        cuMat r(rows_, cols_);
        inverse(r);
        return r;
    }

    /*!
      @sa mat_inverse_d_kernel_exec
     */
    cuMat inverse_d(){
        cuMat r(rows_, cols_);
        inverse_d(r);
        return r;
    }
private:
    //followings are backends of mathematical functions.
    void log(cuMat &r, float alpha){
        mat_log_kernel_exec(m_device_, r.m_device_, cols_, rows_, alpha);
    }

    void sqrt(cuMat &r, float alpha){
        mat_sqrt_kernel_exec(m_device_, r.m_device_, cols_, rows_, alpha);
    }


    void sqrt_d(cuMat &r, float alpha){
        mat_sqrt_d_kernel_exec(m_device_, r.m_device_, cols_, rows_, alpha);
    }

    void sin(cuMat &r){
        mat_sin_kernel_exec(m_device_, r.m_device_, cols_, rows_, 0);
    }

    void cos(cuMat &r){
        mat_cos_kernel_exec(m_device_, r.m_device_, cols_, rows_, 0);
    }

    void relu(cuMat &r){
        relu_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void relu_d(cuMat &r){
        relu_d_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void prelu(cuMat &a, cuMat &r){
        prelu_kernel_exec(m_device_, a.m_device_, r.m_device_, cols_, rows_);
    }

    void prelu_d(cuMat &a, cuMat &r, cuMat &da){
        prelu_d_kernel_exec(m_device_, a.m_device_, r.m_device_, da.m_device_, cols_, rows_);
    }

    void sigmoid(cuMat &r){
        sigmoid_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void sigmoid_d(cuMat &r){
        sigmoid_d_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void tanh(cuMat &r){
        tanh_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void tanh_d(cuMat &r){
        tanh_d_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void softmax(cuMat &r){
        softmax_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void exp(cuMat &r){
        mat_exp_kernel_exec(m_device_, r.m_device_, cols_, rows_, 1e-8);
    }

    void inverse(cuMat &r){
        mat_inverse_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }

    void inverse_d(cuMat &r){
        mat_inverse_d_kernel_exec(m_device_, r.m_device_, cols_, rows_);
    }
};
#endif
