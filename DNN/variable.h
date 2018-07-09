#ifndef VARIABLE_H_
#define VARIABLE_H_

#include <list>
#include <random>
#include <memory>
#include <boost/intrusive_ptr.hpp>

#include "../cuMat/cuMat.h"

class FunctionBase;

class VariableBase{
private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version){
        ar & id_;
        ar & data_;
        ar & grad_;
        ar & seed_;
        ar & is_get_grad_;
    }
public:
    int id_ = 0;
    int opt_ = 0;
    int *last_opt_ = nullptr;
    bool *is_last_backward_ = nullptr;

    int forward_count_ = 0;

    FunctionBase *creator_ = nullptr;
    std::string name_;

    cuMat data_;
    cuMat grad_;
    cuMat seed_;

    int grad_num_ = -999;
    // この変数のgradを求めるかどうか
    bool is_get_grad_ = true;
    bool is_sparse = false;

    VariableBase();
    // Copy constructor
    VariableBase(const VariableBase &rhs);
    VariableBase(int rows, int cols);
    VariableBase(int rows, int cols, bool is_get_grad);
    VariableBase(FunctionBase *creator, int rows, int cols);
    VariableBase(cuMat &input);
    VariableBase(FunctionBase *creator, cuMat &input);
    VariableBase(std::vector<float> &ids, int nums);
    ~VariableBase();

    void creatorSset(FunctionBase *creator);
    // substitution operator
    VariableBase& operator=(const VariableBase &rhs);

    VariableBase sin();
    VariableBase log();

    void backward();
    void backward(VariableBase *v);

    // 逆伝播ではgradを足していくので、一番最後まで到達したら再び0にする
    void zero_grads();
    void zero_grads(VariableBase *v);

    void ones();
    void zeros();
    void unchain();

    void randoms(float m, float a);
    void binomial_randoms(float ratio);

    float val();
};

using Variable = std::shared_ptr<VariableBase>;

VariableBase *variable_construct(int rows, int cols);
void variable_destroy(VariableBase *ptr);

#endif
