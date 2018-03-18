#ifndef VARIABLE_H_
#define VARIABLE_H_

#include <list>
#include <memory>
#include <boost/intrusive_ptr.hpp>

#include "../cuMat/cuMat.h"

class FunctionBase;

class VariableBase{
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

    bool is_get_grad_ = true;

    VariableBase();
    VariableBase(int row, int col);
    VariableBase(int row, int col, bool is_get_grad);
    VariableBase(FunctionBase *f, int row, int col);
    VariableBase(FunctionBase *f, cuMat &input);
    VariableBase(cuMat &input);
    VariableBase(std::vector<float> &ids, int nums);
    VariableBase(const VariableBase &v);
    VariableBase &operator=(const VariableBase &v);
    ~VariableBase();

    void creatorSet(FunctionBase *f);
    VariableBase sin();
    VariableBase log();

    void backward();
    void backward(VariableBase *v);

    void zero_grads();
    void zero_grads(VariableBase *v);

    void ones();
    void zeros();
    void unchain();
    void zero_grad();

    void randoms(float m, float a);
    void binomial_randonms(float ratio);

    float val();
};

using Variable = std::shared_ptr<VariableBase>;

VariableBase *variable_construct(int row, int col);
void variable_destroy(VariableBase *ptr);

#endif
