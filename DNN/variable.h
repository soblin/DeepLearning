#ifndef VARIABLE_H_
#define VARIABLE_H_

#include <list>
#include <memory>
#include <boost/intrusive_ptr.hpp>

#include "../cuMat/cuMat.h"

class Function_base;

class Variable_base{
public:
    int id_ = 0;
    int opt_ = 0;
    int *last_opt_ = nullptr;
    bool *is_last_backward_ = nullptr;

    int forward_count_ = 0;
    Function_base *creator_ = nullptr;

    std::string name_;

    cuMat data_;
    cuMat grad_;
    cuMat seed_;

    int grad_num_ = -999;

    bool is_get_grad_ = true;

    Variable_base();
    Variable_base(int row, int col);
    Variable_base(int row, int col, bool is_get_grad);
    Variable_base(Function_base *f, int row, int col);
    Variable_base(Function_base *f, cuMat &input);
    Variable_base(cuMat &input);
    Variable_base(std::vector<float> &ids, int nums);
    Variable_base(const Variable_base &v);
    Variable_base &operator=(const Variable_base &v);
    ~Variable_base();

    void creatorSet(Function_base *f);
    Variable_base sin();
    Variable_base log();

    void backward();
    void backward(Variable_base *v);

    void zero_grads();
    void zero_grads(Variable_base *v);

    void ones();
    void zeros();
    void unchain();
    void zero_grad();

    void randoms(float m, float a);
    void binomial_randonms(float ratio);

    float val();
};

using Variable = std::shared_ptr<Variable_base>;

Variable_base *variable_construct(int row, int col);
void variable_destroy(Variable_base *ptr);

#endif
