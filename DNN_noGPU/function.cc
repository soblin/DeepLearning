#include <list>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include <math.h>
#include <boost/pool/object_pool.hpp>
#include <sstream>

#include "function.h"
#include "variable.h"

int kfunc_id = 0;

Variable_s variable_construct_for_function(Function *f, int rows, int cols){
    Variable_s r = Variable_s(variable_construct(rows, cols), variable_destroy);
    r->creator_ = f;

    return r;
}

Function::Function(){
    name_ = "Funciton";
    this->id_ = kfunc_id;
    kfunc_id++;
    count_function++;
}

Function::~Function(){
    init();
    count_function--;
}

void Function::init(){
    inputs_.clear();
    outputs_.clear();
}

Variable_s Function::forward(const Variable_s v){
    v->forward_count_++;
    inputs_.push_back(v);

    Variable_s ret = forward(inputs_, outputs_);
    return ret;
}

Variable_s Function::forward(const Variable_s v1, const Variable_s v2){
    v1->forward_count_++;
    v2->forward_count_++;
    inputs_.push_back(v1);
    inputs_.push_back(v2);
    Variable_s ret = forward(inputs_, outputs_);

    return ret;
}

Variable_s Function::forward(const Variable_s v1, const Variable_s v2, const Variable_s v3){
    v1->forward_count_++;
    v2->forward_count_++;
    v3->forward_count_++;

    inputs_.push_back(v1);
    inputs_.push_back(v2);
    inputs_.push_back(v3);
    Variable_s ret = forward(inputs_, outputs_);

    return ret;
}

Variable_s Function::forward(const Variable_s v1, const Variable_s v2, const Variable_s v3, const Variable_s v4){
    v1->forward_count_++;
    v2->forward_count_++;

    inputs_.push_back(v1);
    inputs_.push_back(v2);
    inputs_.push_back(v3);
    inputs_.push_back(v4);
    Variable_s ret = forward(inputs_, outputs_);

    return ret;
}

Variable_s Function::forward(const Variable_s v1, const Variable_s v2, const Variable_s v3, const Variable_s v4,
                             const Variable_s v5, const Variable_s v6, const Variable_s v7, const Variable_s v8,
                             const Variable_s v9, const Variable_s v10, const Variable_s v11, const Variable_s v12){
    v1->forward_count_++;
    v2->forward_count_++;

    inputs_.push_back(v1);
    inputs_.push_back(v2);
    inputs_.push_back(v3);
    inputs_.push_back(v4);
    inputs_.push_back(v5);
    inputs_.push_back(v6);
    inputs_.push_back(v7);
    inputs_.push_back(v8);
    inputs_.push_back(v9);
    inputs_.push_back(v10);
    inputs_.push_back(v11);
    inputs_.push_back(v12);

    Variable_s ret = forward(inputs_, outputs_);

    return ret;
}

void Function::backward(const Eigen::MatrixXf &p_grad){
    backward(p_grad, inputs_, outputs_);
}

void Function::clip_grad(Variable *v){
    float clip_grad_threashold = 5.0;
    float sq = v->grad_.norm();
    float rate = clip_grad_threashold/sq;

    if(rate < 1.) v->grad_ *= rate;
}

void Function::reset_state(){}

FunctionPlus::FunctionPlus() : Function(){
    name_ = "FunctionPlus";
}

Variable_s FunctionPlus::forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs){
    Variable_s v1 = inputs.at(0);
    Variable_s v2 = inputs.at(1);

    Variable_s ret = variable_construct_for_function(this, v1->data_.rows(), v2->data_.cols());

    ret->data_ = v1->data_ + v2->data_;

    return ret;
}

void FunctionPlus::backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &ouputs){
    Variable_s v1 = inputs.at(0);
    Variable_s v2 = inputs.at(1);

    if(v1->is_get_grad_) v1->grad_ += p_grad;
    if(v2->is_get_grad_) v2->grad_ += p_grad;
}

FunctionMinus::FunctionMinus() : Function() {
    name_ = "FunctionMinus";
}

Variable_s FunctionMinus::forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs){
    Variable_s v1 = inputs.at(0);
    Variable_s v2 = inputs.at(1);

    Variable_s ret = variable_construct_for_function(this, v1->data_.rows(), v2->data_.cols());

    ret->data_ = v1->data_ + v2->data_;
    return ret;
}

void FunctionMinus::backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs){
    Variable_s v1 = inputs.at(0);
    Variable_s v2 = inputs.at(1);

    if(v1->is_get_grad_) v1->grad_ += p_grad;
    if(v2->is_get_grad_) v2->grad_ -= p_grad;
}

FunctionMul::FunctionMul() : Function(){
    name_ = "FunctionMul";
}

Variable_s FunctionMul::forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs){
    Variable_s v1 = inputs.at(0);
    Variable_s v2 = inputs.at(1);

    Variable_s ret = variable_construct_for_function(this, v1->data_.rows(), v2->data_.cols());

    ret->data_ = v1->data_.array() * v2->data_.array();

    return ret;
}

void FunctionMul::backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs){
    Variable_s v1 = inputs.at(0);
    Variable_s v2 = inputs.at(1);

    if(v1->is_get_grad_) v1->grad_.array() += p_grad.array() * v2->data_.array();
    if(v2->is_get_grad_) v2->grad_.array() += p_grad.array() * v1->data_.array();
}

FunctionInverse::FunctionInverse() : Function(){
    int a = 0;
    name_ = "FunctionInverse";
}

Variable_s FunctionInverse::forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs){
    Variable_s v = inputs.at(0);

    Variable_s ret = variable_construct_for_function(this, v->data_.rows(), v->data_.cols());

    outputs_.push_back(ret);

    return ret;
}
