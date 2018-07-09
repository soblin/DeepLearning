#include <list>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include <cmath>
#include <boost/pool/object_pool.hpp>
#include <sstream>

#include "function.h"

int func_id = 0;

Variable variable_construct_for_function(FunctionBase *f, int rows, int cols){
    Variable r = Variable(variable_construct(rows, cols), variable_destroy);
    r->creator_ = f;
    return r;
}


FunctionBase::FunctionBase():
    name_("Function"),
    id_(func_id)
{
    func_id++;
    count_function++;
}

void FunctionBase::init(){
    inputs.clear();
    outputs.clear();
}

Variable FunctionBase::forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs){
    return nullptr;
}

void FunctionBase::backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs){}

Variable FunctionBase::forward(Variable v){
    v->forward_count_++;
    inputs.push_back(v);
    Variable r = forward(inputs, outputs);
    return r;
}

Variable FunctionBase::forward(Variable x, Variable t){
    x->forward_count_++;
    t->forward_count_++;

    inputs.push_back(x);
    inputs.push_back(t);
    Variable r = forward(inputs, outputs);
    return r;
}

Variable FunctionBase::forward(Variable input1, Variable input2, Variable input3){
    input1->forward_count_++;
    input2->forward_count_++;
    input3->forward_count_++;

    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    Variable r = forward(inputs, outputs);

    return r;
}

void FunctionBase::backward(cuMat &output_grad){
    backward(output_grad, inputs, outputs);
}


void FunctionBase::clip_grad(VariableBase *v){
    float clip_grad_threashold = 5.0;
    float sq = v->grad_.l2();
    float rate = clip_grad_threashold/sq;
    if(rate < 1.) v->grad_.mul(rate, v->grad_);
}

void FunctionBase::reset_state(){}

FunctionPlus::FunctionPlus() : FunctionBase(){
    name_ = "FunctionPlus";
}

Variable FunctionPlus::forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs){
    //2変数なのでinputは２つあるはず。その合計を返す
    Variable v1 = inputs.at(0);
    Variable v2 = inputs.at(1);

    Variable r = variable_construct_for_function(this, v1->data_.getRow(), v1->data_.getCol());
    outputs.push_back(r);
    v1->data_.plus(v2->data_, r->data_);
    return r;
}
void FunctionPlus::backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs){
    Variable v1 = inputs.at(0);
    Variable v2 = inputs.at(1);
    // c = a + b
    // dE/da = dE/dc, dE/db = dE/dc プラスの場合はそのまま
    if(v1->is_get_grad_) output_grad.mul_plus(1.0, v1->grad_);
    if(v2->is_get_grad_) output_grad.mul_plus(1.0, v2->grad_);
}

FunctionMinus::FunctionMinus() : FunctionBase(){
    name_ = "FunctionMinus";
}

Variable FunctionMinus::forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs){
    Variable v1 = inputs.at(0);
    Variable v2 = inputs.at(1);

    Variable r = variable_construct_for_function(this, v1->data_.getRow(), v1->data_.getCol());
    outputs.push_back(r);
    v1->data_.minus(v2->data_, r->data_);

    return r;
}

void FunctionMinus::backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs){
    Variable v1 = inputs.at(0);
    Variable v2 = inputs.at(1);
    // c = a - b
    // dE/da = dE/dc, dE/db = -dE/dc
    if(v1->is_get_grad_) output_grad.mul_plus(1.0, v1->grad_);
    if(v2->is_get_grad_) output_grad.mul_plus(-1.0, v2->grad_);
}
