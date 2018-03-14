#include <list>
#include <random>
#include <iostream>
#include <chrono>

#include "variable.h"
#include  "function.h"

int count_fuonction = 0;
int count_variable = 0;

std::map<Variable_base*, bool> variable_pool;

Variable_base * variable_construct(int rows, int cols){
    count_variable++;
    for(auto itr=variable_pool.begin(); itr != variable_pool.end(); ++itr){
        if(!itr->second){
            Variable_base *v = static_cast<Variable_base *>(itr->first);
            if(v->data_.getRow() == rows && v->data_.getCol() == cols){
                v->zeros();
                v->creator_ = nullptr;
                variable_pool[v] = true;

                return v;
            }
        }
    }
    Variable_base *r = new Variable_base(rows, cols);
    variable_pool[r] = true;

    return r;
}

void variable_destroy(Variable_base *ptr){
    count_variable--;
    variable_pool[ptr] = false;
    if(variable_pool.size() > 4000){
        variable_pool.erase(ptr);
        delete ptr;
    }
}

int gVariableId = 0;

Variable_base::Variable_base() : id_(gVariableId) {
    gVariableId++;
}

Variable_base::Variable_base(const Variable_base &a){
    id_ = gVariableId;
    gVariableId++;

    data_ = a.data_;
    grad_ = a.grad_;

    seed_ = a.seed_;
    creator_ = a.creator_;

    this->is_get_grad_ = a.is_get_grad_;
}

Variable_base::Variable_base(int rows, int cols) :
    id_(gVariableId),
    data_(cuMat(rows, cols)),
    grad_(cuMat(rows, cols)),
    seed_(cuMat(grad_.getRow(), grad_.getCol())),
    creator_(nullptr)
{
    gVariableId++;
    seed_.ones();
}

Variable_base::Variable_base(int rows, int cols, bool is_get_grad) :
    is_get_grad_(is_get_grad),
    id_(gVariableId),
    data_(cuMat(rows, cols)),
    grad_(cuMat(rows, cols)),
    seed_(cuMat(grad_.getRow(), grad_.getCol())),
    creator_(nullptr)
{
    gVariableId++;
    seed_.ones();
}

Variable_base::Variable_base(cuMat &input):
    id_(gVariableId),
    data_(input),
    grad_(cuMat(input.getRow(), input.getCol())),
    seed_(cuMat(grad_.getRow(), grad_.getCol())),
    creator_(nullptr)
{
    gVariableId++;
    seed_.ones();
}

Variable_base::Variable_base(Function_base *f, int rows, int cols):
    id_(gVariableId),
    data_(cuMat(rows, cols)),
    grad_(cuMat(rows, cols)),
    seed_(cuMat(grad_.getRow(), grad_.getCol())),
    creator_(f)
{
    gVariableId++;
    seed_.ones();
}

Variable_base::Variable_base(Function_base *f, cuMat &input):
    id_(gVariableId),
    data_(input),
    grad_(cuMat(input.getRow(), input.getCol())),
    seed_(cuMat(grad_.getRow(), grad_.getCol())),
    creator_(f)
{
    gVariableId++;
    seed_.ones();
}

Variable_base::~Variable_base(){
}

Variable_base &Variable_base::operator=(const Variable_base &v){
    id_ = gVariableId;
    gVariableId++;
    data_ = v.data_;
    grad_ = v.grad_;
    seed_ = v.seed_;
    creator_ = v.creator_;
    this->is_get_grad_ = v.is_get_grad_;

    return *this;
}

void Variable_base::creatorSet(Function_base *f){
    this->creator_ = f;
}

void Variable_base::backward(){
    this->grad_ = seed_;
    this->backward(this);
}

void Variable_base::backward(Variable_base *v){
    if(v == nullptr) return;

    if(v->creator_ != nullptr){
        if(v->last_opt_ != nullptr && v->opt_ == *v->last_opt_){
            *v->is_last_backward_ = true;
        }
        if(v->forward_count_ != 0) v->forward_count_--;
        if(v->is_last_backward_ != nullptr && *v->is_last_backward_ == false) return;
        if(v->forward_count_ != 0) return;

        v->creator_->backward(v->grad_);

        for(int i=0; i<v->creator_->inputs.size(); i++){
            Variable nv = v->creator_->inputs[i];
            if(nv->is_get_grad_) this->backward(nv.get());
        }
    }
    else {}
}

void Variable_base::zero_grads(){
    this->zero_grads(this);
}

void Variable_base::zero_grads(Variable_base *v){
    if(v == nullptr) return;
    v->grad_.mul(0, v->grad_);
    v->forward_count_ = 0;

    if(v->creator_ != nullptr){
        for(int i=0; i<v->creator_->inputs.size(); i++){
            Variable nv = v->creator_->inputs[i];
            this->zero_grads(nv.get());
        }
    }
}

void Variable_base::ones(){
    data_.ones();
    grad_.mul(0, grad_);
}

void Variable_base::zeros(){
    data_.mul(0, data_);
    grad_.mul(0, grad_);
    forward_count_ = 0;
    last_opt_ = nullptr;
    is_last_backward_ = nullptr;
    this->creator_ = nullptr;
}

void Variable_base::unchain(){
    this->creator_ = nullptr;
}

void Variable_base::randoms(float m, float a){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<float> initd1(m, a);

    for(int i=0; i<data_.getRow(); i++){
        for(int j=0; j<data_.getCol(); j++){
            data_.memSetHost(i, j, initd1(mt));
        }
    }
    data_.memHostToDevice();
}

void Variable_base::binomial_randonms(float ratio){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> initd1(0.0, 1.0);

    for(int i=0; i<data_.getRow(); i++){
        for(int j=0; j<data_.getCol(); j++){
            float h = 1.0;
            if(initd1(mt) < ratio) h = 0.0;
            data_.memSetHost(i, j, h);
        }
    }
    data_.memHostToDevice();
}

float Variable_base::val(){
    return data_(0, 0);
}

