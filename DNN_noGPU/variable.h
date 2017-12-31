#ifndef VARIABLE_H_
#define VARIABLE_H_

#include <list>
#include <random>
#include <memory>
#include <boost/intrusive_ptr.hpp>
#include <boost/serialization/serialization.hpp>

#include <eigen3/Eigen/Dense>

#include "function.h"

class Function;

class Variable{
public:
    int id_ = 0;
    int opt_ = 0;
    int *last_opt_ = nullptr;
    int *is_last_backward_ = nullptr;

    int forward_count_ = 0;

    Function *creator_ = nullptr;

    std::string name_;

    Eigen::MatrixXf data_;
    Eigen::MatrixXf grad_;
    Eigen::MatrixXf seed_;

    int grad_num_ = -999;
    bool is_get_grad_ = true;
    bool is_sparse = true;

    friend class boost::serialization::access;
    template<class Archive> void serializae(Archive &ar, const unsigned int version){
        ar & id_;
        ar & data_;
        ar & grad_;
        ar & seed_;
        ar & is_get_grad_;
    }
public:
    //constructors
    Variable();
    Variable(int rows, int cols);
    Variable(int rows, int cols, bool is_get_grad);
    Variable(Function *f, int rows, int cols);
    Variable(Eigen::MatrixXf &input);
    Variable(Function *f, Eigen::MatrixXf &input);
    Variable(std::vector<float> &ids, int nums);

    ~Variable();

    //accesser, setter
    void CreatorSet(Function *f);
    Variable &operator=(const Variable &v);

    Variable sin();
    Variable log();

    void backward();
    void backward(Variable *v);

    void zero_grads();
    void zero_grads(Variable *V);

    void ones();
    void zeros();
    void unchain();
    void zero_grad();

    void randoms(float m, float a);

    void binomial_randoms(float ratio);

    float val();

};

using Variable_s =  std::shared_ptr<Variable>;

Variable *variable_construct(int rows, int cols);
void variable_destroy(Variable *ptr);

#endif
