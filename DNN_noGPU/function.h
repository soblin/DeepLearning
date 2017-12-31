#include <list>
#include <map>
#include <random>
#include <vector>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>

#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "variable.h"

extern std::map<Variable *, bool> obj_pool2;
extern int count_function;
extern int count_variable;

class Function{
public:
    std::vector<Variable_s> inputs_;
    std::vector<Variable_s> outputs_;

    int id_ = -1;
    std::string name_;
    std::string custom_name_;
    int inner_count_ = 0;

    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar ,const unsigned int version){}
public:
    Function();
    virtual ~Function();
    
    virtual Variable_s forward(const Variable_s inputs);
    virtual Variable_s forward(const Variable_s x, Variable_s t);
    virtual Variable_s forward(const Variable_s input1, const Variable_s input2, const Variable_s input3);
    virtual Variable_s forward(const Variable_s input1, const Variable_s input2, const Variable_s input3, const Variable_s input4);
    virtual Variable_s forward(const Variable_s input1, const Variable_s input2, const Variable_s input3,
                              const Variable_s input4, const Variable_s input5, const Variable_s input6,
                              const Variable_s input7, const Variable_s input8, const Variable_s input9,
                              const Variable_s input10, const Variable_s input11, const Variable_s input12);

    virtual void backward(const Eigen::MatrixXf &p_grad);
    virtual Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    virtual void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);

    void init();

    void clip_grad(Variable *v);

    virtual void reset_state();
};

class FunctionPlus : public Function{
public:
    FunctionPlus();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
};

class FunctionMinus : public Function{
public:
    FunctionMinus();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
};

class FunctionMul : public Function{
public:
    FunctionMul();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs,std::vector<Variable_s> &outputs);
};

class FunctionSin : public Function{
public:
    Variable_s rr = nullptr;
    FunctionSin();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs,std::vector<Variable_s> &outputs);
};

class FunctionCos : public Function{
public:
    Variable_s rr = nullptr;
    FunctionCos();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &ouputs);
};

class FunctionLog : public Function{
public:
    FunctionLog();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &ouputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
};

class FunctionSqrt : public Function{
public:
    FunctionSqrt();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inpus, std::vector<Variable_s> &outputs);
};

class FunctionInverse : public Function{
public:
    FunctionInverse();
    Variable_s forward(const std::vector<Variable_s> &inpus, std::vector<Variable_s> &ouputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inups, std::vector<Variable_s> &outputs);
};

class FunctionLinear : public Function{
private:
    Variable *W_;
    Variable *B_;

    Eigen::MatrixXf i1_;

    bool no_bias_ = false;
    bool is_transpose_ = false;

    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version){
        ar & boost::serialization::base_object<Function>(*this);
        ar & W_;
        ar & B_;
        ar & i1_;
        ar & no_bias_;
        ar & is_transpose_;
    }
public:
    FunctionLinear();
    FunctionLinear(Variable *W, Variable *B, bool is_transpose = false);
    FunctionLinear(Variable *W, bool is_transpose_ = false);
    FunctionLinear(int output_size, int input_size);
    FunctionLinear(int output_size, int input_size, bool no_bias);
    Variable_s forward(Eigen::MatrixXf &p_grad, std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
};

class FunctionEmbed : public Function{
private:
    Variable W_;
    Variable B_;
    Eigen::MatrixXf i1_;

    Eigen::MatrixXf wt_;
    Eigen::MatrixXf rt_;
    Eigen::MatrixXf rtmp_;

    bool no_bias_ = false;
public:
    FunctionEmbed();
    FunctionEmbed(int output_size, int input_size, bool no_bias);

    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inpus, std::vector<Variable_s> &outputs);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version){
        ar & boost::serialization::base_object<Function>(*this);
        ar & W_;
        ar & B_;
        ar & i1_;
        ar & no_bias_;
    }
};

class FunctionReLU : public Function {
public:
    Variable_s rr = nullptr;
    FunctionReLU();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
};

class FunctionPReLU : public Function{
public:
    Variable *a;
    Variable_s xd = nullptr;
    Variable_s ad = nullptr;
    FunctionPReLU();
    FunctionPReLU(Variable *);
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);    
};

class FunctionSigmoid : public Function{
public:
    Variable_s rr = nullptr;
    FunctionSigmoid();
    Variable_s forward(const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
    void backward(const Eigen::MatrixXf &p_grad, const std::vector<Variable_s> &inputs, std::vector<Variable_s> &outputs);
};

#endif
