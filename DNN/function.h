#include <list>
#include <random>
#include <vector>
#include <map>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "variable.h"

extern std::map<VariableBase *, bool> obj_pool2;
extern int count_function;
extern int count_variable;

class FunctionBase{
public:
    // Record the Variable which was substitued into this funciton
    std::vector<Variable> inputs;
    // Record the Variable which was returned from this function
    std::vector<Variable> outputs;

    int id_ = -1;
    std::string name_;
    std::string custom_name_;
    int inner_count_ = 0;

    FunctionBase();
    virtual ~FunctionBase();

    // Normal forward-propagation like FunctionSqrt(Variable) -> Variable
    virtual Variable forward(Variable input);
    // Bi-Arg functions like loss functions
    virtual Variable forward(Variable x, Variable t);
    virtual Variable forward(Variable input1, Variable input2, Variable input3);

    virtual void backward(cuMat &output_grad);

    // Core propagation functions
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);

    void init();
    void clip_grad(VariableBase *v);
    virtual void reset_state();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version){}
};

class FunctionPlus : public FunctionBase{
public:
    FunctionPlus();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};
 
class FunctionMinus : public FunctionBase{
public:
    FunctionMinus();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionMul : public FunctionBase{
public:
    FunctionMul();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSin : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionSin();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionCos : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionCos();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionLog : public FunctionBase{
public:
    FunctionLog();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionInverse : public FunctionBase{
public:
    FunctionInverse();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSqrt : public FunctionBase{
public:
    FunctionSqrt();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionLinear : public FunctionBase{
public:
    VariableBase *w, *b;
    cuMat i1;

    bool no_bias_ = false;
    bool is_transpose_ = false;

    FunctionLinear();
    FunctionLinear(VariableBase *w, VariableBase *b, bool is_transpose = false);
    FunctionLinear(VariableBase *w, bool is_transpose = false);
    FunctionLinear(int output_size, int input_size);
    FunctionLinear(int output_size, int input_size, bool no_bias);

    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void toHostArray();
    void formHostArra();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version){
        ar & boost::serialization::base_object<FunctionBase>(*this);
        ar & * w;
        ar & b;
        ar & i1;
        ar & no_bias_;
        ar & is_transpose_;
    }
};

class FunctionReLU : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionReLU();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionPReLU : public FunctionBase{
    VariableBase *a;
    // differential of x
    Variable x_d = nullptr;
    // differentail of d
    Variable a_d = nullptr;
    FunctionPReLU();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSigmoid : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionSigmoid();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backwrad(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionTanh : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionTanh();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inpus, std::vector<Variable> &outputs);
};

class FunctionSoftmax : public FunctionBase{
public:
    FunctionSoftmax();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outpus);
};

class FunctionSoftmaxCrossEntropy : public FunctionBase{
public:
    Variable rr1 = nullptr, rr2 = nullptr, rr3 = nullptr;
    cuMat loss;
    cuMat *seed = nullptr;

    FunctionSoftmaxCrossEntropy();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionMeanSquaredError : public FunctionBase{
public:
    Variable rr = nullptr;
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionIdentity : public FunctionBase{
public:
    FunctionIdentity();
    virtual Variable forward(std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void bakcward(cuMat &output_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};
#endif
