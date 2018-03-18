#include <list>
#include <random>
#include <vector>
#include <map>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>


#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "variable.h"

extern std::map<Variable *, bool> obj_pool2;
extern int count_funciton;
extern int count_variable;

class FunctionBase{
public:
    std::vector<Variable> inputs;
    std::vector<Variable> outputs;

    int id = -1;
    std::string name_;
    std::string custom_name_;
    int inner_count_ = 0;

    FunctionBase();
    virtual ~FunctionBase();

    virtual Variable forward(const Variable input);
    virtual Variable forward(const Variable x, const Variable t);
    virtual Variable forward(const Variable input1, const Variable input2, const Variable inputs3);
    virtual Variable forward(const Variable input1, const Variable input2, const Variable input3, const Variable input4);
    virtual Variable forward(const Variable input1, const Variable input2, const Variable input3, const Variable input4,
                             const Variable input5, const Variable input6, const Variable input7, const Variable input8,
                             const Variable input9, const Variable input10, const Variable input11, const Variable input12);

    virtual void backward(cuMat &p_grad);
    virtual Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    virtual void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);

    void init();
    void clip_grad(VariableBase *v);

    virtual void reset_state();
    
};

class FunctionPlus : public FunctionBase{
public:
    FunctionPlus();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FuncitonMinus : public FunctionBase{
public:
    FuncitonMinus();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void  backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionMul : public FunctionBase{
    FunctionMul();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSin : public FunctionBase{
public:
    FunctionSin();
    Variable rr = nullptr;
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

//PVariable 
class FunctionCos : public FunctionBase{
public:
    FunctionCos();
    Variable rr = nullptr;
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionLog : public FunctionBase{
public:
    FunctionLog();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSqrt : public FunctionBase{
public:
    FunctionSqrt();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionInverse : public FunctionBase{
public:
    FunctionInverse();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionLinear : public FunctionBase{
public:
    VariableBase *w;
    VariableBase *b;
    cuMat i1;

    bool no_bias_ = false;
    bool is_transpose_ = false;

    FunctionLinear();
    FunctionLinear(VariableBase *w, VariableBase *b, bool is_transpose = false);
    FunctionLinear(VariableBase *w, bool is_transpose = false);
    FunctionLinear(int output_size, int input_size);
    FunctionLinear(int output_size ,int input_size, bool no_bias);
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void toHostArray();
    void formHostArray();
};

class FunctionReLU : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionReLU();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionPReLU : public FunctionBase{
public:
    Variable *a;
    Variable xd = nullptr;
    Variable ad = nullptr;
    FunctionPReLU();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSigmoid : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionSigmoid();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionTanh : public FunctionBase{
public:
    Variable rr = nullptr;
    FunctionTanh();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSoftmax : public FunctionBase{
public:
    FunctionSoftmax();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionSoftmaxCrossEntropy : public FunctionBase{
public:
    Variable rr = nullptr;
    Variable rr2 = nullptr;
    Variable rr3 = nullptr;
    cuMat loss;
    cuMat *seed = nullptr;

    FunctionSoftmaxCrossEntropy();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionMeanSquareError : public FunctionBase{
public:
    Variable rr = nullptr;
    cuMat loss;
    FunctionMeanSquareError();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};

class FunctionIdentity : public FunctionBase{
public:
    FunctionIdentity();
    Variable forward(const std::vector<Variable> &inputs, std::vector<Variable> &ouputs);
    void backward(cuMat &p_grad, const std::vector<Variable> &inputs, std::vector<Variable> &outputs);
};


#endif
