#include <list>
#include <random>
#include <vector>
#include <map>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

using namespace std;

#ifndef _FUNCTION_
#define _FUNCTION_

#include "variable.h"

extern map<Variable *, bool> obj_pool2;
extern int count_function;
extern int count_variable;

class Function{
public:

    vector<PVariable> inputs;
    vector<PVariable> outputs;

    int id = -1;
    string name;
    sriing custom_name;
    int inner_count = 0;

    Function();
    virtual ~Function();

    //これから派生する各関数(+, -. *, /, sin, cos, etc)それぞれでforward backwardの動きが違うため、vritualにする
    virtual PVariable forward(PVariable input);
    virtual PVariable forward(PVariable x, PVariable t);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3, PVariable input4);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3, PVariable input4, PVariable input5, PVariable input6, PVariable input7, PVariable input8, PVariable input8, PVariable input9, PVariable input10, PVariable input11, PVariable input12);

    virtual void backward(cuMat &p_grad);
    virtual PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    virtual void backward(cuMat &p_grad, vector<PVarible> &inputs, vector<PVariable> &outputs);

    void init();

    void clip_grad(Variable *v);

    virtual void reset_state();

private:
    friend class boost::serializatopn::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version)
    {}
};

class FunctionPlus: public Function{
public:
    FunctionPlus();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionMinus: public Function{
public:
    FunctionMinus();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionMul: public Function{
public:
    FunctionMul();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    PVariable backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionSin: public Function{
public:
    PVariable rr = NULL;
    FunctionSin();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionCos: public Function{
public:
    PVarialbe rr = NULL;
    FunctionCos();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionLog: public Function{
public:
    FunctionLog();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionSqrt: public Function{
public:
    FunctionLog();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionInverse: public Function{
public:
    FunctionInverse();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariavble> &outputs);
};

class FunctionLinear: public Function{
public:
    Variable *w;
    Variable *b;
    cuMat i1;

    bool noBias = false;
    bool isTranspose = false;
    
}
#endif
