#ifndef VARIABLE_H_
#define VARIABLE_H_

#include <list>
#include <memory>
#include <boost/intrusive_ptr.hpp>

#include "../cuMat/cuMat.hpp"

using namespace std;

class Function_;

class Variable_{
public:
    int id_ = 0;
    int opt_ = 0;
    int *last_opt_ = nullptr;
    bool *is_las_backward_ = nullptr;

    int foward_count_ = 0;
    Function_ *creator = nullptr;

    string name_;

    cuMat data_;
    cuMat grad_;
    cuMat seed_;

    int grad_num_ = -999;

    bool is_get_grad_ = true;

    Variable_();
}
#endif
