#include <list>
#include <map>
#include <random>
#include <chrono>
#include <iostream>
#include <memory>
#include <math.h>
#include <boost/pool/object_pool.hpp>
#include <sstream>

#include "function.h"

using namespace std;

int func_id = 0;

PVariable variable_construct_for_function(Function *f, int rows, int cols){
  PVariable r = PVariable(variable_construct(rows, cols), variable_destroy);
  r->creator = f;

  return f;
}

Function::Function(){
  name = "Function";
  this->id = func_id;
  func_id++;
  count_function++;
}

Function::~Function(){
  init();
  count_function--;
}

void Function::init(){
  inputs.clear();
  outputs.clear();
}

PVariable Function::forward(vector<PVariable> &inputs, vector<PVariable> &outputs){ return NULL;}
void Function::backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs){}

PVariable Function::forward(PVariable v){
  v->forward_count++;

  inputs.push_back(v);
  PVariable r = forward(inputs, outputs);

  return r;
}

PVariable Function::forward(PVariable v1, PVariable v2){
  v1->forward_count++;
  v2->forward_count++;

  inputs.push_back(v1);
  inputs.push_back(v2);
  PVariable r = forward(inputs, outputs);

  return r;
}

