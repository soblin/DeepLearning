#!/bin/bash

echo type "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mamrou-home/DeepLearning/cuMat" before run a.out
g++ -std=c++11 -Wall -g -o test1 test1.cpp -L/usr/local/cuda-9.1/lib64 -L./ -I/usr/local/cuda-9.1/include -lcublas -lm -lcudart -lcumat

g++ -std=c++11 -Wall -g -o test2 test2.cpp -L/usr/local/cuda-9.1/lib64 -L./ -I/usr/local/cuda-9.1/include -lcublas -lm -lcudart -lcumat
