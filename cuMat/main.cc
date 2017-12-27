#include <iostream>
#include "add.h"

int main(){
    float x1[N][N];
    float x2[N][N];
    float y[N][N];
    
    for(int i=0; i<N; i++){
	for(int j=0; j<N; j++){
	    x1[i][j] = x2[i][j] = j + i*N;
	}
    }
    
    MatAdd_exec((float**)x1, (float**)x2, (float**)y);

    for(int i=0; i<N; i++){
	for(int j=0; j<N; j++){
	    std::cout << y[i][j] << " ";
	}
	std::cout << std::endl;
    }

    return 0;
}
