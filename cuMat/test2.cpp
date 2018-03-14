#include "cuMat.hpp"

MallocCounter mallocCounter;

int main(){
    cuMat A(3, 3);
    for(int i=0; i<A.getRow(); i++){
        for(int j=0; j<A.getCol(); j++){
            A.memSetHost(i, j, j+i*A.getCol());
        }
    }
    A.memHostToDevice();

    std::cout << A;

    cuMat B(3, 3);
    B = A.cos(); std::cout << B; //OK
    B = A.sin(); std::cout << B; //OK
    B = A.exp(); std::cout << B; //OK
    B = A.log(); std::cout << B; //OK
    B = A.sqrt(); std::cout << B;//OK

    float l = A.l2();
    std::cout << l << std::endl; //OK
}
