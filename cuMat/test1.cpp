#include "cuMat.hpp"

MallocCounter mallocCounter;

int main(){
    cuMat A(3, 3);
    std::cout << "Test of initialization." << std::endl;
    std::cout << A;               //OK OK

    std::cout << "Test of memSetHost and memHostToDevice." << std::endl;
    for(int i=0; i<A.row(); i++){
         for(int j=0; j<A.col(); j++){
             A.memSetHost(i,j, j+i*A.col());
         }
     }
    A.memHostToDevice();
    std::cout << A;               //OK OK
    cuMat B(3, 3);
    for(int i=0; i<B.row(); i++){
        for(int j=0; j<B.col(); j++){
            B.memSetHost(i, j, 3);
        }
    }
    B.memHostToDevice();
    
    cuMat C(3, 3);
    std::cout << "Test of add/subtract/multiply/division with cuMat and float" << std::endl;
    std::cout << "cuMat + cuMat" << std::endl; C = A + B; std::cout << C;    //OK OK
    std::cout << "cuMat - cuMat" << std::endl; C = A - B; std::cout << C;    //OK OK
    std::cout << "cuMat * cuMat" << std::endl; C = A * B; std::cout << C;    //OK OK
    std::cout << "cuMat / cuMat" << std::endl; C = A / B; std::cout << C;    //OK OK

    std::cout << "cuMat + float" << std::endl; C = A + 2; std::cout << C;    //OK OK
    std::cout << "cuMat + (-float)" << std::endl; C = A + (-2); std::cout << C; //OK OK
    std::cout << "float + cuMat" << std::endl; C = 2 + A; std::cout << C;    //OK OK
    std::cout << "cuMat * float" << std::endl; C = A * 2; std::cout << C;    //OK OK
    std::cout << "float * cuMat" << std::endl; C = 2 * A; std::cout << C;    //OK OK
    std::cout << "cuMat / float" << std::endl; C = A / 2; std::cout << C;    //OK OK
    std::cout << "float / cuMat" << std::endl; C = 20 / A; std::cout << C;   //OK OK

    std::cout << "Test of matrix-production" << std::endl;
    C = A.dot(B); std::cout << C; //OK OK
    return 0;
}
