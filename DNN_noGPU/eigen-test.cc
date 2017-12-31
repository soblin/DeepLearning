#include <iostream>
#include <eigen3/Eigen/Dense>

int main(){
    Eigen::MatrixXf A(2, 2);
    A << 1, 2,
         3, 4;
    Eigen::MatrixXf B(2, 2);
    B << 4, 3,
         2, 1;

    //A@B
    std::cout << A*B << std::endl << std::endl;;

    //A*B elementwise
    std::cout << A.array()*B.array() << std::endl << std::endl;

    //each element is doubled
    Eigen::MatrixXf C = A;
    C *= 2;
    std::cout << C << std::endl << std::endl;

    //each element is +=10
    C = A;
    C.array() += 10;
    std::cout << C << std::endl << std::endl;

    //C = A@B
    C = A;
    C *= A;
    std::cout << C << std::endl << std::endl;

    //C = A*B
    C = A;
    C.array() *= B.array();
    std::cout << C << std::endl << std::endl;

    //C += A*B
    C = Eigen::MatrixXf::Ones(2, 2);
    std::cout << C << std::endl << std::endl;
    C.array() += A.array() * B.array();
    std::cout << C << std::endl << std::endl;
    return 0;
}
