#include <iostream>
#include <vector>

//If you include Dense, Core, Geometry, LU, Cholesky, SVD, QR, Eigenvalues are alos included. But it takes time to compile
#include <eigen3/Eigen/Dense>

//Matrixの掛け算は行列積、Arrayの掛け算はelementwise

int main(){
    //初期化、値の代入
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    Eigen::MatrixXf B(3, 3);
    B << 11, 12, 13, 14, 15, 16, 17, 18, 19;
    std::cout << "A:\n" << A << std::endl;
    std::cout << "B:\n" << B << std::endl;

    //operator+, -, *, +=, -=, *= with Matirx
    Eigen::MatrixXf C = A + B;
    std::cout << "C:\n" << C << std::endl;
    C = A * B;
    std::cout << "C:\n" << C << std::endl;
    A += B;
    std::cout << "A:\n" << A << std::endl;
    A *= B;
    std::cout << "A:\n" << A << std::endl;

    //initialize with constants
    A = Eigen::MatrixXf::Zero(3, 3);
    std::cout << "A:\n" << A << std::endl;

    A = Eigen::MatrixXf::Ones(4, 3);
    std::cout << "A:\n" << A << std::endl;

    A = Eigen::MatrixXf::Random(3, 3);
    std::cout << "A:\n" << A << std::endl;

    //accesser
    for(int i=0; i<A.rows(); i++){
        for(int j=0 ;j <A.cols(); j++){
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }

    //transepose
    A.transposeInPlace();
    std::cout << "A:\n" << A << std::endl;

    Eigen::MatrixXf D = Eigen::MatrixXf::Constant(5, 5, 0.999);
    std::cout << "D:\n" << D << std::endl;

    //operator= copies all elements(deep copy)
    Eigen::MatrixXf E = B;
    B.adjointInPlace();
    std::cout << "B:\n" << B << std::endl;
    std::cout << "E:\n" << E << std::endl;

    Eigen::MatrixXf *pMatrix = &E;
    std::cout << "*pMatrix:\n" << *pMatrix << std::endl;
    E.transposeInPlace();
    std::cout << "*pMatrix:\n" << *pMatrix << std::endl;

    //deep copy
    Eigen::MatrixXf F = E.transpose();
    std::cout << "F:\n" << F << std::endl;

    E.transpose();
    std::cout << "F:\n" << F << std::endl;
    return 0;
}
