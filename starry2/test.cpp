#include <stdlib.h>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

using Grad = Eigen::AutoDiffScalar<Eigen::Matrix<double, 50, 1>>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

int main() {

    Vector<Grad> y;
    Matrix<Grad> gA;
    Matrix<double> dA;
    Vector<Grad> res;

    for (int i = 0; i < 100000; i++)
        res = dA * y;

}
