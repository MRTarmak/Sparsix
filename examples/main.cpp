#include <Sparsix.h>
#include <complex>
#include <iostream>

using Value = std::complex<double>;
using Entry = MatrixCOO<Value>::Entry;

int main() {
    MatrixCOO<Value> complex_matrix = MatrixCOO<Value>::create_random(10, 20, 0.5, {5, 6.8}, {6, 8});
    MatrixCOO<int> int_matrix = MatrixCOO<int>::create_random(20, 10, 0.2, 50, 300);

    std::cout << "Matrices created successfully!" << std::endl;

    return 0;
}