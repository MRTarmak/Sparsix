#include <Sparsix.h>
#include <complex>
#include <iostream>

using Complex = std::complex<double>;

int main() {
    MatrixCOO<Complex> complex_matrix = MatrixCOO<Complex>::create_random(10, 20, 0.5, {5, 6.8}, {6, 8});
    const MatrixCOO<int> int_matrix = MatrixCOO<int>::create_random(20, 10, 0.2, 50, 300);

    std::cout << "Matrices created successfully!" << std::endl;
    std::cout << "Integer matrix: " << std::endl;

    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 10; j++) {
            std::cout << int_matrix(i, j) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}