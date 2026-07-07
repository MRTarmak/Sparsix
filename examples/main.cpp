#include <iostream>

#include <Sparsix.h>

int main() {
    MatrixCOO<int> int_matrix = MatrixCOO<int>::create_random(20, 10, 0.2, 50, 300);
    std::cout << "Integer matrix created successfully" << std::endl;

    std::cout << "Integer matrix: " << std::endl;
    for (size_t i = 0; i < int_matrix.rows_count(); i++) {
        for (size_t j = 0; j < int_matrix.cols_count(); j++) {
            std::cout << int_matrix(i, j) << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "Is sorted: " << int_matrix.sorted() << std::endl;

    std::cout << "Sorting..." << std::endl;
    int_matrix.sort();
    std::cout << "Is sorted: " << int_matrix.sorted() << std::endl;

    MatrixCOO<double> identity = MatrixCOO<double>::create_identity(10);
    std::cout << "Identity matrix created successfully" << std::endl;

    std::cout << "Identity matrix: " << std::endl;
    for (size_t i = 0; i < identity.rows_count(); i++) {
        for (size_t j = 0; j < identity.cols_count(); j++) {
            std::cout << identity(i, j) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}