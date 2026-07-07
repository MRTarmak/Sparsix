#include <iostream>

#include <Sparsix.h>

int main() {
    MatrixCOO<int> int_matrix = MatrixCOO<int>::create_random(20, 10, 0.2, 50, 300);

    std::cout << "Matrix created successfully!" << std::endl;

    std::cout << "Integer matrix: " << std::endl;
    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 10; j++) {
            std::cout << int_matrix(i, j) << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "Is sorted: " << int_matrix.sorted() << std::endl;

    std::cout << "Sorting..." << std::endl;
    int_matrix.sort();
    std::cout << "Is sorted: " << int_matrix.sorted() << std::endl;

    return 0;
}