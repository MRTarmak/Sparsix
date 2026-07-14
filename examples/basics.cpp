#include <iostream>
#include <vector>

#include <Sparsix.h>

int main() {
    MatrixCOO<double> coo(3, 3, {
        {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0}, {2, 2, 2.0}
    });

    const auto csr = toCSR(coo);
    const auto csc = toCSC(csr);
    const auto transposed = sparsix::transpose(csr);
    const auto doubled = 2.0 * csr;
    const std::vector<double> vector{1.0, 2.0, 3.0};

    std::cout << "COO: " << coo << '\n';
    std::cout << "CSR: " << csr << '\n';
    std::cout << "CSC: " << csc << '\n';
    std::cout << "transpose(CSR): " << transposed << '\n';
    std::cout << "2 * CSR: " << doubled << '\n';
    std::cout << "CSR * [1, 2, 3] =";
    for (const double value : csr * vector)
        std::cout << ' ' << value;
    std::cout << "\nFrobenius norm: " << sparsix::frobenius_norm(csr) << '\n';
}
