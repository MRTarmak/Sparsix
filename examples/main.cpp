#include <Sparsix.h>
#include <complex>
#include <iostream>

using Value = std::complex<double>;
using Entry = MatrixCOO<Value>::Entry;

int main() {
    std::vector<Entry> entries{{0, 0, {1.0, 2.0}}, {1, 2, {4.0, -1.0}}, {1, 1, {3.0, 0.5}}};
    MatrixCOO<Value> matrix{2, 3, entries};

    std::cout << "Matrix created successfully!" << std::endl;

    return 0;
}