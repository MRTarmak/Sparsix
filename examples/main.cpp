#include <Sparsix.h>
#include <iostream>

using Entry = MatrixCOO<int>::Entry;

int main() {
    // std::vector<Entry> emptyEntries;
    // MatrixCOO<int> matrix{1, 1, emptyEntries};

    // std::vector<Entry> duplicateEntries{{0, 0, 1}, {0, 0, 2}};
    // MatrixCOO<int> matrix{1, 2, duplicateEntries};

    std::vector<Entry> entries{{0, 0, 1}, {1, 2, 4}, {1, 1, 3}};
    MatrixCOO<int> matrix{2, 3, entries};

    std::cout << "Matrix created successfully!" << std::endl;

    return 0;
}