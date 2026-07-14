#pragma once

#include <algorithm>
#include <cstddef>
#include <ostream>

#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/Conversions.h>

/**
 * @brief Writes a concise diagnostic representation of a sparse matrix.
 * @details The output contains dimensions, number of stored entries and up to five
 * `(row, column)=value` triplets.
 */
template <SparseMatrix Matrix>
std::ostream &operator<<(std::ostream &output, const Matrix &matrix) {
    const auto &coo = toCOO(matrix);
    output << "SparseMatrix(" << coo.rows_count() << "x" << coo.cols_count() << ", nnz=" << coo.non_zero_count() << ")";
    const size_t preview_size = std::min<size_t>(coo.non_zero_count(), 5);
    if (preview_size == 0)
        return output;

    output << " [";
    for (size_t i = 0; i < preview_size; ++i) {
        if (i != 0)
            output << ", ";
        output << "(" << coo.rows()[i] << ", " << coo.cols()[i] << ")=" << coo.values()[i];
    }
    if (coo.non_zero_count() > preview_size)
        output << ", ...";
    return output << "]";
}
