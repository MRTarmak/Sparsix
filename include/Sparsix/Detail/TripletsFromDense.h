#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Sparsix/Core/Triplet.h>

namespace detail {
    /**
     * @brief Converts a rectangular dense matrix to non-zero coordinate triplets.
     * @param threshold Values whose absolute value is not greater than this are omitted.
     * @throws std::invalid_argument If dense rows have unequal lengths.
     */
    template <typename T>
    std::vector<Triplet<T>> triplets_from_dense(const std::vector<std::vector<T>> &matrix, T threshold) {
        size_t rows_count = matrix.size();
        size_t cols_count = rows_count > 0 ? matrix.front().size() : 0;

        std::vector<Triplet<T>> triplets;

        for (size_t i = 0; i < rows_count; i++) {
            const size_t current_row_size = matrix[i].size();
            if (current_row_size != cols_count) {
                throw std::invalid_argument("Dense matrix must be rectangular.");
            }

            for (size_t j = 0; j < current_row_size; j++) {
                if (std::abs(matrix[i][j]) > std::abs(threshold)) {
                    triplets.emplace_back(i, j, matrix[i][j]);
                }
            }
        }

        return triplets;
    }
}
