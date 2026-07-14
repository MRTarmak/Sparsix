#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Sparsix/MatrixCSR.h>

namespace detail {
    /**
     * @brief Merges two equal-sized CSR matrices row by row.
     * @param on_equal Produces an optional value for coordinates present in both matrices.
     * @param on_left Produces an optional value for a left-only coordinate.
     * @param on_right Produces an optional value for a right-only coordinate.
     * @return CSR matrix containing callback-produced values.
     */
    template <typename T>
    MatrixCSR<T> binary_merge(const MatrixCSR<T> &A, const MatrixCSR<T> &B, 
                              auto on_equal, auto on_left, auto on_right) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimensions must match.");
        }

        size_t rows_count = A.rows_count();
        size_t cols_count = A.cols_count();

        std::vector<size_t> col_indices;
        std::vector<size_t> row_ptr;
        std::vector<T> values;

        col_indices.reserve(A.non_zero_count() + B.non_zero_count());
        values.reserve(A.non_zero_count() + B.non_zero_count());
        row_ptr.assign(rows_count + 1, 0);

        const auto &a_col_indices = A.col_indices();
        const auto &a_row_ptr = A.row_ptr();
        const auto &a_values = A.values();

        const auto &b_col_indices = B.col_indices();
        const auto &b_row_ptr = B.row_ptr();
        const auto &b_values = B.values();

        auto append =
        [&](size_t row, size_t col, const T &value) {
            col_indices.push_back(col);
            values.push_back(value);
            row_ptr[row + 1]++;
        };

        for (size_t row = 0; row < rows_count; row++) {
            size_t a_i = a_row_ptr[row];
            size_t b_i = b_row_ptr[row];
            size_t a_end = a_row_ptr[row + 1];
            size_t b_end = b_row_ptr[row + 1];

            while (a_i < a_end && b_i < b_end) {
                if (a_col_indices[a_i] == b_col_indices[b_i]) {
                    auto result = on_equal(a_values[a_i], b_values[b_i]);
                    if (result)
                        append(row, a_col_indices[a_i], *result);
                    a_i++;
                    b_i++;
                } else if (a_col_indices[a_i] < b_col_indices[b_i]) {
                    auto result = on_left(a_values[a_i]);
                    if (result)
                        append(row, a_col_indices[a_i], *result);
                    a_i++;
                } else {
                    auto result = on_right(b_values[b_i]);
                    if (result)
                        append(row, b_col_indices[b_i], *result);
                    b_i++;
                }
            }

            while (a_i < a_end) {
                auto result = on_left(a_values[a_i]);
                if (result)
                    append(row, a_col_indices[a_i], *result);
                a_i++;
            }
            while (b_i < b_end) {
                auto result = on_right(b_values[b_i]);
                if (result)
                    append(row, b_col_indices[b_i], *result);
                b_i++;
            }
        }

        for (size_t i = 1; i <= rows_count; i++)
            row_ptr[i] += row_ptr[i - 1];

        return MatrixCSR<T>(rows_count, cols_count, 
                         std::move(col_indices), 
                         std::move(row_ptr), 
                         std::move(values));
    }
}
