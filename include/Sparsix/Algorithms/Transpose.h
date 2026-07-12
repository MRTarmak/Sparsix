#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSC.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/Conversions.h>

namespace sparsix {
    template <typename T>
    MatrixCSC<T> transpose(const MatrixCSR<T> &A) {
        const size_t rows_count = A.rows_count();
        const size_t cols_count = A.cols_count();
        const size_t nnz = A.non_zero_count();

        std::vector<size_t> row_indices(nnz);
        std::vector<size_t> col_ptr(cols_count + 1, 0);
        std::vector<T> values(nnz);

        const auto &src_col_indices = A.col_indices();
        const auto &src_row_ptr = A.row_ptr();
        const auto &src_values = A.values();

        for (size_t col : src_col_indices)
            col_ptr[col + 1]++;

        for (size_t i = 1; i <= cols_count; ++i)
            col_ptr[i] += col_ptr[i - 1];

        std::vector<size_t> next = col_ptr;

        for (size_t row = 0; row < rows_count; row++) {
            for (size_t i = src_row_ptr[row]; i < src_row_ptr[row + 1]; i++) {
                size_t col = src_col_indices[i];
                size_t dst = next[col]++;

                row_indices[dst] = row;
                values[dst] = src_values[i];
            }
        }

        return MatrixCSC<T>(
            cols_count,
            rows_count,
            std::move(row_indices),
            std::move(col_ptr),
            std::move(values)
        );
    }

    template <typename T>
    MatrixCSR<T> transpose(const MatrixCSC<T>& A) {
        const size_t rows_count = A.rows_count();
        const size_t cols_count = A.cols_count();
        const size_t nnz = A.non_zero_count();

        std::vector<size_t> col_indices(nnz);
        std::vector<size_t> row_ptr(rows_count + 1, 0);
        std::vector<T> values(nnz);

        const auto &src_row_indices = A.row_indices();
        const auto &src_col_ptr = A.col_ptr();
        const auto &src_values = A.values();

        for (size_t row : src_row_indices)
            row_ptr[row + 1]++;

        for (size_t i = 1; i <= rows_count; ++i)
            row_ptr[i] += row_ptr[i - 1];

        std::vector<size_t> next = row_ptr;

        for (size_t col = 0; col < cols_count; col++) {
            for (size_t i = src_col_ptr[col]; i < src_col_ptr[col + 1]; i++) {
                size_t row = src_row_indices[i];
                size_t dst = next[row]++;

                col_indices[dst] = col;
                values[dst] = src_values[i];
            }
        }

        return MatrixCSR<T>(
            rows_count,
            cols_count,
            std::move(col_indices),
            std::move(row_ptr),
            std::move(values)
        );
    }

    template <typename T>
    MatrixCOO<T> transpose(const MatrixCOO<T>& A) {
        std::vector<Triplet<T>> triplets;
        triplets.reserve(A.non_zero_count());

        const auto &rows = A.rows();
        const auto &cols = A.cols();
        const auto &values = A.values();

        for (size_t i = 0; i < A.non_zero_count(); i++) {
            triplets.emplace_back(
                cols[i],
                rows[i],
                values[i]
            );
        }

        return MatrixCOO<T>(
            A.cols_count(),
            A.rows_count(),
            std::move(triplets)
        );
    }
}