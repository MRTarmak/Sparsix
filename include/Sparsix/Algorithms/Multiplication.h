#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSC.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/Conversions.h>

namespace sparsix {
    template <typename T>
    std::vector<T> multiply(const MatrixCSR<T> &A, const std::vector<T> &vector) {
        if (A.cols_count() != vector.size())
            throw std::invalid_argument("The number of rows in matrix and the size of vector must match.");

        size_t rows_count = A.rows_count();

        const auto &col_indices = A.col_indices();
        const auto &row_ptr = A.row_ptr();
        const auto &values = A.values();

        std::vector<T> result(A.rows_count(), T{});

        for (size_t row = 0; row < rows_count; row++) {
            size_t end = row_ptr[row + 1];
            for (size_t i = row_ptr[row]; i < end; i++) {
                result[row] += values[i] * vector[col_indices[i]];
            }
        }

        return result;
    }

    template <SparseMatrix Matrix>
    auto multiply(const Matrix &A, const std::vector<typename Matrix::value_type> &vector) {
        if (A.cols_count() != vector.size())
            throw std::invalid_argument("The number of rows in matrix and the size of vector must match.");

        return multiply(toCSR(A), vector);
    }

    template <typename T>
    MatrixCSR<T> multiply(const MatrixCSR<T> &A, const MatrixCSC<T> &B) {
        if (A.cols_count() != B.rows_count())
            throw std::invalid_argument("The number of columns in A and the number of rows in B must match.");

        size_t rows_count = A.rows_count();
        size_t cols_count = B.cols_count();

        std::vector<size_t> col_indices;
        std::vector<size_t> row_ptr;
        std::vector<T> values;

        row_ptr.assign(rows_count + 1, 0);

        const auto &a_col_indices = A.col_indices();
        const auto &a_row_ptr = A.row_ptr();
        const auto &a_values = A.values();

        const auto &b_row_indices = B.row_indices();
        const auto &b_col_ptr = B.col_ptr();
        const auto &b_values = B.values();

        for (size_t row = 0; row < rows_count; row++) {
            size_t row_end = a_row_ptr[row + 1];
            for (size_t col = 0; col < cols_count; col++) {
                size_t a_i = a_row_ptr[row];
                size_t b_i = b_col_ptr[col];
                size_t col_end = b_col_ptr[col + 1];
                T sum{};
                while (a_i < row_end && b_i < col_end) {
                    if (a_col_indices[a_i] == b_row_indices[b_i]) {
                        sum += a_values[a_i] * b_values[b_i];
                        a_i++;
                        b_i++;
                    } else if (a_col_indices[a_i] < b_row_indices[b_i]) {
                        a_i++;
                    } else {
                        b_i++;
                    }
                }

                if (sum != T{}) {
                    col_indices.push_back(col);
                    values.push_back(sum);
                    row_ptr[row + 1]++;
                }
            }
        }

        for (size_t i = 1; i <= rows_count; i++)
            row_ptr[i] += row_ptr[i - 1];

        return MatrixCSR<T>(rows_count, cols_count, 
                         std::move(col_indices), 
                         std::move(row_ptr), 
                         std::move(values));
    }

    template <SparseMatrix MatrixA, SparseMatrix MatrixB>
    auto multiply(const MatrixA &A, const MatrixB &B) {
        if (A.cols_count() != B.rows_count())
            throw std::invalid_argument("The number of columns in A and the number of rows in B must match.");

        return multiply(toCSR(A), toCSC(B));
    }
}

template <SparseMatrix Matrix>
auto operator*(const Matrix &A, const std::vector<typename Matrix::value_type> &vector) {
    return sparsix::multiply(A, vector);
}

template <SparseMatrix MatrixA, SparseMatrix MatrixB>
auto operator*(const MatrixA &A, const MatrixB &B) {
    return sparsix::multiply(A, B);
}
