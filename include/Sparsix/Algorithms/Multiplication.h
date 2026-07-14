#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSC.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/Conversions.h>
#include <Sparsix/Concepts/SparseMatrix.h>

namespace sparsix {

/** @brief Multiplies a CSR matrix by a dense vector. */
template <typename T>
std::vector<T> multiply(const MatrixCSR<T> &A, const std::vector<T> &vector) {
    if (A.cols_count() != vector.size())
        throw std::invalid_argument("The number of rows in matrix and the size of vector must match.");
    size_t rows_count = A.rows_count();
    const auto &col_indices = A.col_indices();
    const auto &row_ptr = A.row_ptr();
    const auto &values = A.values();
    std::vector<T> result(A.rows_count(), T{});
    for (size_t row = 0; row < rows_count; ++row) {
        size_t end = row_ptr[row + 1];
        for (size_t i = row_ptr[row]; i < end; ++i) {
            result[row] += values[i] * vector[col_indices[i]];
        }
    }
    return result;
}

/** @brief Multiplies any supported sparse matrix by a dense vector. */
template <SparseMatrix Matrix>
auto multiply(const Matrix &A, const std::vector<typename Matrix::value_type> &vector) {
    if (A.cols_count() != vector.size())
        throw std::invalid_argument("The number of rows in matrix and the size of vector must match.");
    return multiply(toCSR(A), vector);
}

/**
 * @brief Legacy CSR-by-CSC inner-product multiplication.
 * @warning This is the previous, slower implementation. It is retained only
 * for comparison and is not used by the public multiplication API.
 */
template <typename T>
MatrixCSR<T> naive_multiply(const MatrixCSR<T> &A, const MatrixCSC<T> &B) {
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

    for (size_t row = 0; row < rows_count; ++row) {
        size_t row_end = a_row_ptr[row + 1];
        for (size_t col = 0; col < cols_count; ++col) {
            size_t a_i = a_row_ptr[row];
            size_t b_i = b_col_ptr[col];
            size_t col_end = b_col_ptr[col + 1];
            T sum{};
            while (a_i < row_end && b_i < col_end) {
                if (a_col_indices[a_i] == b_row_indices[b_i]) {
                    sum += a_values[a_i] * b_values[b_i];
                    ++a_i;
                    ++b_i;
                } else if (a_col_indices[a_i] < b_row_indices[b_i]) {
                    ++a_i;
                } else {
                    ++b_i;
                }
            }
            if (sum != T{}) {
                col_indices.push_back(col);
                values.push_back(sum);
                ++row_ptr[row + 1];
            }
        }
    }

    for (size_t i = 1; i <= rows_count; ++i)
        row_ptr[i] += row_ptr[i - 1];

    return MatrixCSR<T>(rows_count, cols_count,
                     std::move(col_indices),
                     std::move(row_ptr),
                     std::move(values));
}

/** @brief Multiplies two CSR matrices with Gustavson's row-wise sparse algorithm. */
template <typename T>
MatrixCSR<T> multiply(const MatrixCSR<T> &A, const MatrixCSR<T> &B) {
    if (A.cols_count() != B.rows_count())
        throw std::invalid_argument("The number of columns in A and the number of rows in B must match.");

    size_t rows_count = A.rows_count();
    size_t cols_count = B.cols_count();

    std::vector<size_t> col_indices;
    std::vector<size_t> row_ptr(rows_count + 1, 0);
    std::vector<T> result_values;

    const auto &a_col_indices = A.col_indices();
    const auto &a_row_ptr = A.row_ptr();
    const auto &a_values = A.values();

    const auto &b_col_indices = B.col_indices();
    const auto &b_row_ptr = B.row_ptr();
    const auto &b_values = B.values();

    std::vector<T> accumulator(cols_count, T{});
    std::vector<size_t> marker(cols_count, rows_count);
    std::vector<size_t> touched_columns;

    for (size_t row = 0; row < rows_count; ++row) {
        touched_columns.clear();

        for (size_t a_i = a_row_ptr[row]; a_i < a_row_ptr[row + 1]; ++a_i) {
            const size_t shared_index = a_col_indices[a_i];
            const T &a_value = a_values[a_i];

            for (size_t b_i = b_row_ptr[shared_index]; b_i < b_row_ptr[shared_index + 1]; ++b_i) {
                const size_t col = b_col_indices[b_i];
                const T product = a_value * b_values[b_i];

                if (marker[col] != row) {
                    marker[col] = row;
                    accumulator[col] = product;
                    touched_columns.push_back(col);
                } else {
                    accumulator[col] += product;
                }
            }
        }

        std::sort(touched_columns.begin(), touched_columns.end());
        for (size_t col : touched_columns) {
            if (accumulator[col] != T{}) {
                col_indices.push_back(col);
                result_values.push_back(accumulator[col]);
            }
        }
        row_ptr[row + 1] = col_indices.size();
    }

    return MatrixCSR<T>(rows_count, cols_count,
                        std::move(col_indices),
                        std::move(row_ptr),
                        std::move(result_values));
}

/** @brief Multiplies any two supported sparse matrix formats. */
template <SparseMatrix MatrixA, SparseMatrix MatrixB>
auto multiply(const MatrixA &A, const MatrixB &B) {
    if (A.cols_count() != B.rows_count())
        throw std::invalid_argument("The number of columns in A and the number of rows in B must match.");
    
    return multiply(toCSR(A), toCSR(B));
}

/** @brief Operator form of sparse matrix-vector multiplication. */
template <SparseMatrix Matrix>
auto operator*(const Matrix &A, const std::vector<typename Matrix::value_type> &vector) {
    return sparsix::multiply(A, vector);
}

/** @brief Operator form of sparse matrix-matrix multiplication. */
template <SparseMatrix MatrixA, SparseMatrix MatrixB>
auto operator*(const MatrixA &A, const MatrixB &B) {
    return sparsix::multiply(A, B);
}

} // namespace sparsix
