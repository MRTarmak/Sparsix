#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Sparsix/MatrixCSR.h>
#include <Sparsix/MatrixCSC.h>
#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Conversions.h>

namespace sparsix {

/** @brief Checks whether a CSR matrix is square and symmetric. */
template <typename T>
bool is_symmetric(const MatrixCSR<T> &A) {
    size_t rows_count = A.rows_count();
    size_t cols_count = A.cols_count();
    if (rows_count != cols_count)
        return false;
    const auto &col_indices = A.col_indices();
    const auto &row_ptr = A.row_ptr();
    const auto &values = A.values();
    for (size_t row = 0; row < rows_count; ++row) {
        size_t row_end = row_ptr[row + 1];
        for (size_t i = row_ptr[row]; i < row_end; ++i)
            if (values[i] != A(col_indices[i], row))
                return false;
    }
    return true;
}
/** @brief Checks symmetry after converting a supported matrix format to CSR. */
template <SparseMatrix Matrix>
bool is_symmetric(const Matrix &A) {
    return is_symmetric(toCSR(A));
}
/** @brief Returns the sum of diagonal elements of a sparse matrix. */
template <SparseMatrix Matrix>
typename Matrix::value_type trace(const Matrix &A) {
    using T = typename Matrix::value_type;
    const size_t diagonal_size = std::min(A.rows_count(), A.cols_count());
    T sum = T{};
    for (size_t i = 0; i < diagonal_size; ++i) {
        sum += A(i, i);
    }
    return sum;
}
/** @brief Returns the Frobenius norm of any supported matrix format. */
template <SparseMatrix Matrix>
real_type<typename Matrix::value_type> frobenius_norm(const Matrix &A) {
    using T = typename Matrix::value_type;
    const auto &values = A.values();
    real_type<T> sum = real_type<T>{};
    for (auto v : values) {
        sum += std::norm(v);
    }
    return std::sqrt(sum);
}
/** @brief Returns the maximum absolute column sum of a CSC matrix. */
template <typename T>
real_type<T> one_norm(const MatrixCSC<T> &A) {
    size_t cols_count = A.cols_count();
    const auto &col_ptr = A.col_ptr();
    const auto &values = A.values();
    real_type<T> max_sum = real_type<T>{};
    for (size_t col = 0; col < cols_count; ++col) {
        size_t col_end = col_ptr[col + 1];
        real_type<T> sum = real_type<T>{};
        for (size_t i = col_ptr[col]; i < col_end; ++i) {
            sum += std::abs(values[i]);
        }
        max_sum = sum > max_sum ? sum : max_sum;
    }
    return max_sum;
}
/** @brief Returns the one norm of any supported matrix format. */
template <SparseMatrix Matrix>
real_type<typename Matrix::value_type> one_norm(const Matrix &A) {
    return one_norm(toCSC(A));
}
/** @brief Returns the maximum absolute row sum of a CSR matrix. */
template <typename T>
real_type<T> infinity_norm(const MatrixCSR<T> &A) {
    size_t rows_count = A.rows_count();
    const auto &row_ptr = A.row_ptr();
    const auto &values = A.values();
    real_type<T> max_sum = real_type<T>{};
    for (size_t row = 0; row < rows_count; ++row) {
        size_t row_end = row_ptr[row + 1];
        real_type<T> sum = real_type<T>{};
        for (size_t i = row_ptr[row]; i < row_end; ++i) {
            sum += std::abs(values[i]);
        }
        max_sum = sum > max_sum ? sum : max_sum;
    }
    return max_sum;
}
/** @brief Returns the infinity norm of any supported matrix format. */
template <SparseMatrix Matrix>
real_type<typename Matrix::value_type> infinity_norm(const Matrix &A) {
    return infinity_norm(toCSR(A));
}

} // namespace sparsix
