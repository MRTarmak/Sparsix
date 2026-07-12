#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSC.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/Conversions.h>

namespace sparsix {
    template <typename T>
    MatrixCSR<T> add(const MatrixCSR<T> &A, const MatrixCSR<T> &B) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimentions bla bla"); // TODO write exception message
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

        for (size_t row = 0; row < rows_count; row++) {
            size_t a_i = a_row_ptr[row];
            size_t b_i = b_row_ptr[row];
            size_t a_end = a_row_ptr[row + 1];
            size_t b_end = b_row_ptr[row + 1];

            while (a_i < a_end && b_i < b_end) {
                if (a_col_indices[a_i] == b_col_indices[b_i]) {
                    auto sum = a_values[a_i] + b_values[b_i];
                    if (sum != T{}) {
                        col_indices.push_back(a_col_indices[a_i]);
                        values.push_back(sum);
                        row_ptr[row + 1]++;
                    }
                    a_i++;
                    b_i++;
                } else if (a_col_indices[a_i] < b_col_indices[b_i]) {
                    col_indices.push_back(a_col_indices[a_i]);
                    values.push_back(a_values[a_i]);
                    row_ptr[row + 1]++;
                    a_i++;
                } else {
                    col_indices.push_back(b_col_indices[b_i]);
                    values.push_back(b_values[b_i]);
                    row_ptr[row + 1]++;
                    b_i++;
                }
            }

            while (a_i < a_end) {
                col_indices.push_back(a_col_indices[a_i]);
                values.push_back(a_values[a_i]);
                row_ptr[row + 1]++;
                a_i++;
            }
            while (b_i < b_end) {
                col_indices.push_back(b_col_indices[b_i]);
                values.push_back(b_values[b_i]);
                row_ptr[row + 1]++;
                b_i++;
            }
        }

        for (size_t i = 1; i <= rows_count; i++)
            row_ptr[i] += row_ptr[i - 1];

        return MatrixCSR(rows_count, cols_count, 
                         std::move(col_indices), 
                         std::move(row_ptr), 
                         std::move(values));
    }

    template <SparseMatrix MatrixA, SparseMatrix MatrixB>
    auto add(const MatrixA &A, const MatrixB &B) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimentions bla bla"); // TODO write exception message
        }

        return add(toCSR(A), toCSR(B));
    }

    // A += B

    // A - B, A -= B

    // A * scalar, scalar * A, A *= scalar

    // A / scalar, A /= scalar

    // cwiseMultiply(A, B)

    // cwiseDivide(A, B)
}

template <SparseMatrix MatrixA, SparseMatrix MatrixB>
auto operator+(const MatrixA &A, const MatrixB &B) {
    return sparsix::add(A, B);
}
