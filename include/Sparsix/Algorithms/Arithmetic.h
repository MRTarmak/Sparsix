#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSC.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/BinaryMerge.h>
#include <Sparsix/Detail/Conversions.h>

namespace sparsix {
    template <typename T>
    MatrixCSR<T> add(const MatrixCSR<T> &A, const MatrixCSR<T> &B) {
        return detail::binary_merge(A, B, 
        [](auto a_value, auto b_value) -> std::optional<T> {
            auto sum = a_value + b_value;
            if (sum != T{})
                return std::optional{sum};
            return std::nullopt;
        },
        [](auto a_value) -> std::optional<T> {
            return std::optional{a_value};
        },
        [](auto b_value) -> std::optional<T> {
            return std::optional{b_value};
        });
    }

    template <SparseMatrix MatrixA, SparseMatrix MatrixB>
    auto add(const MatrixA &A, const MatrixB &B) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimensions must match.");
        }

        return add(toCSR(A), toCSR(B));
    }

    template <typename T>
    MatrixCSR<T> subtract(const MatrixCSR<T> &A, const MatrixCSR<T> &B) {
        return detail::binary_merge(A, B, 
        [](auto a_value, auto b_value) -> std::optional<T> {
            auto sum = a_value - b_value;
            if (sum != T{})
                return std::optional{sum};
            return std::nullopt;
        },
        [](auto a_value) -> std::optional<T> {
            return std::optional{a_value};
        },
        [](auto b_value) -> std::optional<T> {
            return std::optional{-b_value};
        });
    }

    template <SparseMatrix MatrixA, SparseMatrix MatrixB>
    auto subtract(const MatrixA &A, const MatrixB &B) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimensions must match.");
        }

        return subtract(toCSR(A), toCSR(B));
    }

    template <typename T>
    MatrixCSR<T> multiply(const MatrixCSR<T> &A, const T &x) {
        std::vector<T> values;
        values.reserve(A.non_zero_count());

        for (const auto &v : A.values())
            values.push_back(v * x);

        return MatrixCSR(A.rows_count(), A.cols_count(), A.col_indices(), A.row_ptr(), std::move(values));
    }

    template <SparseMatrix Matrix>
    auto multiply(const Matrix &A, const typename Matrix::value_type &x) {
        return multiply(toCSR(A), x);
    }

    template <SparseMatrix Matrix>
    auto multiply(const typename Matrix::value_type &x, const Matrix &A) {
        return multiply(toCSR(A), x);
    }

    template <typename T>
    MatrixCSR<T> divide(const MatrixCSR<T> &A, const T &x) {
        if (x == T{})
            throw std::runtime_error("Division by zero.");

        std::vector<T> values;
        values.reserve(A.non_zero_count());

        for (const auto &v : A.values())
            values.push_back(v / x);

        return MatrixCSR(A.rows_count(), A.cols_count(), A.col_indices(), A.row_ptr(), std::move(values));
    }

    template <SparseMatrix Matrix>
    auto divide(const Matrix &A, const typename Matrix::value_type &x) {
        if (x == typename Matrix::value_type{})
            throw std::runtime_error("Division by zero.");

        return divide(toCSR(A), x);
    }

    template <typename T>
    MatrixCSR<T> cwise_multiply(const MatrixCSR<T> &A, const MatrixCSR<T> &B) {
        return detail::binary_merge(A, B, 
        [](auto a_value, auto b_value) -> std::optional<T> {
            return std::optional{a_value * b_value};
        },
        [](auto a_value) -> std::optional<T> {
            return std::nullopt; 
        },
        [](auto b_value) -> std::optional<T> {
            return std::nullopt;
        });
    }

    template <SparseMatrix MatrixA, SparseMatrix MatrixB>
    auto cwise_multiply(const MatrixA &A, const MatrixB &B) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimensions must match.");
        }

        return cwise_multiply(toCSR(A), toCSR(B));
    }

    template <typename T>
    MatrixCSR<T> cwise_divide(const MatrixCSR<T> &A, const MatrixCSR<T> &B) {
        return detail::binary_merge(A, B, 
        [](auto a_value, auto b_value) -> std::optional<T> {
            if (b_value == T{})
                throw std::runtime_error("Division by zero.");
            return std::optional{a_value / b_value};
        },
        [](auto a_value) -> std::optional<T> {
            throw std::runtime_error("Division by zero.");
        },
        [](auto b_value) -> std::optional<T> {
            return std::nullopt;
        });
    }

    template <SparseMatrix MatrixA, SparseMatrix MatrixB>
    auto cwise_divide(const MatrixA &A, const MatrixB &B) {
        if (A.rows_count() != B.rows_count() ||
            A.cols_count() != B.cols_count()) 
        {
            throw std::invalid_argument("Matrices dimensions must match.");
        }

        return cwise_divide(toCSR(A), toCSR(B));
    }
}

template <SparseMatrix MatrixA, SparseMatrix MatrixB>
auto operator+(const MatrixA &A, const MatrixB &B) {
    return sparsix::add(A, B);
}

template <SparseMatrix MatrixA, SparseMatrix MatrixB>
auto operator-(const MatrixA &A, const MatrixB &B) {
    return sparsix::subtract(A, B);
}

template <SparseMatrix Matrix>
auto operator*(const Matrix &A, const typename Matrix::value_type &x) {
    return sparsix::multiply(A, x);
}

template <SparseMatrix Matrix>
auto operator*(const typename Matrix::value_type &x, const Matrix &A) {
    return sparsix::multiply(x, A);
}

template <SparseMatrix Matrix>
auto operator/(const Matrix &A, const typename Matrix::value_type &x) {
    return sparsix::divide(A, x);
}
