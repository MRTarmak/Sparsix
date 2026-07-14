#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/MatrixCSC.h>

namespace sparsix {

/** @brief Converts CSR storage to an owning COO matrix. */
template <typename T>
MatrixCOO<T> toCOO(const MatrixCSR<T> &csr) {
    std::vector<Triplet<T>> triplets;
    triplets.reserve(csr.non_zero_count());
    for (size_t row = 0; row < csr.rows_count(); ++row)
        for (size_t i = csr.row_ptr()[row]; i < csr.row_ptr()[row + 1]; ++i)
            triplets.push_back({row, csr.col_indices()[i], csr.values()[i]});
    return MatrixCOO<T>(csr.rows_count(), csr.cols_count(), std::move(triplets));
}

/** @brief Converts CSC storage to an owning COO matrix. */
template <typename T>
MatrixCOO<T> toCOO(const MatrixCSC<T> &csc) {
    std::vector<Triplet<T>> triplets;
    triplets.reserve(csc.non_zero_count());
    for (size_t col = 0; col < csc.cols_count(); ++col)
        for (size_t i = csc.col_ptr()[col]; i < csc.col_ptr()[col + 1]; ++i)
            triplets.push_back({csc.row_indices()[i], col, csc.values()[i]});
    return MatrixCOO<T>(csc.rows_count(), csc.cols_count(), std::move(triplets));
}

/** @brief Converts COO storage to CSR without an intermediate matrix. */
template <typename T>
MatrixCSR<T> toCSR(const MatrixCOO<T> &coo) {
    if (!coo.sorted()) {
        MatrixCOO<T> ordered = coo;
        ordered.sort();
        return toCSR(ordered);
    }

    std::vector<size_t> row_ptr(coo.rows_count() + 1, 0);
    for (size_t row : coo.rows())
        ++row_ptr[row + 1];
    for (size_t row = 1; row < row_ptr.size(); ++row)
        row_ptr[row] += row_ptr[row - 1];

    return MatrixCSR<T>(coo.rows_count(), coo.cols_count(),
                        std::vector<size_t>(coo.cols().begin(), coo.cols().end()),
                        std::move(row_ptr),
                        std::vector<T>(coo.values().begin(), coo.values().end()));
}

/** @brief Converts CSC storage to CSR without an intermediate COO matrix. */
template <typename T>
MatrixCSR<T> toCSR(const MatrixCSC<T> &csc) {
    std::vector<size_t> col_indices(csc.non_zero_count());
    std::vector<size_t> row_ptr(csc.rows_count() + 1, 0);
    std::vector<T> values(csc.non_zero_count());

    for (size_t row : csc.row_indices())
        ++row_ptr[row + 1];
    for (size_t row = 1; row < row_ptr.size(); ++row)
        row_ptr[row] += row_ptr[row - 1];
    auto next = row_ptr;
    for (size_t col = 0; col < csc.cols_count(); ++col)
        for (size_t i = csc.col_ptr()[col]; i < csc.col_ptr()[col + 1]; ++i) {
            const size_t dst = next[csc.row_indices()[i]]++;
            col_indices[dst] = col;
            values[dst] = csc.values()[i];
        }

    return MatrixCSR<T>(csc.rows_count(), csc.cols_count(), std::move(col_indices),
                        std::move(row_ptr), std::move(values));
}

/** @brief Converts COO storage to CSC without an intermediate matrix. */
template <typename T>
MatrixCSC<T> toCSC(const MatrixCOO<T> &coo) {
    std::vector<size_t> order(coo.non_zero_count());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&coo](size_t lhs, size_t rhs) {
        if (coo.cols()[lhs] != coo.cols()[rhs])
            return coo.cols()[lhs] < coo.cols()[rhs];
        return coo.rows()[lhs] < coo.rows()[rhs];
    });

    std::vector<size_t> row_indices;
    std::vector<T> values;
    row_indices.reserve(order.size());
    values.reserve(order.size());
    std::vector<size_t> col_ptr(coo.cols_count() + 1, 0);
    for (size_t index : order)
        col_ptr[coo.cols()[index] + 1]++;
    for (size_t index : order) {
        row_indices.push_back(coo.rows()[index]);
        values.push_back(coo.values()[index]);
    }
    for (size_t col = 1; col < col_ptr.size(); ++col)
        col_ptr[col] += col_ptr[col - 1];

    return MatrixCSC<T>(coo.rows_count(), coo.cols_count(),
                        std::move(row_indices),
                        std::move(col_ptr),
                        std::move(values));
}

/** @brief Converts CSR storage to CSC without an intermediate COO matrix. */
template <typename T>
MatrixCSC<T> toCSC(const MatrixCSR<T> &csr) {
    std::vector<size_t> row_indices(csr.non_zero_count());
    std::vector<size_t> col_ptr(csr.cols_count() + 1, 0);
    std::vector<T> values(csr.non_zero_count());

    for (size_t col : csr.col_indices())
        ++col_ptr[col + 1];
    for (size_t col = 1; col < col_ptr.size(); ++col)
        col_ptr[col] += col_ptr[col - 1];
    auto next = col_ptr;
    for (size_t row = 0; row < csr.rows_count(); ++row)
        for (size_t i = csr.row_ptr()[row]; i < csr.row_ptr()[row + 1]; ++i) {
            const size_t dst = next[csr.col_indices()[i]]++;
            row_indices[dst] = row;
            values[dst] = csr.values()[i];
        }

    return MatrixCSC<T>(csr.rows_count(), csr.cols_count(), std::move(row_indices),
                        std::move(col_ptr), std::move(values));
}

template <typename T>
const MatrixCOO<T> &toCOO(const MatrixCOO<T> &coo) { return coo; }

template <typename T>
const MatrixCSR<T> &toCSR(const MatrixCSR<T> &csr) { return csr; }

template <typename T>
const MatrixCSC<T> &toCSC(const MatrixCSC<T> &csc) { return csc; }

} // namespace sparsix
