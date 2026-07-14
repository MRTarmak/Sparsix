#pragma once

#include <utility>
#include <vector>

#include <Sparsix/MatrixCOO.h>
#include <Sparsix/MatrixCSR.h>
#include <Sparsix/MatrixCSC.h>

/** @brief Converts CSR storage to an owning COO matrix. */
template <typename T>
MatrixCOO<T> toCOO(const MatrixCSR<T> &csr) {
    std::vector<Triplet<T>> triplets;
    triplets.reserve(csr.non_zero_count());

    const auto &row_ptr = csr.row_ptr();
    const auto &col_indices = csr.col_indices();
    const auto &values = csr.values();

    for (size_t row = 0; row < csr.rows_count(); row++) {
        for (size_t index = row_ptr[row]; index < row_ptr[row + 1]; index++) {
            triplets.push_back({row, col_indices[index], values[index]});
        }
    }

    return MatrixCOO<T>(csr.rows_count(), csr.cols_count(), std::move(triplets));
}

/** @brief Converts an rvalue CSR matrix to COO without modifying the source. */
template <typename T>
MatrixCOO<T> toCOO(MatrixCSR<T> &&csr) {
    return toCOO(static_cast<const MatrixCSR<T>&>(csr));
}

/** @brief Converts CSC storage to an owning COO matrix. */
template <typename T>
MatrixCOO<T> toCOO(const MatrixCSC<T> &csc) {
    std::vector<Triplet<T>> triplets;
    triplets.reserve(csc.non_zero_count());

    const auto &col_ptr = csc.col_ptr();
    const auto &row_indices = csc.row_indices();
    const auto &values = csc.values();

    for (size_t col = 0; col < csc.cols_count(); col++) {
        for (size_t index = col_ptr[col]; index < col_ptr[col + 1]; index++) {
            triplets.push_back({row_indices[index], col, values[index]});
        }
    }

    return MatrixCOO<T>(csc.rows_count(), csc.cols_count(), std::move(triplets));
}

/** @brief Converts an rvalue CSC matrix to COO without modifying the source. */
template <typename T>
MatrixCOO<T> toCOO(MatrixCSC<T> &&csc) {
    return toCOO(static_cast<const MatrixCSC<T>&>(csc));
}

/** @brief Converts COO storage to CSR. */
template <typename T>
MatrixCSR<T> toCSR(const MatrixCOO<T> &coo) {
    return MatrixCSR<T>(coo);
}

/** @brief Converts an rvalue COO matrix to CSR without modifying the source. */
template <typename T>
MatrixCSR<T> toCSR(MatrixCOO<T> &&coo) {
    return toCSR(static_cast<const MatrixCOO<T>&>(coo));
}

/** @brief Converts CSC storage to CSR. */
template <typename T>
MatrixCSR<T> toCSR(const MatrixCSC<T> &csc) {
    return MatrixCSR<T>(toCOO(csc));
}

/** @brief Converts an rvalue CSC matrix to CSR without modifying the source. */
template <typename T>
MatrixCSR<T> toCSR(MatrixCSC<T> &&csc) {
    return toCSR(static_cast<const MatrixCSC<T>&>(csc));
}

/** @brief Converts COO storage to CSC. */
template <typename T>
MatrixCSC<T> toCSC(const MatrixCOO<T> &coo) {
    return MatrixCSC<T>(coo);
}

/** @brief Converts an rvalue COO matrix to CSC without modifying the source. */
template <typename T>
MatrixCSC<T> toCSC(MatrixCOO<T> &&coo) {
    return toCSC(static_cast<const MatrixCOO<T>&>(coo));
}

/** @brief Converts CSR storage to CSC. */
template <typename T>
MatrixCSC<T> toCSC(const MatrixCSR<T> &csr) {
    return MatrixCSC<T>(toCOO(csr));
}

/** @brief Converts an rvalue CSR matrix to CSC without modifying the source. */
template <typename T>
MatrixCSC<T> toCSC(MatrixCSR<T> &&csr) {
    return toCSC(static_cast<const MatrixCSR<T>&>(csr));
}

/** @brief Returns a COO matrix unchanged. */
template<typename T>
const MatrixCOO<T>& toCOO(const MatrixCOO<T>& coo) {
    return coo;
}

/** @brief Returns a CSR matrix unchanged. */
template<typename T>
const MatrixCSR<T>& toCSR(const MatrixCSR<T>& csr) {
    return csr;
}

/** @brief Returns a CSC matrix unchanged. */
template<typename T>
const MatrixCSC<T>& toCSC(const MatrixCSC<T>& csc) {
    return csc;
}
