#pragma once

#include <cstddef>

/** @brief Mutable-looking value object returned by sparse matrix iterators. */
template<typename T>
struct MatrixEntry {
    /** @brief Zero-based row index. */
    size_t row;
    /** @brief Zero-based column index. */
    size_t col;
    /** @brief Reference to the stored value. */
    T &value;
};

/** @brief Read-only value object returned by const sparse matrix iterators. */
template<typename T>
struct ConstMatrixEntry {
    /** @brief Zero-based row index. */
    size_t row;
    /** @brief Zero-based column index. */
    size_t col;
    /** @brief Read-only reference to the stored value. */
    const T &value;
};
