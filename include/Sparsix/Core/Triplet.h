#pragma once

#include <cstddef>

#include <Sparsix/Concepts/MatrixScalar.h>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/** @brief A coordinate and value used to construct sparse matrices. */
struct Triplet {
    /** @brief Zero-based row index. */
    size_t row;
    /** @brief Zero-based column index. */
    size_t col;
    /** @brief Stored non-zero value. */
    T value;
};
