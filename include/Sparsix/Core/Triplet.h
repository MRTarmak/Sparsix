#pragma once

#include <cstddef>

#include <Sparsix/Concepts/MatrixScalar.h>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
struct Triplet {
    size_t row;
    size_t col;
    T value;

    auto operator<=>(const Triplet&) const = default;
};