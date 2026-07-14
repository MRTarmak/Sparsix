#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Sparsix/Concepts/MatrixScalar.h>

namespace sparsix {
    /** @brief Returns the conjugate of a real scalar (the value itself). */
    template <typename T>
    constexpr T conjugate(const T &value) {
        return value;
    }

    /** @brief Returns the complex conjugate of a complex scalar. */
    template <typename T>
    constexpr std::complex<T> conjugate(const std::complex<T> &value) {
        return std::conj(value);
    }

    /** @brief Computes the Hermitian dot product of two equal-sized vectors. */
    template <typename T>
    T dot(const std::vector<T> &a, const std::vector<T> &b) {
        if (a.size() != b.size())
            throw std::invalid_argument("Vectors sizes must match.");

        T result = T{};
        for (size_t i = 0; i < a.size(); i++) {
            result += conjugate(a[i]) * b[i];
        }

        return result;
    }

    /** @brief Computes the Euclidean norm of a vector. */
    template <typename T>
    real_type<T> norm2(const std::vector<T> &x) {
        return std::sqrt(std::real(dot(x, x)));
    }

    /** @brief Performs y += alpha * x for equal-sized vectors. */
    template <typename T>
    void axpy(const T &alpha, const std::vector<T> &x, std::vector<T> &y) {
        if (x.size() != y.size())
            throw std::invalid_argument("Vectors sizes must match.");

        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    /** @brief Multiplies every vector element by alpha. */
    template <typename T>
    void scal(std::vector<T> &x, const T &alpha) {
        for (size_t i = 0; i < x.size(); i++) {
            x[i] *= alpha;
        }
    }
}
