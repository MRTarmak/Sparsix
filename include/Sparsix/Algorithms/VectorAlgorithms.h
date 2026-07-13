#pragma once

#include <stdexcept>
#include <vector>

#include <Sparsix/Concepts/MatrixScalar.h>

namespace sparsix {
    template <typename T>
    constexpr T conjugate(const T &value) {
        return value;
    }

    template <typename T>
    constexpr std::complex<T> conjugate(const std::complex<T> &value) {
        return std::conj(value);
    }

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

    template <typename T>
    real_type<T> norm2(const std::vector<T> &x) {
        return std::sqrt(std::real(dot(x, x)));
    }

    template <typename T>
    void axpy(const T &alpha, const std::vector<T> &x, std::vector<T> &y) {
        if (x.size() != y.size())
            throw std::invalid_argument("Vectors sizes must match.");

        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    template <typename T>
    void scal(std::vector<T> &x, const T &alpha) {
        for (size_t i = 0; i < x.size(); i++) {
            x[i] *= alpha;
        }
    }
}
