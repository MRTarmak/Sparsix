#pragma once

#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Sparsix/MatrixCSR.h>
#include <Sparsix/Algorithms/Solvers/SolverResult.h>
#include <Sparsix/Algorithms/Utilities.h>
#include <Sparsix/Algorithms/VectorAlgorithms.h>

namespace sparsix {
    // Precondition: A must be symmetric positive definite.
    template <typename T>
    SolverResult<T> cg(
        const MatrixCSR<T> &A,
        const std::vector<T> &b,
        std::optional<std::vector<T>> x0 = std::nullopt,
        size_t max_iterations = 1000,
        real_type<T> tolerance = 1e-8
    ) {
        if (!is_symmetric(A))
            throw std::invalid_argument("Matrix A must be symmetric. Use sparsix::is_symmetric(A) to check symmetry.");

        if (b.size() != A.rows_count())
            throw std::invalid_argument("The number of rows in A and the size of vector b must match.");
        
        if (x0 && x0->size() != A.cols_count())
            throw std::invalid_argument("The number of columns in A and the size of vector x0 must match.");

        std::vector<T> x;
        if (!x0)
            x.assign(A.cols_count(), 0);
        else
            x = *x0;

        std::vector<T> r = b;
        axpy(T{-1}, A * x, r);

        auto residual = norm2(r);
        if (residual <= tolerance)
            return SolverResult<T>(x, 0, residual, true);

        std::vector<T> p = r;
        auto rr = dot(r, r);

        size_t iterations = 0;
        while(iterations < max_iterations) {
            iterations++;
            auto Ap = A * p;

            auto denominator = dot(p, Ap);
            if (std::abs(denominator) <= std::numeric_limits<real_type<T>>::epsilon())
                break;
            auto alpha = rr / denominator;

            axpy(alpha, p, x);
            axpy(-alpha, Ap, r);

            residual = norm2(r);
            if (residual <= tolerance)
                return SolverResult<T>(x, iterations, residual, true);

            if (std::abs(rr) <= std::numeric_limits<real_type<T>>::epsilon())
                break;
            auto rr_new = dot(r, r);
            auto beta = rr_new / rr;

            rr = rr_new;

            scal(p, beta);
            axpy(T{1}, r, p);
        }

        return SolverResult<T>(x, iterations, norm2(r), false);
    }
}
