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

/**
 * @brief Solves a general square linear system with BiCGSTAB.
 * @param A Coefficient matrix in CSR format.
 * @param b Right-hand-side vector.
 * @param x0 Optional initial solution estimate.
 * @param max_iterations Maximum number of iterations.
 * @param tolerance Target Euclidean residual norm.
 * @return Solution, residual, iteration count and convergence state.
 * @throws std::invalid_argument If vector sizes do not match A.
 */
template <typename T>
SolverResult<T> bicgstab(
    const MatrixCSR<T> &A,
    const std::vector<T> &b,
    const std::optional<std::vector<T>> &x0 = std::nullopt,
    size_t max_iterations = 1000,
    real_type<T> tolerance = 1e-8
) {
    if (A.rows_count() != A.cols_count())
        throw std::invalid_argument("Matrix A must be square.");
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
        return SolverResult<T>(std::move(x), 0, residual, true);
    std::vector<T> r_hat = r;
    T rho = T{1};
    T alpha = T{1};
    T omega = T{1};
    std::vector<T> p(b.size(), 0);
    std::vector<T> v(b.size(), 0);
    std::vector<T> s(b.size());
    std::vector<T> t(b.size());
    size_t iterations = 0;
    while(iterations < max_iterations) {
        T rho_old = rho;
        rho = dot(r_hat, r);
        if (std::abs(rho) <= std::numeric_limits<real_type<T>>::epsilon())
            break;
        T beta = T{};
        if (iterations != 0) {
            beta = rho / rho_old * alpha / omega;
        }
        ++iterations;

        axpy(-omega, v, p);
        scal(p, beta);
        axpy(T{1}, r, p);
        v = A * p;
        auto denominator = dot(r_hat, v);
        if (std::abs(denominator) <= std::numeric_limits<real_type<T>>::epsilon())
            break;
        alpha = rho / denominator;
        s = r;
        axpy(-alpha, v, s);
        residual = norm2(s);
        if (residual <= tolerance) {
            axpy(alpha, p, x);
            return SolverResult<T>(std::move(x), iterations, residual, true);
        }
        t = A * s;
        denominator = dot(t, t);
        if (std::abs(denominator) <= std::numeric_limits<real_type<T>>::epsilon())
            break;
        omega = dot(t, s) / denominator;
        if (std::abs(omega) <= std::numeric_limits<real_type<T>>::epsilon())
            break;
        axpy(alpha, p, x);
        axpy(omega, s, x);
        r = s;
        axpy(-omega, t, r);
        residual = norm2(r);
        if (residual <= tolerance) {
            return SolverResult<T>(std::move(x), iterations, residual, true);
        }
    }
    return SolverResult<T>(std::move(x), iterations, norm2(r), false);
}

} // namespace sparsix
