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
            throw std::invalid_argument("The number of columns in A and the size of vector b must match.");
        
        if (x0 && x0->size() != A.cols_count())
            throw std::invalid_argument("The number of rows in A and the size of vector x0 must match.");

        std::vector<T> x;
        if (!x0)
            x.assign(A.cols_count(), 0);
        else
            x = *x0;

        auto estimation = A * x;
        std::vector<T> r(b.size());
        for (size_t i = 0; i < b.size(); i++) {
            r[i] = b[i] - estimation[i];
        }
        std::vector<T> p = r;
        auto rr = dot(r, r);

        if (norm2(r) <= tolerance)
            return SolverResult(x, 0, norm2(r), true);

        size_t iterations = 0;
        while(iterations < max_iterations) {
            iterations++;
            auto Ap = A * p;

            auto denominator = dot(p, Ap);
            if (std::abs(denominator) == real_type<T>{})
                throw std::runtime_error("Division by zero.");
            auto alpha = rr / denominator;

            axpy(alpha, p, x);
            axpy(-alpha, Ap, r);

            if (norm2(r) <= tolerance)
                return SolverResult(x, iterations, norm2(r), true);

            if (std::abs(rr) == real_type<T>{})
                throw std::runtime_error("Division by zero.");
            auto rr_new = dot(r, r);
            auto beta = rr_new / rr;

            rr = rr_new;

            scal(p, beta);
            axpy(T{1}, r, p);
        }

        return SolverResult(x, iterations, norm2(r), false);
    }
}
