#pragma once

#include <vector>

namespace sparsix {

/** @brief Result returned by an iterative linear solver. */
template<typename T>
struct SolverResult
{
    /** @brief Computed solution vector. */
    std::vector<T> x;
    /** @brief Number of completed iterations. */
    size_t iterations;
    /** @brief Euclidean norm of the final residual. */
    double residual;
    /** @brief True when the requested tolerance was reached. */
    bool converged;
};

} // namespace sparsix
