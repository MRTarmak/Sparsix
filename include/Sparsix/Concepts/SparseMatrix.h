#pragma once

/** @brief Constraint for matrix types exposing dimensions and a value_type. */
template<typename M>
concept SparseMatrix = requires(M m) {
    typename M::value_type;

    m.rows_count();
    m.cols_count();
};
