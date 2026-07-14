#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Sparsix/Core/MajorOrder.h>
#include <Sparsix/Core/Triplet.h>

namespace detail {
    /**
     * @brief Validates, removes zero values and orders construction triplets.
     * @throws std::invalid_argument For invalid dimensions or duplicate coordinates.
     * @throws std::out_of_range For coordinates outside the matrix dimensions.
     */
    template <typename T>
    void prepare_triplets(size_t rows_count,
                          size_t cols_count,
                          std::vector<Triplet<T>> &triplets,
                          MajorOrder order = MajorOrder::RowOrder)
    {
        if (rows_count == 0 || cols_count == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        for (const auto &triplet : triplets) {
            if (triplet.row >= rows_count || triplet.col >= cols_count) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }
        }

        triplets.erase(
            std::remove_if(triplets.begin(), triplets.end(), 
                        [](const Triplet<T> &triplet) {
                            return triplet.value == T{};
                        }), 
            triplets.end());

        if (order == MajorOrder::RowOrder) {
            std::sort(triplets.begin(), triplets.end(), 
            [](const Triplet<T> &a, const Triplet<T> &b) {
                if (a.row != b.row)
                    return a.row < b.row;
                return a.col < b.col;
            });
        } else {
            std::sort(triplets.begin(), triplets.end(),
            [](const Triplet<T> &a, const Triplet<T> &b) {
                if (a.col != b.col)
                    return a.col < b.col;
                return a.row < b.row;
            });
        }

        auto dup = std::adjacent_find(triplets.begin(), triplets.end(),
            [](const auto &a, const auto &b) {
                return a.row == b.row &&
                       a.col == b.col;
            });

        if (dup != triplets.end()) {
            throw std::invalid_argument("Duplicate entry found.");
        }
    }
}
