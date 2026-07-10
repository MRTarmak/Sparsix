#pragma once

#include <cstddef>
#include <vector>

namespace detail {
    template <typename T>
    void prepare_triplets(size_t rows_count,
                              size_t cols_count,
                              std::vector<Triplet<T>> &triplets)
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

        std::sort(triplets.begin(), triplets.end());

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