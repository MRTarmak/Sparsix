#pragma once

#include <random>

/** @brief Owns the shared random engine used by sparse matrix generators. */
class Randomizer {
public:
    /** @brief Returns the process-wide Mersenne Twister engine. */
    static std::mt19937 &generator() {
        static std::mt19937 gen{std::random_device{}()};
        return gen;
    }
};
