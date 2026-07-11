#pragma once

#include <random>

class Randomizer {
public:
    static std::mt19937 &generator() {
        static std::mt19937 gen{std::random_device{}()};
        return gen;
    }
};