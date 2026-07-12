#pragma once

template<typename T>
struct MatrixEntry {
    size_t row;
    size_t col;
    T &value;
};

template<typename T>
struct ConstMatrixEntry {
    size_t row;
    size_t col;
    const T &value;
};