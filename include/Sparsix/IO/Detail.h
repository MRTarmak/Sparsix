#pragma once

#include <algorithm>
#include <fstream>
#include <string>

#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/Conversions.h>

namespace sparsix::io_detail {
    inline std::string lowercase(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    inline std::string comma_to_space(std::string value) {
        std::replace(value.begin(), value.end(), ',', ' ');
        return value;
    }

    template <typename T>
    T read_value(std::istringstream &input, bool complex_value) {
        if constexpr (is_std_complex_v<T>) {
            using Scalar = typename T::value_type;
            Scalar real{};
            Scalar imag{};
            input >> real;
            if (complex_value)
                input >> imag;
            return T(real, imag);
        } else {
            if (complex_value)
                throw std::invalid_argument("Cannot read complex values into a real matrix type.");
            T value{};
            input >> value;
            return value;
        }
    }

    template <typename T>
    void write_value(std::ostream &output, const T &value) {
        if constexpr (is_std_complex_v<T>)
            output << value.real() << ' ' << value.imag();
        else
            output << value;
    }

    inline std::string next_data_line(std::istream &input, char comment_marker) {
        std::string line;
        while (std::getline(input, line)) {
            const auto first = line.find_first_not_of(" \t\r");
            if (first != std::string::npos && line[first] != comment_marker)
                return line;
        }
        throw std::runtime_error("Unexpected end of matrix file.");
    }

    template <SparseMatrix Matrix>
    Matrix build_matrix(size_t rows, size_t cols, std::vector<Triplet<typename Matrix::value_type>> triplets) {
        const auto coo = MatrixCOO<typename Matrix::value_type>(rows, cols, std::move(triplets));
        if constexpr (std::same_as<Matrix, MatrixCOO<typename Matrix::value_type>>)
            return coo;
        else if constexpr (std::same_as<Matrix, MatrixCSR<typename Matrix::value_type>>)
            return toCSR(coo);
        else if constexpr (std::same_as<Matrix, MatrixCSC<typename Matrix::value_type>>)
            return toCSC(coo);
        else
            return Matrix(coo);
    }
}