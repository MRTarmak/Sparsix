#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Sparsix/Concepts/SparseMatrix.h>
#include <Sparsix/Detail/Conversions.h>
#include <Sparsix/IO/Detail.h>

namespace sparsix {
    /**
     * @brief Writes a matrix as a CSV-like list of zero-based triplets.
     * @param matrix Matrix to serialize.
     * @param path Destination file.
     *
     * The first line is `rows,cols,nnz`; real entries use `row,col,value` and
     * complex entries use `row,col,real,imag`.
     */
    template <SparseMatrix Matrix>
    void write_triplets_csv(const Matrix &matrix, const std::filesystem::path &path) {
        std::ofstream output(path);
        if (!output)
            throw std::runtime_error("Unable to open CSV file for writing: " + path.string());

        const auto &coo = toCOO(matrix);
        output << coo.rows_count() << ',' << coo.cols_count() << ',' << coo.non_zero_count() << '\n';
        output << std::setprecision(std::numeric_limits<long double>::max_digits10);
        for (size_t i = 0; i < coo.non_zero_count(); i++) {
            output << coo.rows()[i] << ',' << coo.cols()[i] << ',';
            if constexpr (is_std_complex_v<typename Matrix::value_type>)
                output << coo.values()[i].real() << ',' << coo.values()[i].imag();
            else
                output << coo.values()[i];
            output << '\n';
        }
    }

    /**
     * @brief Reads a CSV-like triplet file into a selected sparse matrix format.
     * @tparam Matrix Target sparse matrix type.
     * @param path Source file written by write_triplets_csv().
     * @return Parsed matrix.
     * @throws std::invalid_argument For malformed dimensions, coordinates or values.
     */
    template <SparseMatrix Matrix>
    Matrix read_triplets_csv(const std::filesystem::path &path) {
        using T = typename Matrix::value_type;
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Unable to open CSV file for reading: " + path.string());

        const auto header = io_detail::next_data_line(input, '#');
        std::istringstream dimensions(io_detail::comma_to_space(header));
        size_t rows{}, cols{}, nnz{};
        if (!(dimensions >> rows >> cols >> nnz))
            throw std::invalid_argument("Invalid CSV dimensions line.");

        std::vector<Triplet<T>> triplets;
        triplets.reserve(nnz);
        for (size_t i = 0; i < nnz; i++) {
            const auto line = io_detail::next_data_line(input, '#');
            std::istringstream entry(io_detail::comma_to_space(line));
            size_t row{}, col{};
            if (!(entry >> row >> col) || row >= rows || col >= cols)
                throw std::invalid_argument("Invalid CSV coordinate.");
            const T value = io_detail::read_value<T>(entry, is_std_complex_v<T>);
            if (!entry)
                throw std::invalid_argument("Invalid CSV value.");
            triplets.push_back({row, col, value});
        }

        return io_detail::build_matrix<Matrix>(rows, cols, std::move(triplets));
    }

    /** @brief Reads a CSV-like triplet file into COO storage. */
    template <typename T>
    MatrixCOO<T> load_triplets_csv(const std::filesystem::path &path) {
        return read_triplets_csv<MatrixCOO<T>>(path);
    }

    /** @brief Writes a matrix as CSV-like triplets; alias for write_triplets_csv(). */
    template <SparseMatrix Matrix>
    void save_triplets_csv(const Matrix &matrix, const std::filesystem::path &path) {
        write_triplets_csv(matrix, path);
    }
}
