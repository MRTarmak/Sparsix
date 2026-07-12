#pragma once

#include <vector>

#include <Sparsix/Concepts/MatrixScalar.h>
#include <Sparsix/Core/Triplet.h>
#include <Sparsix/Detail/PrepareTriplets.h>
#include <Sparsix/Detail/TripletsFromDense.h>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
class MatrixCSC {
public:
    static_assert(is_matrix_scalar_v<T>, "MatrixCSC requires an arithmetic or std::complex value type.");

    using value_type = T;

    explicit MatrixCSC() : row_indices_(), col_ptr_(1, 0), values_(), rows_count_(0), cols_count_(0) {}

    explicit MatrixCSC(size_t rows_count, size_t cols_count) 
        : row_indices_(), col_ptr_(cols_count + 1, 0), values_(), rows_count_(rows_count), cols_count_(cols_count) {}

    explicit MatrixCSC(size_t rows_count, size_t cols_count, const std::vector<Triplet<T>> &triplets) {
        detail::prepare_triplets(rows_count, cols_count, triplets, MajorOrder::ColumnOrder);

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    explicit MatrixCSC(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> triplets) {
        std::vector<Triplet<T>> tmp(triplets);
        detail::prepare_triplets(rows_count, cols_count, tmp, MajorOrder::ColumnOrder);

        initialize(rows_count, cols_count, tmp.begin(), tmp.end());
    }

    explicit MatrixCSC(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
        auto triplets = detail::triplets_from_dense(matrix, threshold);

        const size_t rows_count = matrix.size();
        const size_t cols_count = rows_count ? matrix.front().size() : 0;

        detail::prepare_triplets(rows_count, cols_count, triplets, MajorOrder::ColumnOrder);
        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    explicit MatrixCSC(MatrixCOO<T> coo) {
        coo.sort(MajorOrder::ColumnOrder);

        rows_count_ = coo.rows_count();
        cols_count_ = coo.cols_count();

        row_indices_ = coo.rows();
        values_ = coo.values();

        col_ptr_.assign(cols_count_ + 1, 0);

        for (auto col : coo.cols())
            col_ptr_[col + 1]++;

        for (size_t i = 1; i <= cols_count_; i++)
            col_ptr_[i] += col_ptr_[i - 1];
    }

    explicit MatrixCSC(size_t rows_count, size_t cols_count, 
                       std::vector<size_t> &&row_indices, 
                       std::vector<size_t> &&col_ptr, 
                       std::vector<T> &&values)
                : rows_count_(rows_count), 
                  cols_count_(cols_count), 
                  row_indices_(std::move(row_indices)), 
                  col_ptr_(std::move(col_ptr)), 
                  values_(std::move(values)) {}

    T at(size_t row, size_t col) const {
        check_bounds(row, col);
        auto index = find_index_unchecked(row, col);
        if (index) {
            return values_[*index];
        }
        return T{};
    }

    T operator()(size_t row, size_t col) const {
        return at(row, col);
    }

    void reshape(size_t rows_count, size_t cols_count, bool force = false) {
        if (rows_count == 0 || cols_count == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        if (rows_count < rows_count_ || cols_count < cols_count_) {
            if (!force) {
                throw std::invalid_argument(
                    "Reshape may erase stored elements. Pass force=true to allow truncation.");
            } else {
                std::vector<size_t> new_row_indices;
                std::vector<size_t> new_col_ptr;
                std::vector<T> new_values;

                new_row_indices.reserve(row_indices_.size());
                new_values.reserve(values_.size());

                new_col_ptr.assign(cols_count + 1, 0);

                for (size_t col = 0; col < cols_count; col++) {
                    size_t begin = col_ptr_[col];
                    size_t end = col_ptr_[col + 1];

                    for (size_t i = begin; i < end; i++) {
                        if (row_indices_[i] < rows_count) {
                            new_row_indices.push_back(row_indices_[i]);
                            new_values.push_back(values_[i]);

                            new_col_ptr[col + 1]++;
                        }
                    }
                }

                for (size_t i = 1; i <= cols_count; i++)
                    new_col_ptr[i] += new_col_ptr[i - 1];

                row_indices_ = std::move(new_row_indices);
                col_ptr_ = std::move(new_col_ptr);
                values_ = std::move(new_values);

                rows_count_ = rows_count;
                cols_count_ = cols_count;
            }
        } else {
            rows_count_ = rows_count;
            cols_count_ = cols_count;
        }
    }

    void insert(size_t row, size_t col, const T &value) {
        check_bounds(row, col);

        if (value == T{}) {
            throw std::invalid_argument("Cannot insert default (zero) value into CSC matrix.");
        }

        auto col_begin = row_indices_.begin() + col_ptr_[col];
        auto col_end = row_indices_.begin() + col_ptr_[col + 1];

        std::vector<size_t>::iterator position = std::lower_bound(col_begin, col_end, row);

        if (position != col_end && *position == row) {
            throw std::invalid_argument("Element already exists in CSC matrix. \
                                                            Use set() to modify it.");
        }

        size_t index = std::distance(row_indices_.begin(), position);

        row_indices_.insert(position, row);
        values_.insert(values_.begin() + index, value);

        for (size_t i = col + 1; i <= cols_count_; i++)
            col_ptr_[i]++;
    }

    void set(size_t row, size_t col, const T &value) {
        check_bounds(row, col);

        if (value == T{}) {
            throw std::invalid_argument("Cannot store default (zero) value in CSC matrix. \
                                                                Use erase() to remove it.");
        }

        auto index = find_index_unchecked(row, col);
        if (index) {
            values_[*index] = value;
            return;
        }

        throw std::out_of_range("Element does not exist in CSC matrix. \
                                                Use insert() to add it.");
    }

    bool contains(size_t row, size_t col) const {
        check_bounds(row, col);
        if (find_index_unchecked(row, col))
            return true;
        return false;
    }

    void erase(size_t row, size_t col) {
        check_bounds(row, col);

        auto index = find_index_unchecked(row, col);
        if (!index)
            return;

        row_indices_.erase(row_indices_.begin() + *index);
        values_.erase(values_.begin() + *index);

        for (size_t i = col + 1; i <= cols_count_; i++)
            col_ptr_[i]--;
    }

    void reserve(size_t nnz) {
        row_indices_.reserve(nnz);
        values_.reserve(nnz);
    }

    void clear() {
        row_indices_.clear();
        values_.clear();
        col_ptr_.assign(cols_count_ + 1, 0);
    }

    size_t rows_count() const {
        return rows_count_;
    }

    size_t cols_count() const {
        return cols_count_;
    }

    const std::vector<size_t> &row_indices() const {
        return row_indices_;
    }

    const std::vector<size_t> &col_ptr() const {
        return col_ptr_;
    }

    const std::vector<T> &values() const {
        return values_;
    }

    size_t non_zero_count() const {
        return values_.size();
    }

private:
    template <typename InputIt>
    void initialize(size_t rows_count, size_t cols_count, InputIt first, InputIt last) {
        rows_count_ = rows_count;
        cols_count_ = cols_count;

        const auto count = std::distance(first, last);

        row_indices_.reserve(count);
        values_.reserve(count);

        col_ptr_.assign(cols_count_ + 1, 0);

        for (auto it = first; it != last; it++) {
            const auto &entry = *it;

            row_indices_.push_back(entry.row);
            values_.push_back(entry.value);

            col_ptr_[entry.col + 1]++;
        }

        for (size_t i = 1; i <= cols_count_; i++)
            col_ptr_[i] += col_ptr_[i - 1];
    }

    inline void check_bounds(size_t row, size_t col) const {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
    }

    std::optional<size_t> find_index_unchecked(size_t row, size_t col) const {
        size_t l = col_ptr_[col];
        size_t r = col_ptr_[col + 1];
        
        while (l < r) {
            size_t m = l + (r - l) / 2;

            if (row > row_indices_[m]) {
                l = m + 1;
            } else {
                r = m;
            }
        }

        if (l < col_ptr_[col + 1] &&
            row_indices_[l] == row) 
        {
            return l;
        }

        return std::nullopt;
    }

    std::vector<size_t> row_indices_;
    std::vector<size_t> col_ptr_;
    std::vector<T> values_;

    size_t rows_count_;
    size_t cols_count_;
};