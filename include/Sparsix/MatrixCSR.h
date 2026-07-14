#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Sparsix/Concepts/MatrixScalar.h>
#include <Sparsix/Core/MatrixEntry.h>
#include <Sparsix/Core/Triplet.h>
#include <Sparsix/Detail/PrepareTriplets.h>
#include <Sparsix/Detail/TripletsFromDense.h>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/**
 * @brief Sparse matrix stored in compressed sparse row (CSR) format.
 * @tparam T Arithmetic or complex value type.
 *
 * The public API provides construction from dense data or COO triplets, random
 * access, row-oriented mutation, iteration and direct read-only access to CSR
 * arrays (`row_ptr`, `col_indices`, `values`).
 */
class MatrixCSR {
public:
    static_assert(is_matrix_scalar_v<T>, "MatrixCSR requires an arithmetic or std::complex value type.");

    /** @brief Value type stored by the matrix. */
    using value_type = T;

    /** @brief Forward iterator over mutable CSR entries. */
    class Iterator;
    /** @brief Forward iterator over read-only CSR entries. */
    class ConstIterator;

    using iterator = Iterator;
    using const_iterator = ConstIterator;

    /** @brief Returns an iterator to the first stored entry. */
    iterator begin() {
        return iterator(this, 0, 0);
    }

    /** @brief Returns the past-the-end iterator. */
    iterator end() {
        return iterator(this, values_.size(), rows_count_);
    }

    /** @brief Returns a const iterator to the first stored entry. */
    const_iterator begin() const {
        return const_iterator(this, 0, 0);
    }

    /** @brief Returns the const past-the-end iterator. */
    const_iterator end() const {
        return const_iterator(this, values_.size(), rows_count_);
    }

    /** @brief Returns a const iterator to the first stored entry. */
    const_iterator cbegin() const {
        return begin();
    }

    /** @brief Returns the const past-the-end iterator. */
    const_iterator cend() const {
        return end();
    }

    /** @brief Constructs an empty 0-by-0 CSR matrix. */
    explicit MatrixCSR() : col_indices_(), row_ptr_(1, 0), values_(), rows_count_(0), cols_count_(0) {}

    /** @brief Constructs an empty CSR matrix with the requested dimensions. */
    explicit MatrixCSR(size_t rows_count, size_t cols_count) 
        : col_indices_(), row_ptr_(rows_count + 1, 0), values_(), rows_count_(rows_count), cols_count_(cols_count) {}

    /** @brief Constructs a CSR matrix from validated coordinate triplets. */
    explicit MatrixCSR(size_t rows_count, size_t cols_count, const std::vector<Triplet<T>> &triplets) {
        detail::prepare_triplets(rows_count, cols_count, triplets);

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    /** @brief Constructs a CSR matrix from an initializer list of triplets. */
    explicit MatrixCSR(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> triplets) {
        std::vector<Triplet<T>> tmp(triplets);
        detail::prepare_triplets(rows_count, cols_count, tmp);

        initialize(rows_count, cols_count, tmp.begin(), tmp.end());
    }

    /** @brief Constructs a CSR matrix from dense rows, dropping values at the threshold. */
    explicit MatrixCSR(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
        auto triplets = detail::triplets_from_dense(matrix, threshold);

        const size_t rows_count = matrix.size();
        const size_t cols_count = rows_count ? matrix.front().size() : 0;

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    /** @brief Converts a COO matrix to CSR storage. */
    explicit MatrixCSR(MatrixCOO<T> coo) {
        if (!coo.sorted()) {
            coo.sort();
        }

        rows_count_ = coo.rows_count();
        cols_count_ = coo.cols_count();

        col_indices_ = coo.cols();
        values_ = coo.values();

        row_ptr_.assign(rows_count_ + 1, 0);

        for (auto row : coo.rows())
            row_ptr_[row + 1]++;

        for (size_t i = 1; i <= rows_count_; i++)
            row_ptr_[i] += row_ptr_[i - 1];
    }

    /** @brief Constructs CSR storage by taking ownership of raw CSR arrays. */
    explicit MatrixCSR(size_t rows_count, size_t cols_count, 
                       std::vector<size_t> &&col_indices, 
                       std::vector<size_t> &&row_ptr, 
                       std::vector<T> &&values)
                : rows_count_(rows_count), 
                  cols_count_(cols_count), 
                  col_indices_(std::move(col_indices)), 
                  row_ptr_(std::move(row_ptr)), 
                  values_(std::move(values)) {}

    /** @brief Constructs CSR storage from index arrays and an owned value array. */
    explicit MatrixCSR(size_t rows_count, size_t cols_count,
                       const std::vector<size_t> &col_indices,
                       const std::vector<size_t> &row_ptr,
                       std::vector<T> &&values)
                : rows_count_(rows_count), 
                  cols_count_(cols_count), 
                  col_indices_(col_indices), 
                  row_ptr_(row_ptr), 
                  values_(std::move(values)) {}

    /** @brief Returns the value at a coordinate, or zero when it is not stored. */
    T at(size_t row, size_t col) const {
        check_bounds(row, col);
        auto index = find_index_unchecked(row, col);
        if (index) {
            return values_[*index];
        }
        return T{};
    }

    /** @brief Shorthand for at(row, col). */
    T operator()(size_t row, size_t col) const {
        return at(row, col);
    }

    /** @brief Changes matrix dimensions; force permits truncation of stored entries. */
    void reshape(size_t rows_count, size_t cols_count, bool force = false) {
        if (rows_count == 0 || cols_count == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        if (rows_count < rows_count_ || cols_count < cols_count_) {
            if (!force) {
                throw std::invalid_argument(
                    "Reshape may erase stored elements. Pass force=true to allow truncation.");
            } else {
                std::vector<size_t> new_col_indices;
                std::vector<size_t> new_row_ptr;
                std::vector<T> new_values;

                new_col_indices.reserve(col_indices_.size());
                new_values.reserve(values_.size());

                new_row_ptr.assign(rows_count + 1, 0);
                
                for (size_t i = 0; i < rows_count; i++) {
                    size_t begin = row_ptr_[i];
                    size_t end = row_ptr_[i + 1];

                    for (size_t j = begin; j < end; j++) {
                        if (col_indices_[j] < cols_count) {
                            new_col_indices.push_back(col_indices_[j]);
                            new_values.push_back(values_[j]);

                            new_row_ptr[i + 1]++;
                        }
                    }
                }

                for (size_t i = 1; i <= rows_count; i++)
                    new_row_ptr[i] += new_row_ptr[i - 1];

                col_indices_ = std::move(new_col_indices);
                row_ptr_ = std::move(new_row_ptr);
                values_ = std::move(new_values);

                rows_count_ = rows_count;
                cols_count_ = cols_count;
            }
        } else {
            rows_count_ = rows_count;
            cols_count_ = cols_count;
        }
    }

    /** @brief Inserts a new non-zero value at an empty coordinate. */
    void insert(size_t row, size_t col, const T &value) {
        check_bounds(row, col);

        if (value == T{}) {
            throw std::invalid_argument("Cannot insert default (zero) value into CSR matrix.");
        }

        auto row_begin = col_indices_.begin() + row_ptr_[row];
        auto row_end = col_indices_.begin() + row_ptr_[row + 1];

        std::vector<size_t>::iterator position = std::lower_bound(row_begin, row_end, col);

        if (position != row_end && *position == col) {
            throw std::invalid_argument("Element already exists in CSR matrix. \
                                                            Use set() to modify it.");
        }

        size_t index = std::distance(col_indices_.begin(), position);

        col_indices_.insert(position, col);
        values_.insert(values_.begin() + index, value);

        for (size_t i = row + 1; i <= rows_count_; i++)
            row_ptr_[i]++;
    }

    /** @brief Replaces an existing stored non-zero value. */
    void set(size_t row, size_t col, const T &value) {
        check_bounds(row, col);

        if (value == T{}) {
            throw std::invalid_argument("Cannot store default (zero) value in CSR matrix. \
                                                                Use erase() to remove it.");
        }

        auto index = find_index_unchecked(row, col);
        if (index) {
            values_[*index] = value;
            return;
        }

        throw std::out_of_range("Element does not exist in CSR matrix. \
                                                Use insert() to add it.");
    }

    /** @brief Returns true when a coordinate has a stored entry. */
    bool contains(size_t row, size_t col) const {
        check_bounds(row, col);
        if (find_index_unchecked(row, col))
            return true;
        return false;
    }

    /** @brief Erases the stored entry at a coordinate, if present. */
    void erase(size_t row, size_t col) {
        check_bounds(row, col);

        auto index = find_index_unchecked(row, col);
        if (!index)
            return;

        col_indices_.erase(col_indices_.begin() + *index);
        values_.erase(values_.begin() + *index);

        for (size_t i = row + 1; i <= rows_count_; i++)
            row_ptr_[i]--;
    }

    /** @brief Reserves capacity for at least nnz stored entries. */
    void reserve(size_t nnz) {
        col_indices_.reserve(nnz);
        values_.reserve(nnz);
    }

    /** @brief Removes all stored entries while preserving dimensions. */
    void clear() {
        col_indices_.clear();
        values_.clear();
        row_ptr_.assign(rows_count_ + 1, 0);
    }

    /** @brief Returns the number of rows. */
    size_t rows_count() const {
        return rows_count_;
    }

    /** @brief Returns the number of columns. */
    size_t cols_count() const {
        return cols_count_;
    }

    /** @brief Returns the CSR column-index array. */
    const std::vector<size_t> &col_indices() const {
        return col_indices_;
    }

    /** @brief Returns the CSR row-offset array. */
    const std::vector<size_t> &row_ptr() const {
        return row_ptr_;
    }

    /** @brief Returns the stored non-zero values. */
    const std::vector<T> &values() const {
        return values_;
    }

    /** @brief Returns the number of stored non-zero entries. */
    size_t non_zero_count() const {
        return values_.size();
    }

private:
    template <typename InputIt>
    void initialize(size_t rows_count, size_t cols_count, InputIt first, InputIt last) {
        rows_count_ = rows_count;
        cols_count_ = cols_count;

        const auto count = std::distance(first, last);

        col_indices_.reserve(count);
        values_.reserve(count);

        row_ptr_.assign(rows_count_ + 1, 0);

        for (auto it = first; it != last; it++) {
            const auto &entry = *it;

            col_indices_.push_back(entry.col);
            values_.push_back(entry.value);

            row_ptr_[entry.row + 1]++;
        }

        for (size_t i = 1; i <= rows_count_; i++)
            row_ptr_[i] += row_ptr_[i - 1];
    }

    inline void check_bounds(size_t row, size_t col) const {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
    }

    std::optional<size_t> find_index_unchecked(size_t row, size_t col) const {
        size_t l = row_ptr_[row];
        size_t r = row_ptr_[row + 1];
        
        while (l < r) {
            size_t m = l + (r - l) / 2;

            if (col > col_indices_[m]) {
                l = m + 1;
            } else {
                r = m;
            }
        }

        if (l < row_ptr_[row + 1] &&
            col_indices_[l] == col) 
        {
            return l;
        }

        return std::nullopt;
    }

    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptr_;
    std::vector<T> values_;

    size_t rows_count_;
    size_t cols_count_;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/** @brief Forward iterator implementation for MatrixCSR stored entries. */
class MatrixCSR<T>::Iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = MatrixEntry<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    Iterator() : matrix_(nullptr), index_(0), row_(0) {}

    Iterator(MatrixCSR<T> *matrix, size_t index, size_t row) : matrix_(matrix), index_(index), row_(row) {}

    reference operator*() const {
        cache_ = MatrixEntry<T>(row_, matrix_->col_indices_[index_], matrix_->values_[index_]);
        return cache_;
    }

    pointer operator->() const {
        operator*();
        return &cache_;
    }

    Iterator& operator++() {
        index_++;
        while (row_ + 1 < matrix_->row_ptr_.size() && index_ >= matrix_->row_ptr[row_ + 1]) {
            row_++;
        }

        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const Iterator& other) const {
        return matrix_ == other.matrix_ && index_ == other.index_;
    }

    bool operator!=(const Iterator& other) const {
        return !operator==(other);
    }

private:
    MatrixCSR<T> *matrix_;
    size_t index_;
    size_t row_;

    mutable MatrixEntry<T> cache_;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/** @brief Const forward iterator implementation for MatrixCSR stored entries. */
class MatrixCSR<T>::ConstIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ConstMatrixEntry<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    ConstIterator() : matrix_(nullptr), index_(0), row_(0) {}

    ConstIterator(const MatrixCSR<T> *matrix, size_t index, size_t row) : matrix_(matrix), index_(index), row_(row) {}

    ConstIterator(const Iterator &other) : matrix_(other.matrix_), index_(other.index_), row_(other.row_) {}

    reference operator*() const {
        cache_ = ConstMatrixEntry<T>(row_, matrix_->col_indices_[index_], matrix_->values_[index_]);
        return cache_;
    }

    pointer operator->() const {
        operator*();
        return &cache_;
    }

    Iterator& operator++() {
        index_++;
        while (row_ + 1 < matrix_->row_ptr_.size() && index_ >= matrix_->row_ptr[row_ + 1]) {
            row_++;
        }

        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const Iterator& other) const {
        return matrix_ == other.matrix_ && index_ == other.index_;
    }

    bool operator!=(const Iterator& other) const {
        return !operator==(other);
    }

private:
    const MatrixCSR<T> *matrix_;
    size_t index_;
    size_t row_;

    mutable ConstMatrixEntry<T> cache_;
};
