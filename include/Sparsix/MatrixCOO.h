#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Sparsix/Concepts/MatrixScalar.h>
#include <Sparsix/Core/MatrixEntry.h>
#include <Sparsix/Core/Triplet.h>
#include <Sparsix/Detail/PrepareTriplets.h>
#include <Sparsix/Detail/TripletsFromDense.h>
#include <Sparsix/Detail/Randomizer.h>

namespace sparsix {

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/**
 * @brief Sparse matrix stored as coordinate triplets (COO).
 * @tparam T Arithmetic or complex value type.
 *
 * Public operations create matrices, access and mutate stored non-zero values,
 * reshape storage, iterate triplets and inspect the coordinate arrays. Entries
 * created from triplets are validated for bounds, duplicate coordinates and zeros.
 */
class MatrixCOO {
public:
    static_assert(is_matrix_scalar_v<T>, "MatrixCOO requires an arithmetic or std::complex value type.");

    /** @brief Value type stored by the matrix. */
    using value_type = T;

    /** @brief Forward iterator over mutable COO entries. */
    class Iterator;
    /** @brief Forward iterator over read-only COO entries. */
    class ConstIterator;

    using iterator = Iterator;
    using const_iterator = ConstIterator;

    /** @brief Returns an iterator to the first stored triplet. */
    iterator begin() {
        return iterator(this, 0, 0);
    }

    /** @brief Returns the past-the-end iterator. */
    iterator end() {
        return iterator(this, values_.size(), rows_count_);
    }

    /** @brief Returns a const iterator to the first stored triplet. */
    const_iterator begin() const {
        return const_iterator(this, 0, 0);
    }

    /** @brief Returns the const past-the-end iterator. */
    const_iterator end() const {
        return const_iterator(this, values_.size(), rows_count_);
    }

    /** @brief Returns a const iterator to the first stored triplet. */
    const_iterator cbegin() const {
        return begin();
    }

    /** @brief Returns the const past-the-end iterator. */
    const_iterator cend() const {
        return end();
    }

    /** @brief Constructs an empty 0-by-0 COO matrix. */
    explicit MatrixCOO() : rows_(), cols_(), values_(), rows_count_(0), cols_count_(0), sorted_(true) {}

    /** @brief Constructs an empty COO matrix with the requested dimensions. */
    explicit MatrixCOO(size_t rows_count, size_t cols_count)
        : rows_(), cols_(), values_(), rows_count_(rows_count), cols_count_(cols_count), sorted_(true) {}

    /** @brief Constructs a COO matrix from owned, validated coordinate triplets. */
    explicit MatrixCOO(size_t rows_count, size_t cols_count, std::vector<Triplet<T>> triplets) {
        detail::prepare_triplets(rows_count, cols_count, triplets);

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    /** @brief Constructs a COO matrix from an initializer list of triplets. */
    explicit MatrixCOO(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> triplets) {
        std::vector<Triplet<T>> tmp(triplets);
        detail::prepare_triplets(rows_count, cols_count, tmp);

        initialize(rows_count, cols_count, tmp.begin(), tmp.end());
    }

    /** @brief Constructs a COO matrix from dense rows, dropping values at the threshold. */
    explicit MatrixCOO(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
        auto triplets = detail::triplets_from_dense(matrix, threshold);

        const size_t rows_count = matrix.size();
        const size_t cols_count = rows_count > 0 ? matrix.front().size() : 0;

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    /** @brief Creates a square diagonal matrix with value on its diagonal. */
    static MatrixCOO<T> create_identity(size_t size, T value = T{1}) {
        std::vector<Triplet<T>> triplets;
        triplets.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            triplets.push_back({i, i, value});
        }
        return MatrixCOO<T>(size, size, triplets);
    }

    /** @brief Creates a square matrix from the supplied diagonal values. */
    static MatrixCOO<T> create_diagonal(size_t size, const std::vector<T> &values) {
        if (values.size() != size) {
            throw std::invalid_argument("Values size must match the specified size.");
        }

        std::vector<Triplet<T>> triplets;
        triplets.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            triplets.push_back({i, i, values[i]});
        }
        return MatrixCOO<T>(size, size, triplets);
    }

    /** @brief Creates a random sparse matrix with a requested density and value range. */
    static MatrixCOO<T> create_random(size_t rows_count, size_t cols_count,
                                      double density, T min_value, T max_value) {
        if (rows_count == 0 || cols_count == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }
        if (density < 0.0 || density > 1.0) {
            throw std::invalid_argument("Density must be in [0, 1].");
        }
        if (cols_count > std::numeric_limits<size_t>::max() / rows_count) {
            throw std::overflow_error("Matrix size exceeds representable range.");
        }

        const size_t total_count = rows_count * cols_count;
        const long double requested_non_zero_count = static_cast<long double>(total_count) * density;
        size_t non_zero_count = static_cast<size_t>(std::llround(requested_non_zero_count));

        std::vector<Triplet<T>> triplets;
        triplets.reserve(non_zero_count);

        if (non_zero_count == 0) {
            return MatrixCOO<T>(rows_count, cols_count, triplets);
        }

        std::vector<size_t> positions(total_count);
        std::iota(positions.begin(), positions.end(), size_t{0});

        auto &generator = detail::Randomizer::generator();
        std::shuffle(positions.begin(), positions.end(), generator);

        auto value_is_zero = [](const T &value) {
            if constexpr (is_std_complex_v<T>) {
                return value.real() == typename T::value_type{} && value.imag() == typename T::value_type{};
            } else {
                return value == T{};
            }
        };

        auto make_matrix = [&](auto make_value) -> MatrixCOO<T> {
            for (size_t i = 0; i < non_zero_count; ++i) {
                const size_t linear_index = positions[i];
                const size_t row = linear_index / cols_count;
                const size_t col = linear_index % cols_count;
                T value = make_value();

                while (value_is_zero(value)) {
                    value = make_value();
                }

                triplets.push_back({row, col, value});
            }

            return MatrixCOO<T>(rows_count, cols_count, triplets);
        };

        if constexpr (is_std_complex_v<T>) {
            using ValueType = typename T::value_type;

            if (min_value.real() > max_value.real() || min_value.imag() > max_value.imag()) {
                throw std::invalid_argument("Invalid complex random range.");
            }
            if (min_value.real() == max_value.real() && min_value.imag() == max_value.imag() && value_is_zero(min_value)) {
                throw std::invalid_argument("Random value range cannot produce a non-zero complex value.");
            }

            std::uniform_real_distribution<ValueType> real_distribution(min_value.real(), max_value.real());
            std::uniform_real_distribution<ValueType> imag_distribution(min_value.imag(), max_value.imag());

            auto make_value = [&]() -> T {
                return T(real_distribution(generator), imag_distribution(generator));
            };

            return make_matrix(make_value);
        } else {
            if (min_value > max_value) {
                throw std::invalid_argument("Invalid random range.");
            }
            if (min_value == max_value && value_is_zero(min_value)) {
                throw std::invalid_argument("Random value range cannot produce a non-zero value.");
            }

            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> distribution(min_value, max_value);

                auto make_value = [&]() -> T {
                    return distribution(generator);
                };

                return make_matrix(make_value);
            } else {
                std::uniform_real_distribution<T> distribution(min_value, max_value);

                auto make_value = [&]() -> T {
                    return distribution(generator);
                };

                return make_matrix(make_value);
            }
        }
    }

    /** @brief Sorts triplets by row or column major order. */
    void sort() {
        if (sorted_) {
            return;
        }
        if (values_.size() <= 1) {
            sorted_ = true;
            return;
        }

        std::vector<size_t> order(values_.size());
        std::iota(order.begin(), order.end(), size_t{0});

        std::sort(order.begin(), order.end(),
        [&](auto a, auto b) {
            if (rows_[a] != rows_[b])
                return rows_[a] < rows_[b];
            return cols_[a] < cols_[b];
        });

        std::vector<size_t> new_rows;
        std::vector<size_t> new_cols;
        std::vector<T> new_values;

        new_rows.reserve(rows_.size());
        new_cols.reserve(cols_.size());
        new_values.reserve(values_.size());

        for (size_t i = 0; i < order.size(); ++i) {
            new_rows.push_back(rows_[order[i]]);
            new_cols.push_back(cols_[order[i]]);
            new_values.push_back(values_[order[i]]);
        }

        rows_ = std::move(new_rows);
        cols_ = std::move(new_cols);
        values_ = std::move(new_values);

        sorted_ = true;
    };

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
                std::vector<Triplet<T>> triplets;
                triplets.reserve(values_.size());
                for (size_t i = 0; i < values_.size(); ++i) {
                    if (rows_[i] < rows_count && cols_[i] < cols_count) {
                        triplets.emplace_back(rows_[i], cols_[i], values_[i]);
                    }
                }
                rows_.clear();
                cols_.clear();
                values_.clear();

                detail::prepare_triplets(rows_count, cols_count, triplets);
                initialize(rows_count, cols_count, triplets.begin(), triplets.end());
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
            throw std::invalid_argument("Cannot insert default (zero) value into COO matrix.");
        }

        auto index = find_index_unchecked(row, col);
        if (index) {
            throw std::invalid_argument("Element already exists in COO matrix. \
                                                            Use set() to modify it.");
        }

        rows_.push_back(row);
        cols_.push_back(col);
        values_.push_back(value);

        sorted_ = false;
    }

    /** @brief Replaces an existing stored non-zero value. */
    void set(size_t row, size_t col, const T &value) {
        check_bounds(row, col);

        if (value == T{}) {
            throw std::invalid_argument("Cannot store default (zero) value in COO matrix. \
                                                                Use erase() to remove it.");
        }

        auto index = find_index_unchecked(row, col);
        if (index) {
            values_[*index] = value;
            return;
        }

        throw std::out_of_range("Element does not exist in COO matrix. \
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
        if (index) {
            rows_.erase(rows_.begin() + *index);
            cols_.erase(cols_.begin() + *index);
            values_.erase(values_.begin() + *index);
            return;
        }
    }

    /** @brief Reserves capacity for at least nnz stored entries. */
    void reserve(size_t nnz) {
        rows_.reserve(nnz);
        cols_.reserve(nnz);
        values_.reserve(nnz);
    }

    /** @brief Removes all stored entries while preserving dimensions. */
    void clear() {
        rows_.clear();
        cols_.clear();
        values_.clear();

        sorted_ = true;
    }

    /** @brief Returns the stored zero-based row indices. */
    const std::vector<size_t> &rows() const {
        return rows_;
    }

    /** @brief Returns the stored zero-based column indices. */
    const std::vector<size_t> &cols() const {
        return cols_;
    }

    /** @brief Returns the stored non-zero values. */
    const std::vector<T> &values() const {
        return values_;
    }

    /** @brief Returns the number of rows. */
    size_t rows_count() const {
        return rows_count_;
    }

    /** @brief Returns the number of columns. */
    size_t cols_count() const {
        return cols_count_;
    }

    /** @brief Reports whether triplets are sorted in row-major order. */
    bool sorted() const {
        return sorted_;
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

        rows_.reserve(count);
        cols_.reserve(count);
        values_.reserve(count);

        for (auto it = first; it != last; ++it) {
            const auto &entry = *it;

            rows_.push_back(entry.row);
            cols_.push_back(entry.col);
            values_.push_back(entry.value);
        }

        sorted_ = true;
    }

    inline void check_bounds(size_t row, size_t col) const {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
    }

    std::optional<size_t> find_index_unchecked(size_t row, size_t col) const {
        if (!sorted_) {
            for (size_t i = 0; i < values_.size(); ++i) {
                if (rows_[i] == row && cols_[i] == col) {
                    return i;
                }
            }
        } else {
            size_t l = 0;
            size_t r = values_.size();

            while (l < r) {
                size_t m = l + (r - l) / 2;

                const auto current = std::pair{rows_[m], cols_[m]};
                const auto target = std::pair{row, col};

                if (target > current) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }

            if (l < values_.size() &&
                rows_[l] == row &&
                cols_[l] == col)
            {
                return l;
            }
        }

        return std::nullopt;
    }

    std::vector<size_t> rows_;
    std::vector<size_t> cols_;
    std::vector<T> values_;

    size_t rows_count_;
    size_t cols_count_;

    bool sorted_;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/** @brief Forward iterator implementation for MatrixCOO stored entries. */
class MatrixCOO<T>::Iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = MatrixEntry<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    Iterator() : matrix_(nullptr), index_(0) {}

    Iterator(MatrixCOO<T> *matrix, size_t index) : matrix_(matrix), index_(index) {}

    reference operator*() const {
        cache_ = MatrixEntry<T>(matrix_->rows_[index_], matrix_->cols_[index_], matrix_->values_[index_]);
        return cache_;
    }

    pointer operator->() const {
        operator*();
        return &cache_;
    }

    Iterator& operator++() {
        ++index_;

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
    MatrixCOO<T> *matrix_;
    size_t index_;

    mutable MatrixEntry<T> cache_;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
/** @brief Const forward iterator implementation for MatrixCOO stored entries. */
class MatrixCOO<T>::ConstIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ConstMatrixEntry<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    ConstIterator() : matrix_(nullptr), index_(0) {}

    ConstIterator(const MatrixCOO<T> *matrix, size_t index) : matrix_(matrix), index_(index) {}

    ConstIterator(const Iterator &other) : matrix_(other.matrix_), index_(other.index_) {}

    reference operator*() const {
        cache_ = ConstMatrixEntry<T>(matrix_->rows_[index_], matrix_->cols_[index_], matrix_->values_[index_]);
        return cache_;
    }

    pointer operator->() const {
        operator*();
        return &cache_;
    }

    Iterator& operator++() {
        ++index_;

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
    const MatrixCOO<T> *matrix_;
    size_t index_;

    mutable ConstMatrixEntry<T> cache_;
};

} // namespace sparsix
