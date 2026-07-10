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
class MatrixCSR {
    static_assert(is_matrix_scalar_v<T>, "MatrixCOO requires an arithmetic or std::complex value type.");

    explicit MatrixCSR() : col_indices_(), row_ptr_(), values_(), rows_count_(0), cols_count_(0) {}

    explicit MatrixCSR(size_t rows_count, size_t cols_count) 
        : col_indices_(), row_ptr_(), values_(), rows_count_(rows_count), cols_count_(cols_count) {}

    explicit MatrixCSR(size_t rows_count, size_t cols_count, const std::vector<Triplet<T>> &triplets) {
        detail::prepare_triplets(rows_count, cols_count, triplets);

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    explicit MatrixCSR(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> triplets) {
        detail::prepare_triplets(rows_count, cols_count, triplets);

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    explicit MatrixCSR(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
        auto triplets = detail::triplets_from_dense(matrix, threshold);

        rows_count = matrix.size();
        cols_count = rows_count > 0 ? matrix.front().size() : 0;

        initialize(rows_count, cols_count, triplets.begin(), triplets.end());
    }

    explicit MatrixCSR(MatrixCOO<T> coo);

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

    void reshape(size_t rows_count, size_t cols_count, bool force = false);

    void insert(size_t row, size_t col, const T &value);

    void set(size_t row, size_t col, const T& value);

    bool contains(size_t row, size_t col) const;

    void erase(size_t row, size_t col);

    void reserve(size_t nnz);

    void clear();

    size_t rows_count() const {
        return rows_count_;
    }

    size_t cols_count() const {
        return cols_count_;
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
            row_ptr_[i] += row_ptr_[i-1];
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