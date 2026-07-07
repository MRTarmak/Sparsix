#include <vector>

#include <Sparsix/Concepts/MatrixScalar.h>
#include <Sparsix/Core/Triplet.h>

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

    explicit MatrixCSR(size_t rows_count, size_t cols_count, const std::vector<Triplet<T>> &entries) {
        initialize(rows_count, cols_count, entries.begin(), entries.end());
    }

    explicit MatrixCSR(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> entries) {
        initialize(rows_count, cols_count, entries.begin(), entries.end());
    }

    explicit MatrixCSR(const std::vector<std::vector<T>> &matrix, T threshold = T{});

    T at(size_t row, size_t col) const;

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
        if (rows_count == 0 || cols_count == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        rows_count_ = rows_count;
        cols_count_ = cols_count;

        const auto count = std::distance(first, last);

        if (count > 1) {
            std::sort(first, last, 
            [](const Triplet<T> &a, const Triplet<T> &b) {
                if (a.row != b.row)
                    return a.row < b.row;

                return a.col < b.col;
            });
        }

        col_indices_.reserve(count);
        values_.reserve(count);

        row_ptr_.assign(rows_count_ + 1, 0);

        for (auto it = first; it != last; it++) {
            const auto &entry = *it;

            if (entry.row >= rows_count_ || entry.col >= cols_count_) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }

            if (entry.value == T{}) {
                continue;
            }

            if (it != first) {
                auto prev = std::prev(it);

                if (prev->row == entry.row &&
                    prev->col == entry.col)
                    throw std::invalid_argument("Duplicate entry found.");
            }

            col_indices_.push_back(entry.col);
            values_.push_back(entry.value);

            row_ptr_[entry.row + 1]++;
        }

        for (size_t i = 1; i <= rows_count_; i++)
            row_ptr_[i] += row_ptr_[i-1];
    }

    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptr_;
    std::vector<T> values_;

    size_t rows_count_;
    size_t cols_count_;
};