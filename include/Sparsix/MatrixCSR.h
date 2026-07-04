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

    MatrixCSR();

    MatrixCSR(size_t rows_count, size_t cols_count);

    MatrixCSR(size_t rows_count, size_t cols_count, const std::vector<Triplet<T>> &entries);

    MatrixCSR(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> entries);

    MatrixCSR(const std::vector<std::vector<T>> &matrix, T threshold = T{});

    T at(size_t row, size_t col) const;

    T operator()(size_t row, size_t col) const;

    void reshape(size_t rows_count, size_t cols_count, bool force = false);

    void insert(size_t row, size_t col, const T &value);

    void set(size_t row, size_t col, const T& value);

    void erase(size_t row, size_t col);

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
    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptr_;
    std::vector<T> values_;
    size_t rows_count_;
    size_t cols_count_;
};