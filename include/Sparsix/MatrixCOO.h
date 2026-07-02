#include <cstddef>
#include <cmath>
#include <complex>
#include <concepts>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

template <typename T>
inline constexpr bool is_matrix_scalar_v = std::is_arithmetic_v<T> || is_std_complex_v<T>;

template <typename T>
concept MatrixScalar = is_matrix_scalar_v<T>;

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
class MatrixCOO {
public:
    static_assert(is_matrix_scalar_v<T>, "MatrixCOO requires an arithmetic or std::complex value type.");

    struct Entry {
        size_t row;
        size_t col;
        T value;
    };

    MatrixCOO() : rows(), cols(), values(), rows_count_(0), cols_count_(0) {}

    MatrixCOO(size_t rows_count, size_t cols_count, const std::vector<Entry> &entries) {
        initialize(rows_count, cols_count, entries.begin(), entries.end());
    }

    MatrixCOO(size_t rows_count, size_t cols_count, std::initializer_list<Entry> entries) {
        initialize(rows_count, cols_count, entries.begin(), entries.end());
    }

    MatrixCOO(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
        rows_count_ = matrix.size();
        cols_count_ = rows_count_ > 0 ? matrix.front().size() : 0;

        for (size_t i = 0; i < rows_count_; i++) {
            const size_t current_row_size = matrix[i].size();
            if (current_row_size != cols_count_) {
                throw std::invalid_argument("Dense matrix must be rectangular.");
            }

            for (size_t j = 0; j < current_row_size; j++) {
                if (std::abs(matrix[i][j]) > std::abs(threshold)) {
                    rows.push_back(i);
                    cols.push_back(j);
                    values.push_back(matrix[i][j]);
                }
            }
        }
    }

    static MatrixCOO<T> create_identity(size_t size, T value = T{1}) {
        std::vector<Entry> entries;
        for (size_t i = 0; i < size; i++) {
            entries.push_back({i, i, value});
        }
        return MatrixCOO<T>(size, size, entries);
    }

    static MatrixCOO<T> create_diagonal(size_t size, const std::vector<T> &values) {
        if (values.size() != size) {
            throw std::invalid_argument("Values size must match the specified size.");
        }
        std::vector<Entry> entries;
        for (size_t i = 0; i < size; i++) {
            entries.push_back({i, i, values[i]});
        }
        return MatrixCOO<T>(size, size, entries);
    }

    // TODO create_random()

    // TODO overload operator()

    size_t rows_count() const {
        return rows_count_;
    }

    size_t cols_count() const {
        return cols_count_;
    }

    size_t non_zero_count() const {
        return values.size();
    }

private:
    template <typename InputIt>
    void initialize(size_t rows_count, size_t cols_count, InputIt first, InputIt last) {
        if (rows_count == 0 || cols_count == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        rows_count_ = rows_count;
        cols_count_ = cols_count;

        std::set<std::pair<size_t, size_t>> entry_set;

        for (auto it = first; it != last; it++) {
            const auto &entry = *it;

            if (entry.row >= rows_count_ || entry.col >= cols_count_) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }

            const std::pair<size_t, size_t> position{entry.row, entry.col};
            if (entry_set.find(position) != entry_set.end()) {
                throw std::invalid_argument("Duplicate entry found.");
            }
            entry_set.insert(position);

            if (entry.value == T{}) {
                continue;
            }

            rows.push_back(entry.row);
            cols.push_back(entry.col);
            values.push_back(entry.value);
        }
    }

    std::vector<size_t> rows;
    std::vector<size_t> cols;
    std::vector<T> values;
    size_t rows_count_;
    size_t cols_count_;
};
