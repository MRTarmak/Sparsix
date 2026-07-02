#include <algorithm>
#include <cstddef>
#include <cmath>
#include <complex>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Sparsix/Utils/Randomizer.h>

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

    MatrixCOO() : rows_(), cols_(), values_(), rows_count_(0), cols_count_(0) {}

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
                    rows_.push_back(i);
                    cols_.push_back(j);
                    values_.push_back(matrix[i][j]);
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

        std::vector<Entry> entries;
        entries.reserve(non_zero_count);

        if (non_zero_count == 0) {
            return MatrixCOO<T>(rows_count, cols_count, entries);
        }

        std::vector<size_t> positions(total_count);
        std::iota(positions.begin(), positions.end(), size_t{0});

        auto &generator = Randomizer::generator();
        std::shuffle(positions.begin(), positions.end(), generator);

        auto value_is_zero = [](const T &value) {
            if constexpr (is_std_complex_v<T>) {
                return value.real() == typename T::value_type{} && value.imag() == typename T::value_type{};
            } else {
                return value == T{};
            }
        };

        auto make_matrix = [&](auto make_value) -> MatrixCOO<T> {
            for (size_t i = 0; i < non_zero_count; i++) {
                const size_t linear_index = positions[i];
                const size_t row = linear_index / cols_count;
                const size_t col = linear_index % cols_count;
                T value = make_value();

                while (value_is_zero(value)) {
                    value = make_value();
                }

                entries.push_back({row, col, value});
            }

            return MatrixCOO<T>(rows_count, cols_count, entries);
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

    T get(size_t row, size_t col) const {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
        for (size_t i = 0; i < values_.size(); i++) {
            if (rows_[i] == row && cols_[i] == col) {
                return values_[i];
            }
        }
        return T{};
    }

    T operator()(size_t row, size_t col) const {
        return get(row, col);
    }

    T &at(size_t row, size_t col) {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
        for (size_t i = 0; i < values_.size(); i++) {
            if (rows_[i] == row && cols_[i] == col) {
                return values_[i];
            }
        }
        throw std::out_of_range("Element at specified position is not stored in COO format.");
    }

    T &operator()(size_t row, size_t col) {
        return at(row, col);
    }

    void insert(size_t row, size_t col, const T &value) {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
        if (value == T{}) {
            throw std::invalid_argument("Cannot insert default (zero) value into COO matrix.");
        }
        for (size_t i = 0; i < values_.size(); i++) {
            if (rows_[i] == row && cols_[i] == col) {
                values_[i] = value;
                return;
            }
        }

        rows_.push_back(row);
        cols_.push_back(col);
        values_.push_back(value);
    }

    void remove(size_t row, size_t col) {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }
        for (size_t i = 0; i < values_.size(); i++) {
            if (rows_[i] == row && cols_[i] == col) {
                rows_.erase(rows_.begin() + i);
                cols_.erase(cols_.begin() + i);
                values_.erase(values_.begin() + i);
                return;
            }
        }
    }

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

            rows_.push_back(entry.row);
            cols_.push_back(entry.col);
            values_.push_back(entry.value);
        }
    }

    std::vector<size_t> rows_;
    std::vector<size_t> cols_;
    std::vector<T> values_;
    size_t rows_count_;
    size_t cols_count_;
};
