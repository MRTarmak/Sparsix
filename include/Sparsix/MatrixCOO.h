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
#include <Sparsix/Core/Triplet.h>
#include <Sparsix/Utils/Randomizer.h>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <MatrixScalar T>
#else
template <typename T>
#endif
class MatrixCOO {
public:
    static_assert(is_matrix_scalar_v<T>, "MatrixCOO requires an arithmetic or std::complex value type.");

    explicit MatrixCOO() : rows_(), cols_(), values_(), rows_count_(0), cols_count_(0), sorted_(true) {}

    explicit MatrixCOO(size_t rows_count, size_t cols_count)
        : rows_(), cols_(), values_(), rows_count_(rows_count), cols_count_(cols_count), sorted_(true) {}

    explicit MatrixCOO(size_t rows_count, size_t cols_count, const std::vector<Triplet<T>> &entries) {
        initialize(rows_count, cols_count, entries.begin(), entries.end());
    }

    explicit MatrixCOO(size_t rows_count, size_t cols_count, std::initializer_list<Triplet<T>> entries) {
        initialize(rows_count, cols_count, entries.begin(), entries.end());
    }

    explicit MatrixCOO(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
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

        sorted_ = false;
    }

    static MatrixCOO<T> create_identity(size_t size, T value = T{1}) {
        std::vector<Triplet<T>> entries;
        entries.reserve(size);
        for (size_t i = 0; i < size; i++) {
            entries.push_back({i, i, value});
        }
        return MatrixCOO<T>(size, size, entries);
    }

    static MatrixCOO<T> create_diagonal(size_t size, const std::vector<T> &values) {
        if (values.size() != size) {
            throw std::invalid_argument("Values size must match the specified size.");
        }

        std::vector<Triplet<T>> entries;
        entries.reserve(size);
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

        std::vector<Triplet<T>> entries;
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

        for (size_t i = 0; i < order.size(); i++) {
            new_rows.push_back(rows_[order[i]]);
            new_cols.push_back(cols_[order[i]]);
            new_values.push_back(values_[order[i]]);
        }

        rows_ = std::move(new_rows);
        cols_ = std::move(new_cols);
        values_ = std::move(new_values);

        sorted_ = true;
    };

    T at(size_t row, size_t col) const {
        auto index = find_index(row, col);
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
                std::vector<Triplet<T>> entries;
                entries.reserve(values_.size());
                for (size_t i = 0; i < values_.size(); i++) {
                    if (rows_[i] < rows_count && cols_[i] < cols_count) {
                        entries.emplace_back(rows_[i], cols_[i], values_[i]);
                    }
                }
                rows_.clear();
                cols_.clear();
                values_.clear();
                initialize(rows_count, cols_count, entries.begin(), entries.end());
            }
        } else {
            rows_count_ = rows_count;
            cols_count_ = cols_count;
        }
    }

    void insert(size_t row, size_t col, const T &value) {
        if (value == T{}) {
            throw std::invalid_argument("Cannot insert default (zero) value into COO matrix.");
        }

        auto index = find_index(row, col);
        if (index) {
            throw std::invalid_argument("Element already exists in COO matrix. \
                                                            Use set() to modify it.");
        }

        rows_.push_back(row);
        cols_.push_back(col);
        values_.push_back(value);

        sorted_ = false;
    }

    void set(size_t row, size_t col, const T& value) {
        if (value == T{}) {
            throw std::invalid_argument("Cannot store default (zero) value in COO matrix. \
                                                                Use erase() to remove it.");
        }
        
        auto index = find_index(row, col);
        if (index) {
            values_[*index] = value;
            return;
        }

        throw std::out_of_range("Element does not exist in COO matrix. \
                                                Use insert() to add it.");
    }

    bool contains(size_t row, size_t col) {
        if (find_index(row, col))
            return true;
        return false;
    }

    void erase(size_t row, size_t col) {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }

        auto index = find_index(row, col);
        if (index) {
            rows_.erase(rows_.begin() + *index);
            cols_.erase(cols_.begin() + *index);
            values_.erase(values_.begin() + *index);
            return;
        }
    }

    void reserve(size_t nnz) {
        rows_.reserve(nnz);
        cols_.reserve(nnz);
        values_.reserve(nnz);
    }

    void clear() {
        rows_.clear();
        cols_.clear();
        values_.clear();

        sorted_ = true;
    }

    size_t rows_count() const {
        return rows_count_;
    }

    size_t cols_count() const {
        return cols_count_;
    }

    bool sorted() const {
        return sorted_;
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

        rows_.reserve(count);
        cols_.reserve(count);
        values_.reserve(count);

        std::set<std::pair<size_t, size_t>> entry_set;

        for (auto it = first; it != last; it++) {
            const auto &entry = *it;

            if (entry.row >= rows_count_ || entry.col >= cols_count_) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }

            const std::pair<size_t, size_t> position{entry.row, entry.col};

            auto [entry_it, inserted] = entry_set.insert(position);
            if (!inserted) {
                throw std::invalid_argument("Duplicate entry found.");
            }

            if (entry.value == T{}) {
                continue;
            }

            rows_.push_back(entry.row);
            cols_.push_back(entry.col);
            values_.push_back(entry.value);
        }

        sorted_ = false;
    }

    std::optional<size_t> find_index(size_t row, size_t col) const {
        if (row >= rows_count_ || col >= cols_count_) {
            throw std::out_of_range("Matrix indices are out of bounds.");
        }

        if (!sorted_) {
            for (size_t i = 0; i < values_.size(); i++) {
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
