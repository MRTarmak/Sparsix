#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class MatrixCOO {
public:
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
            const size_t currentRowSize = matrix[i].size();
            if (currentRowSize != cols_count_) {
                throw std::invalid_argument("Dense matrix must be rectangular.");
            }

            for (size_t j = 0; j < currentRowSize; j++) {
                if (std::abs(matrix[i][j]) > std::abs(threshold)) {
                    rows.push_back(i);
                    cols.push_back(j);
                    values.push_back(matrix[i][j]);
                }
            }
        }
    }

    static MatrixCOO<T> createIdentity(size_t size, T value = T{1}) {
        std::vector<Entry> entries;
        for (size_t i = 0; i < size; i++) {
            entries.push_back({i, i, value});
        }
        return MatrixCOO<T>(size, size, entries);
    }

    static MatrixCOO<T> createDiagonal(size_t size, const std::vector<T> &values) {
        if (values.size() != size) {
            throw std::invalid_argument("Values size must match the specified size.");
        }
        std::vector<Entry> entries;
        for (size_t i = 0; i < size; i++) {
            entries.push_back({i, i, values[i]});
        }
        return MatrixCOO<T>(size, size, entries);
    }

    // TODO createRandom()

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
