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

    MatrixCOO() : rows(), cols(), values(), rowsNum(0), colsNum(0) {}

    MatrixCOO(size_t rowsNum, size_t colsNum, const std::vector<Entry> &entries)
        : rows(), cols(), values(), rowsNum(0), colsNum(0) {
        initialize(rowsNum, colsNum, entries.begin(), entries.end());
    }

    MatrixCOO(size_t rowsNum, size_t colsNum, std::initializer_list<Entry> entries)
        : rows(), cols(), values(), rowsNum(0), colsNum(0) {
        initialize(rowsNum, colsNum, entries.begin(), entries.end());
    }

    MatrixCOO(const std::vector<std::vector<T>> &matrix, T threshold = T{}) {
        rowsNum = matrix.size();
        colsNum = rowsNum > 0 ? matrix.front().size() : 0;

        for (size_t i = 0; i < rowsNum; i++) {
            const size_t currentRowSize = matrix[i].size();
            if (currentRowSize != colsNum) {
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

private:
    template <typename InputIt>
    void initialize(size_t rowsCount, size_t colsCount, InputIt first, InputIt last) {
        if (rowsCount == 0 || colsCount == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        rowsNum = rowsCount;
        colsNum = colsCount;

        std::set<std::pair<size_t, size_t>> entrySet;

        for (auto it = first; it != last; it++) {
            const auto &entry = *it;

            if (entry.row >= rowsNum || entry.col >= colsNum) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }

            const std::pair<size_t, size_t> position{entry.row, entry.col};
            if (entrySet.find(position) != entrySet.end()) {
                throw std::invalid_argument("Duplicate entry found.");
            }
            entrySet.insert(position);

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
    size_t rowsNum, colsNum;
};