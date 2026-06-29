#include <cstddef>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class MatrixCOO {
public:
    struct Entry {
        std::size_t row;
        std::size_t col;
        T value;
    };

    MatrixCOO() : rows(), cols(), values(), rowsNum(0), colsNum(0) {}

    MatrixCOO(std::size_t rowsNum, std::size_t colsNum, const std::vector<Entry> &entries)
        : rows(), cols(), values(), rowsNum(0), colsNum(0) {
        initialize(rowsNum, colsNum, entries.begin(), entries.end());
    }

    MatrixCOO(std::size_t rowsNum, std::size_t colsNum, std::initializer_list<Entry> entries)
        : rows(), cols(), values(), rowsNum(0), colsNum(0) {
        initialize(rowsNum, colsNum, entries.begin(), entries.end());
    }

private:
    template <typename InputIt>
    void initialize(std::size_t rowsCount, std::size_t colsCount, InputIt first, InputIt last) {
        if (rowsCount == 0 || colsCount == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }

        rowsNum = rowsCount;
        colsNum = colsCount;

        std::set<std::pair<std::size_t, std::size_t>> entrySet;

        for (auto it = first; it != last; ++it) {
            const auto &entry = *it;

            if (entry.row >= rowsNum || entry.col >= colsNum) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }

            const std::pair<std::size_t, std::size_t> position{entry.row, entry.col};
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

    std::vector<std::size_t> rows;
    std::vector<std::size_t> cols;
    std::vector<T> values;
    std::size_t rowsNum, colsNum;
};