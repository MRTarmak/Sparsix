#include <cstddef>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class MatrixCOO {
    struct Entry {
        std::size_t row;
        std::size_t col;
        T value;
    };

    std::vector<std::size_t> rows;
    std::vector<std::size_t> cols;
    std::vector<T> values;
    std::size_t rowsNum, colsNum;
    
public:
    MatrixCOO() : rows(), cols(), values(), rowsNum(0), colsNum(0) {}

    MatrixCOO(std::size_t rowsNum, std::size_t colsNum, const std::vector<Entry> &entries) {
        if (rowsNum == 0 || colsNum == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }
        if (entries.empty()) {
            throw std::invalid_argument("Entries vector cannot be empty.");
        }
        if (entries.size() > rowsNum * colsNum) {
            throw std::invalid_argument("Number of entries exceeds matrix dimensions.");
        }

        this->rowsNum = rowsNum;
        this->colsNum = colsNum;

        std::set<std::pair<std::size_t, std::size_t>> entrySet; // To check for duplicates

        for (const auto &entry : entries) {
            if (entry.row >= rowsNum || entry.col >= colsNum) {
                throw std::out_of_range("Entry position is out of matrix bounds.");
            }
            if (entrySet.find({entry.row, entry.col}) != entrySet.end()) {
                throw std::invalid_argument("Duplicate entry found.");
            }
            if (entry.value == T{}) {
                continue; // Skip zero entries
            }
            rows.push_back(entry.row);
            cols.push_back(entry.col);
            values.push_back(entry.value);
            entrySet.insert({entry.row, entry.col});
        }
    }
};