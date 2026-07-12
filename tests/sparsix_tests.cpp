#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <Sparsix.h>

namespace {
    template <typename MatrixType>
    void expect_dense_matrix_eq(const MatrixType &matrix, const std::vector<std::vector<typename MatrixType::value_type>> &expected) {
        ASSERT_EQ(matrix.rows_count(), expected.size());
        ASSERT_EQ(matrix.cols_count(), expected.empty() ? 0U : expected.front().size());

        for (size_t row = 0; row < expected.size(); row++) {
            ASSERT_EQ(expected[row].size(), matrix.cols_count());
            for (size_t col = 0; col < expected[row].size(); col++) {
                EXPECT_EQ(matrix(row, col), expected[row][col]) << "Mismatch at (" << row << ", " << col << ")";
            }
        }
    }
}

TEST(MatrixCOOTest, DenseConstructionAndLookup) {
    const std::vector<std::vector<int>> dense{{0, 1, 0}, {2, 0, 3}, {0, 0, 4}};
    const MatrixCOO<int> matrix(dense);

    expect_dense_matrix_eq(matrix, dense);
    EXPECT_EQ(matrix.non_zero_count(), 4U);
    EXPECT_TRUE(matrix.sorted());
    EXPECT_TRUE(matrix.contains(1, 2));
    EXPECT_FALSE(matrix.contains(0, 0));
}

TEST(MatrixCOOTest, InsertSetEraseAndExceptions) {
    MatrixCOO<int> matrix(3, 3);

    EXPECT_THROW(matrix.insert(0, 0, 0), std::invalid_argument);
    matrix.insert(2, 1, 7);

    EXPECT_TRUE(matrix.contains(2, 1));
    EXPECT_EQ(matrix.at(2, 1), 7);
    EXPECT_FALSE(matrix.sorted());

    EXPECT_THROW(matrix.insert(2, 1, 8), std::invalid_argument);
    EXPECT_THROW(matrix.set(1, 1, 0), std::invalid_argument);
    EXPECT_THROW(matrix.set(1, 1, 5), std::out_of_range);

    matrix.set(2, 1, 11);
    EXPECT_EQ(matrix(2, 1), 11);

    matrix.erase(2, 1);
    EXPECT_FALSE(matrix.contains(2, 1));
    EXPECT_EQ(matrix.non_zero_count(), 0U);
}

TEST(MatrixCOOTest, SortOrdersByRowAndColumn) {
    MatrixCOO<int> matrix(3, 3);
    matrix.insert(2, 0, 5);
    matrix.insert(0, 2, 7);
    matrix.insert(1, 1, 3);

    matrix.sort();

    EXPECT_TRUE(matrix.sorted());
    ASSERT_EQ(matrix.rows().size(), 3U);
    EXPECT_EQ(matrix.rows()[0], 0U);
    EXPECT_EQ(matrix.cols()[0], 2U);
    EXPECT_EQ(matrix.rows()[1], 1U);
    EXPECT_EQ(matrix.cols()[1], 1U);
    EXPECT_EQ(matrix.rows()[2], 2U);
    EXPECT_EQ(matrix.cols()[2], 0U);

    matrix.sort(MajorOrder::ColumnOrder);
    EXPECT_FALSE(matrix.sorted());
    EXPECT_EQ(matrix.cols()[0], 0U);
    EXPECT_EQ(matrix.cols()[1], 1U);
    EXPECT_EQ(matrix.cols()[2], 2U);
}

TEST(MatrixCOOTest, ReshapeTruncatesWhenForced) {
    MatrixCOO<int> matrix(3, 3, {{0, 0, 1}, {1, 1, 2}, {2, 2, 3}});

    EXPECT_THROW(matrix.reshape(2, 2), std::invalid_argument);
    matrix.reshape(2, 2, true);

    EXPECT_EQ(matrix.rows_count(), 2U);
    EXPECT_EQ(matrix.cols_count(), 2U);
    EXPECT_EQ(matrix.non_zero_count(), 2U);
    EXPECT_EQ(matrix.at(0, 0), 1);
    EXPECT_EQ(matrix.at(1, 1), 2);
    EXPECT_EQ(matrix(0, 1), 0);
}

TEST(MatrixCOOTest, KnownIdentityAndDiagonalMatrices) {
    const auto identity = MatrixCOO<int>::create_identity(4);
    const std::vector<std::vector<int>> expected_identity{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    expect_dense_matrix_eq(identity, expected_identity);

    const auto diagonal = MatrixCOO<int>::create_diagonal(3, std::vector<int>{4, 5, 6});
    const std::vector<std::vector<int>> expected_diagonal{{4, 0, 0}, {0, 5, 0}, {0, 0, 6}};
    expect_dense_matrix_eq(diagonal, expected_diagonal);
}

TEST(MatrixCOOTest, RandomMatrixHasExpectedShapeAndValueRange) {
    const auto matrix = MatrixCOO<int>::create_random(2, 3, 1.0, 1, 3);

    EXPECT_EQ(matrix.rows_count(), 2U);
    EXPECT_EQ(matrix.cols_count(), 3U);
    EXPECT_EQ(matrix.non_zero_count(), 6U);

    for (size_t row = 0; row < matrix.rows_count(); row++) {
        for (size_t col = 0; col < matrix.cols_count(); col++) {
            const int value = matrix(row, col);
            EXPECT_GE(value, 1);
            EXPECT_LE(value, 3);
        }
    }
}

TEST(MatrixConversionsTest, COOToCSRAndCSCRoundTripKnownMatrix) {
    const std::vector<std::vector<int>> dense{{0, 2, 0, 0}, {5, 0, 0, 7}, {0, 0, 3, 0}};
    const MatrixCOO<int> coo(dense);

    const auto csr = toCSR(coo);
    const auto csc = toCSC(coo);

    expect_dense_matrix_eq(csr, dense);
    expect_dense_matrix_eq(csc, dense);

    const auto coo_from_csr = toCOO(csr);
    const auto coo_from_csc = toCOO(csc);

    expect_dense_matrix_eq(coo_from_csr, dense);
    expect_dense_matrix_eq(coo_from_csc, dense);

    EXPECT_EQ(csr.non_zero_count(), 4U);
    EXPECT_EQ(csc.non_zero_count(), 4U);
}

TEST(MatrixCSCTest, ContainerAccessorsAndMutation) {
    MatrixCSC<int> matrix(3, 3, {{0, 0, 1}, {1, 2, 4}});

    EXPECT_EQ(matrix.rows_count(), 3U);
    EXPECT_EQ(matrix.cols_count(), 3U);
    EXPECT_EQ(matrix.non_zero_count(), 2U);
    EXPECT_TRUE(matrix.contains(1, 2));
    EXPECT_EQ(matrix.at(1, 2), 4);

    matrix.insert(2, 1, 6);
    EXPECT_TRUE(matrix.contains(2, 1));
    EXPECT_EQ(matrix.at(2, 1), 6);

    matrix.set(2, 1, 8);
    EXPECT_EQ(matrix.at(2, 1), 8);

    matrix.erase(0, 0);
    EXPECT_FALSE(matrix.contains(0, 0));

    matrix.clear();
    EXPECT_EQ(matrix.non_zero_count(), 0U);
}

TEST(MatrixCSRTest, ContainerAccessorsAndMutation) {
    MatrixCSR<int> matrix(3, 3, {{0, 1, 2}, {2, 2, 9}});

    EXPECT_EQ(matrix.rows_count(), 3U);
    EXPECT_EQ(matrix.cols_count(), 3U);
    EXPECT_EQ(matrix.non_zero_count(), 2U);
    EXPECT_TRUE(matrix.contains(2, 2));
    EXPECT_EQ(matrix.at(2, 2), 9);

    matrix.insert(1, 0, 5);
    EXPECT_TRUE(matrix.contains(1, 0));
    EXPECT_EQ(matrix.at(1, 0), 5);

    matrix.set(1, 0, 11);
    EXPECT_EQ(matrix.at(1, 0), 11);

    matrix.erase(0, 1);
    EXPECT_FALSE(matrix.contains(0, 1));

    matrix.clear();
    EXPECT_EQ(matrix.non_zero_count(), 0U);
}

TEST(MatrixConversionTest, CSRAndCSCCrossConversionsMatchOriginalValues) {
    const MatrixCSR<int> csr(3, 4, {{0, 1, 3}, {1, 3, 5}, {2, 0, 7}});

    const auto csc = toCSC(csr);
    const auto csr_roundtrip = toCSR(csc);

    EXPECT_EQ(csc.at(0, 1), 3);
    EXPECT_EQ(csc.at(1, 3), 5);
    EXPECT_EQ(csc.at(2, 0), 7);

    EXPECT_EQ(csr_roundtrip.at(0, 1), 3);
    EXPECT_EQ(csr_roundtrip.at(1, 3), 5);
    EXPECT_EQ(csr_roundtrip.at(2, 0), 7);
}


TEST(MatrixArithmetic, MatricesAddition) {
    const MatrixCOO<int> m1(3, 3, {{0, 0, 1}, {0, 1, -1}, {2, 1, 2}});
    const MatrixCSR<int> m2(3, 3, {{0, 1, 1}, {1, 2, 3}, {0, 0, 2}});

    const auto result = m1 + m2;

    EXPECT_EQ(result.non_zero_count(), 3);
    EXPECT_EQ(result(0, 0), 3);
    EXPECT_EQ(result(1, 2), 3);
    EXPECT_EQ(result(2, 1), 2);
    EXPECT_FALSE(result.contains(0, 1));
}