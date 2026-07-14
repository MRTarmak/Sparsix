#include <vector>
#include <stdexcept>

#include <gtest/gtest.h>

#include <Sparsix/Algorithms/Arithmetic.h>
#include <Sparsix/Algorithms/Multiplication.h>

using namespace sparsix;

namespace {
    template <typename MatrixType>
    void expect_dense_matrix_eq(const MatrixType &matrix, const std::vector<std::vector<typename MatrixType::value_type>> &expected) {
        ASSERT_EQ(matrix.rows_count(), expected.size());
        ASSERT_EQ(matrix.cols_count(), expected.empty() ? 0U : expected.front().size());

        for (size_t row = 0; row < expected.size(); ++row) {
            ASSERT_EQ(expected[row].size(), matrix.cols_count());
            for (size_t col = 0; col < expected[row].size(); ++col) {
                EXPECT_EQ(matrix(row, col), expected[row][col]) << "Mismatch at (" << row << ", " << col << ")";
            }
        }
    }
}

TEST(MatrixArithmeticTest, AdditionAndSubtractionAcrossFormats) {
    const MatrixCOO<int> lhs(3, 3, {{0, 0, 1}, {0, 1, -1}, {2, 1, 2}});
    const MatrixCSR<int> rhs(3, 3, {{0, 1, 1}, {1, 2, 3}, {0, 0, 2}});

    const auto sum = lhs + rhs;
    const auto difference = rhs - lhs;

    const std::vector<std::vector<int>> expected_sum{{3, 0, 0}, {0, 0, 3}, {0, 2, 0}};
    const std::vector<std::vector<int>> expected_difference{{1, 2, 0}, {0, 0, 3}, {0, -2, 0}};

    expect_dense_matrix_eq(sum, expected_sum);
    expect_dense_matrix_eq(difference, expected_difference);
}

TEST(MatrixArithmeticTest, ScalarMultiplyAndDivideKnownMatrix) {
    const MatrixCSC<int> matrix(2, 3, {{0, 0, 2}, {1, 1, -4}, {1, 2, 6}});

    const auto left_scaled = 3 * matrix;
    const auto right_scaled = matrix * 3;
    const auto divided = right_scaled / 3;

    const std::vector<std::vector<int>> expected_scaled{{6, 0, 0}, {0, -12, 18}};

    expect_dense_matrix_eq(left_scaled, expected_scaled);
    expect_dense_matrix_eq(right_scaled, expected_scaled);
    expect_dense_matrix_eq(divided, std::vector<std::vector<int>>{{2, 0, 0}, {0, -4, 6}});

    EXPECT_THROW(matrix / 0, std::runtime_error);
}

TEST(MatrixArithmeticTest, CwiseMultiplyAndDivideKnownMatrix) {
    const MatrixCOO<int> lhs(3, 3, {{0, 0, 2}, {1, 1, 4}, {2, 2, 8}});
    const MatrixCOO<int> rhs(3, 3, {{0, 0, 1}, {1, 1, 2}, {2, 0, 3}, {2, 2, 4}});

    const auto product = sparsix::cwise_multiply(lhs, rhs);
    const auto quotient = sparsix::cwise_divide(lhs, rhs);

    expect_dense_matrix_eq(product, std::vector<std::vector<int>>{{2, 0, 0}, {0, 8, 0}, {0, 0, 32}});
    expect_dense_matrix_eq(quotient, std::vector<std::vector<int>>{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}});
}

TEST(MatrixArithmeticTest, AdditionAndSubtractionRejectDimensionMismatch) {
    const MatrixCOO<int> lhs(2, 3, {{0, 0, 1}});
    const MatrixCOO<int> rhs(3, 2, {{0, 0, 1}});

    EXPECT_THROW((lhs + rhs), std::invalid_argument);
    EXPECT_THROW((lhs - rhs), std::invalid_argument);
    EXPECT_THROW(sparsix::cwise_multiply(lhs, rhs), std::invalid_argument);
    EXPECT_THROW(sparsix::cwise_divide(lhs, rhs), std::invalid_argument);
}

TEST(MatrixMultiplicationTest, MatrixVectorProductKnownMatrix) {
    const MatrixCSR<int> matrix(3, 3, {{0, 0, 1}, {0, 2, 2}, {1, 1, 3}, {2, 0, 4}, {2, 2, 5}});
    const std::vector<int> vector{1, 2, 3};

    const auto result = matrix * vector;

    EXPECT_EQ(result.size(), 3U);
    EXPECT_EQ(result[0], 7);
    EXPECT_EQ(result[1], 6);
    EXPECT_EQ(result[2], 19);
}

TEST(MatrixMultiplicationTest, MatrixMatrixProductKnownMatrix) {
    const MatrixCSR<int> lhs(2, 3, {{0, 0, 1}, {0, 2, 2}, {1, 1, 3}});
    const MatrixCSC<int> rhs(3, 2, {{0, 0, 4}, {1, 1, 5}, {2, 0, 6}});

    const auto result = lhs * rhs;

    expect_dense_matrix_eq(result, std::vector<std::vector<int>>{{16, 0}, {0, 15}});
}

TEST(MatrixMultiplicationTest, RejectsDimensionMismatch) {
    const MatrixCSR<int> lhs(2, 3, {{0, 0, 1}});
    const MatrixCSC<int> rhs(4, 2, {{0, 0, 1}});
    const std::vector<int> vector{1, 2};

    EXPECT_THROW((lhs * rhs), std::invalid_argument);
    EXPECT_THROW((lhs * vector), std::invalid_argument);
}
