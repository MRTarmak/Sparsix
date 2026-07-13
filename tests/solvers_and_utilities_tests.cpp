#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include <Sparsix.h>

namespace {
    template <typename T>
    void expect_vector_near(const std::vector<T> &actual, const std::vector<T> &expected, double tolerance = 1e-10) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); i++)
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
    }
}

TEST(VectorAlgorithmsTest, DotUsesConjugateAndNorm2ComputesEuclideanLength) {
    const std::vector<std::complex<double>> lhs{{1.0, 2.0}, {3.0, -1.0}};
    const std::vector<std::complex<double>> rhs{{2.0, -1.0}, {-1.0, 4.0}};

    EXPECT_EQ(sparsix::dot(lhs, rhs), std::complex<double>(-7.0, 6.0));
    EXPECT_NEAR(sparsix::norm2(std::vector<double>{3.0, 4.0}), 5.0, 1e-12);
    EXPECT_EQ(sparsix::conjugate(std::complex<double>(2.0, -3.0)), std::complex<double>(2.0, 3.0));
}

TEST(VectorAlgorithmsTest, AxpyAndScalMutateVectorAndRejectDifferentSizes) {
    const std::vector<double> x{1.0, -2.0, 3.0};
    std::vector<double> y{4.0, 5.0, 6.0};

    sparsix::axpy(2.0, x, y);
    sparsix::scal(y, 0.5);

    expect_vector_near(y, {3.0, 0.5, 6.0});
    EXPECT_THROW(sparsix::dot(x, std::vector<double>{1.0}), std::invalid_argument);
    std::vector<double> short_y{1.0};
    EXPECT_THROW(sparsix::axpy(1.0, x, short_y), std::invalid_argument);
}

TEST(MatrixUtilitiesTest, ComputesSymmetryTraceAndNormsAcrossFormats) {
    const MatrixCSR<double> symmetric(3, 3, {
        {0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 3.0}, {1, 2, 4.0}, {2, 1, 4.0}, {2, 2, -5.0}
    });
    const MatrixCOO<double> nonsymmetric(2, 2, {{0, 1, 1.0}});
    const MatrixCSC<double> rectangular(2, 3, {{0, 0, 1.0}});

    EXPECT_TRUE(sparsix::is_symmetric(symmetric));
    EXPECT_FALSE(sparsix::is_symmetric(nonsymmetric));
    EXPECT_FALSE(sparsix::is_symmetric(rectangular));
    EXPECT_DOUBLE_EQ(sparsix::trace(symmetric), 0.0);
    EXPECT_NEAR(sparsix::frobenius_norm(symmetric), std::sqrt(72.0), 1e-12);
    EXPECT_DOUBLE_EQ(sparsix::one_norm(symmetric), 9.0);
    EXPECT_DOUBLE_EQ(sparsix::infinity_norm(symmetric), 9.0);
}

TEST(CGTest, SolvesSymmetricPositiveDefiniteSystem) {
    const MatrixCSR<double> matrix(3, 3, {
        {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0}, {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 2.0}
    });
    const std::vector<double> expected{1.0, 2.0, -1.0};
    const std::vector<double> rhs = matrix * expected;

    const auto result = sparsix::cg(matrix, rhs, std::optional<std::vector<double>>{}, 10, 1e-12);

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 3U);
    EXPECT_LE(result.residual, 1e-12);
    expect_vector_near(result.x, expected);
}

TEST(CGTest, AcceptsExactInitialGuessWithoutIterations) {
    const MatrixCSR<double> identity(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
    const std::vector<double> rhs{2.0, -4.0};

    const auto result = sparsix::cg(identity, rhs, std::optional<std::vector<double>>(rhs), 5, 1e-12);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0U);
    EXPECT_DOUBLE_EQ(result.residual, 0.0);
    EXPECT_EQ(result.x, rhs);
}

TEST(CGTest, ReportsUnconvergedWhenIterationBudgetIsExhausted) {
    const MatrixCSR<double> matrix(2, 2, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 2.0}});

    const auto result = sparsix::cg(matrix, std::vector<double>{1.0, 0.0}, std::optional<std::vector<double>>{}, 1, 1e-16);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 1U);
    EXPECT_GT(result.residual, 1e-16);
}

TEST(CGTest, RejectsInvalidInput) {
    const MatrixCSR<double> nonsymmetric(2, 2, {{0, 1, 1.0}});
    const MatrixCSR<double> symmetric(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});

    EXPECT_THROW(sparsix::cg(nonsymmetric, std::vector<double>{1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(sparsix::cg(symmetric, std::vector<double>{1.0}), std::invalid_argument);
    EXPECT_THROW(sparsix::cg(symmetric, std::vector<double>{1.0, 2.0}, std::optional<std::vector<double>>(std::vector<double>{0.0})), std::invalid_argument);
}

TEST(BiCGSTABTest, SolvesNonsymmetricSystem) {
    const MatrixCSR<double> matrix(3, 3, {
        {0, 0, 4.0}, {0, 1, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}, {1, 2, -1.0}, {2, 0, 1.0}, {2, 2, 2.0}
    });
    const std::vector<double> expected{1.0, -2.0, 3.0};
    const std::vector<double> rhs = matrix * expected;

    const auto result = sparsix::bicgstab(matrix, rhs, std::optional<std::vector<double>>{}, 10, 1e-12);

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 3U);
    EXPECT_LE(result.residual, 1e-12);
    expect_vector_near(result.x, expected);
}

TEST(BiCGSTABTest, AcceptsExactInitialGuessWithoutIterations) {
    const MatrixCSR<double> matrix(2, 2, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 0, -1.0}, {1, 1, 3.0}});
    const std::vector<double> rhs{4.0, -2.0};

    const auto result = sparsix::bicgstab(matrix, rhs, std::optional<std::vector<double>>(std::vector<double>{2.0, 0.0}), 5, 1e-12);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0U);
    EXPECT_DOUBLE_EQ(result.residual, 0.0);
    EXPECT_EQ(result.x, std::vector<double>({2.0, 0.0}));
}

TEST(BiCGSTABTest, ReportsUnconvergedWhenIterationBudgetIsExhausted) {
    const MatrixCSR<double> matrix(2, 2, {{0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 3.0}});

    const auto result = sparsix::bicgstab(matrix, std::vector<double>{1.0, 0.0}, std::optional<std::vector<double>>{}, 1, 1e-16);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 1U);
    EXPECT_GT(result.residual, 1e-16);
}

TEST(BiCGSTABTest, RejectsInvalidVectorSizes) {
    const MatrixCSR<double> matrix(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});

    EXPECT_THROW(sparsix::bicgstab(matrix, std::vector<double>{1.0}), std::invalid_argument);
    EXPECT_THROW(sparsix::bicgstab(matrix, std::vector<double>{1.0, 2.0}, std::optional<std::vector<double>>(std::vector<double>{0.0})), std::invalid_argument);
}
