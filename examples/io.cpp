#include <complex>
#include <filesystem>
#include <iostream>

#include <Sparsix.h>

int main() {
    using Complex = std::complex<double>;
    const MatrixCOO<Complex> matrix(2, 3, {
        {0, 2, {1.5, -2.0}}, {1, 0, {-3.0, 0.25}}
    });
    const auto output_dir = std::filesystem::current_path();
    const auto matrix_market_path = output_dir / "complex_matrix.mtx";
    const auto csv_path = output_dir / "complex_matrix.csv";

    sparsix::write_matrix_market(matrix, matrix_market_path);
    sparsix::write_triplets_csv(matrix, csv_path);

    const auto from_mtx = sparsix::read_matrix_market<MatrixCSR<Complex>>(matrix_market_path);
    const auto from_csv = sparsix::read_triplets_csv<MatrixCSC<Complex>>(csv_path);

    std::cout << "Written: " << matrix_market_path << " and " << csv_path << '\n';
    std::cout << "Matrix Market read: " << from_mtx << '\n';
    std::cout << "CSV read: " << from_csv << '\n';
}
