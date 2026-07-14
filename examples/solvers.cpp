#include <iostream>
#include <optional>
#include <vector>

#include <Sparsix.h>

namespace {
    void print_result(const char *name, const sparsix::SolverResult<double> &result) {
        std::cout << name << ": converged=" << std::boolalpha << result.converged
                  << ", iterations=" << result.iterations
                  << ", residual=" << result.residual << "\nx =";
        for (const double value : result.x)
            std::cout << ' ' << value;
        std::cout << "\n\n";
    }
}

int main() {
    const MatrixCSR<double> spd(3, 3, {
        {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0},
        {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 2.0}
    });
    print_result("CG", sparsix::cg(spd, std::vector<double>{6.0, 10.0, 8.0}, std::optional<std::vector<double>>{}, 100, 1e-10));

    const MatrixCSR<double> nonsymmetric(2, 2, {
        {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 3.0}
    });
    print_result("BiCGSTAB", sparsix::bicgstab(nonsymmetric, std::vector<double>{6.0, 8.0}));
}
