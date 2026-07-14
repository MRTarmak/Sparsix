# Sparsix

Sparsix is a header-only C++20 library for sparse matrices in COO, CSR and CSC
formats. It provides arithmetic, transposition, matrix products, utilities,
iterative solvers, Matrix Market I/O and a CSV-like triplet format.

## Build

```bash
cmake --preset default
cmake --build --preset default
ctest --test-dir build/default --output-on-failure
```

Use the umbrella header in a C++20 project:

```cpp
#include <Sparsix.h>

using namespace sparsix;
```

Link the interface CMake target when adding Sparsix as a subdirectory:

```cmake
target_link_libraries(my_app PRIVATE Sparsix::Sparsix)
```

## Creating and converting matrices

```cpp
#include <Sparsix.h>

using namespace sparsix;

MatrixCOO<double> coo(3, 3, {
    {0, 0, 4.0}, {0, 1, 1.0}, {1, 1, 3.0}, {2, 2, 2.0}
});

auto csr = toCSR(coo);                 // efficient row-oriented storage
auto csc = toCSC(csr);                 // efficient column-oriented storage
auto transposed = sparsix::transpose(csr);
auto scaled = 2.0 * csr;
```

Stored values must be non-zero. Use `insert`, `set` and `erase` to mutate a
matrix. `MatrixCOO` accepts unsorted insertion and can be normalized with
`sort()`; CSR and CSC maintain their respective major-order indexing.

## Products and utilities

```cpp
std::vector<double> x{1.0, 2.0, 3.0};
std::vector<double> y = csr * x;
auto product = csr * csc;

double norm = sparsix::frobenius_norm(csr);
bool symmetric = sparsix::is_symmetric(csr);
```

`+`, `-`, `*` and `/` support sparse arithmetic and scalar operations.
`sparsix::cwise_multiply` and `sparsix::cwise_divide` perform element-wise
operations.

## Iterative solvers

CG requires a symmetric positive-definite CSR matrix. BiCGSTAB supports a
general square CSR matrix.

```cpp
MatrixCSR<double> A(2, 2, {
    {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0}
});
std::vector<double> b{6.0, 7.0};

auto result = sparsix::cg(A, b);
if (result.converged) {
    // result.x is the approximate solution
}
```

See [examples/solvers.cpp](examples/solvers.cpp) for both solvers and result
handling.

## Files and debugging output

```cpp
MatrixCOO<std::complex<double>> matrix(2, 2, {
    {0, 1, {2.0, -1.0}}, {1, 0, {2.0, 1.0}}
});

sparsix::write_matrix_market(matrix, "matrix.mtx");
auto restored = sparsix::read_matrix_market<MatrixCSR<std::complex<double>>>("matrix.mtx");

sparsix::write_triplets_csv(matrix, "matrix.csv");
auto csv_matrix = sparsix::read_triplets_csv<MatrixCOO<std::complex<double>>>("matrix.csv");

std::cout << restored << '\n';
```

Matrix Market uses standard one-based `coordinate` entries and supports real,
integer, complex, pattern, symmetric, skew-symmetric and Hermitian input.
The CSV-like format uses zero-based coordinates: its first line is
`rows,cols,nnz`, followed by `row,col,value` (or `row,col,real,imag` for
complex values).

## Examples

The build creates these executables when `SPARSIX_BUILD_EXAMPLES=ON`:

- `sparsix_basics` - construction, conversion, arithmetic and inspection;
- `sparsix_solvers` - CG and BiCGSTAB;
- `sparsix_io` - Matrix Market, CSV and complex values.

API documentation is written in Doxygen format in public headers. Generate
reference documentation with any Doxygen configuration that includes
`include/`.
