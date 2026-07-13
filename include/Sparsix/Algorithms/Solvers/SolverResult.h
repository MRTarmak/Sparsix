namespace sparsix {
    template<typename T>
    struct SolverResult
    {
        std::vector<T> x;
        size_t iterations;
        double residual;
        bool converged;
    };
}
