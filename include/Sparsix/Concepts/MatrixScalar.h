#pragma once

#include <concepts>
#include <complex>

template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

template <typename T>
inline constexpr bool is_matrix_scalar_v = std::is_arithmetic_v<T> || is_std_complex_v<T>;

template <typename T>
concept MatrixScalar = is_matrix_scalar_v<T>;