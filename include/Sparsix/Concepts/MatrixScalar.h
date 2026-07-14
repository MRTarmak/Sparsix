#pragma once

#include <concepts>
#include <complex>
#include <type_traits>
#include <utility>

/** @brief Type trait identifying std::complex specializations. */
template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

/** @brief True when T is a std::complex specialization. */
template <typename T>
inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

/** @brief True when T can be used as a matrix value. */
template <typename T>
inline constexpr bool is_matrix_scalar_v = std::is_arithmetic_v<T> || is_std_complex_v<T>;

/** @brief Constraint for arithmetic and complex matrix scalar types. */
template <typename T>
concept MatrixScalar = is_matrix_scalar_v<T>;

/** @brief Real scalar type associated with T. */
template <typename T>
using real_type = decltype(std::abs(std::declval<T>()));
