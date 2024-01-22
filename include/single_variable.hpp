#pragma once

#include <array>
#include <cassert>
#include <cinttypes>
#include <complex>
#include <concepts>
#include <iostream>
#include <numbers>

namespace Autodiff {

constexpr std::array<std::array<double, 8>, 8> Combination{
    {
        {1, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 0, 0},
        {1, 2, 1, 0, 0, 0, 0, 0},
        {1, 3, 3, 1, 0, 0, 0, 0},
        {1, 4, 6, 4, 1, 0, 0, 0},
        {1, 5, 10, 10, 5, 1, 0, 0},
        {1, 6, 15, 20, 15, 6, 1, 0},
        {1, 7, 21, 35, 35, 21, 7, 1},
    },
};

template <size_t N>
using Value = std::tuple<size_t, std::array<size_t, N>, size_t>;

/*!
 * 変数が一つの自動微分
 **/
template <std::size_t Order, std::floating_point ValType = double>
class SingleVariable {
private:
  std::array<ValType, Order + 1> values{};

public:
  SingleVariable() = default;

  explicit constexpr SingleVariable(ValType value) {
    this->values[0] = value;
    this->values[1] = 1.0;
  }

  explicit constexpr SingleVariable(std::array<ValType, Order + 1> values)
      : values(values) {}

  void constexpr set_value(ValType value, size_t index = 0) {
    assert(index <= Order);
    this->values[index] = value;
  }

  [[nodiscard]] constexpr ValType &operator[](size_t index) {
    return this->values[index];
  }

  [[nodiscard]] constexpr ValType get_value(size_t index) const {
    return this->values[index];
  }

  [[nodiscard]] constexpr ValType operator[](size_t index) const {
    return this->get_value(index);
  }

  [[nodiscard]] constexpr SingleVariable
  operator+(const SingleVariable &rhs) const {
    SingleVariable result{};
    for (std::size_t i = 0; i < Order + 1; ++i) {
      result.values[i] = this->values[i] + rhs.values[i];
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable operator+(const ValType &rhs) const {
    SingleVariable result{*this};
    result.values[0] += rhs;
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  operator+(const ValType &lhs, const SingleVariable &rhs) {
    SingleVariable result{rhs};
    result.values[0] += lhs;
    return result;
  }

  [[nodiscard]] constexpr SingleVariable operator-() const {
    SingleVariable result{};
    for (std::size_t i = 0; i < Order + 1; ++i) {
      result.values[i] = -this->values[i];
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable
  operator-(const SingleVariable &rhs) const {
    SingleVariable result{};
    for (std::size_t i = 0; i < Order + 1; ++i) {
      result.values[i] = this->values[i] - rhs.values[i];
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable operator-(const ValType &rhs) const {
    SingleVariable result{*this};
    result.values[0] -= rhs;
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  operator-(const ValType &lhs, const SingleVariable &rhs) {
    SingleVariable result{-rhs};
    result.values[0] += lhs;
    return result;
  }

  [[nodiscard]] constexpr SingleVariable
  operator*(const SingleVariable &rhs) const {
    SingleVariable result{};
    result.values[0] = this->values[0] * rhs.values[0];
    for (std::size_t n = 1; n < Order + 1; ++n) {
      for (std::size_t i = 0; i <= n; i++) {
        result.values[n] +=
            Combination[n][i] * this->values[i] * rhs.values[n - i];
      }
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable operator*(const ValType &rhs) const {
    SingleVariable result{};
    result.values[0] = this->values[0] * rhs;
    for (std::size_t i = 1; i < Order + 1; ++i) {
      result.values[i] = this->values[i] * rhs;
    }
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  operator*(const ValType &lhs, const SingleVariable &rhs) {
    SingleVariable result{};
    result.values[0] = lhs * rhs.values[0];
    for (std::size_t i = 1; i < Order + 1; ++i) {
      result.values[i] = lhs * rhs.values[i];
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable inv() const {
    SingleVariable result{};
    result.values[0] = 1. / this->values[0];
    for (std::size_t n = 1; n < Order + 1; ++n) {
      result.values[n] = -this->values[n] * result.values[0];
      for (std::size_t i = 1; i < n; i++) {
        if (this->values[n - i] == 0. || result.values[i] == 0.) {
          continue;
        }
        result.values[n] -=
            Combination[n][i] * result.values[i] * this->values[n - i];
      }
      result.values[n] *= result.values[0];
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable
  operator/(const SingleVariable &rhs) const {
    SingleVariable result{};
    auto inv_value = 1. / rhs.values[0];
    result.values[0] = this->values[0] * inv_value;
    for (std::size_t n = 1; n < Order + 1; ++n) {
      result.values[n] = this->values[n] * inv_value;
      for (std::size_t i = 0; i < n; i++) {
        result.values[n] -= Combination[n][i] * result.values[i] *
                            rhs.values[n - i] * inv_value;
      }
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable operator/(const ValType &rhs) const {
    return *this * (1. / rhs);
  }

  [[nodiscard]] friend constexpr SingleVariable
  operator/(const ValType &lhs, const SingleVariable &rhs) {
    SingleVariable result{};
    result.values[0] = lhs / rhs.values[0];
    for (std::size_t n = 1; n < Order + 1; ++n) {
      result.values[n] = -rhs.values[n] * result.values[0];
      for (std::size_t i = 1; i < n; i++) {
        if (rhs.values[n - i] == 0. || result.values[i] == 0.) {
          continue;
        }
        result.values[n] -=
            Combination[n][i] * result.values[i] * rhs.values[n - i];
      }
      result.values[n] *= result.values[0];
    }
    return result;
  }

  [[nodiscard]] constexpr SingleVariable pow(const ValType &rhs) const {
    SingleVariable result{};
    result.values[0] = std::pow(this->values[0], rhs);
    ValType inv_value = 1 / this->values[0];
    for (std::size_t n = 1; n <= Order; ++n) {
      for (std::size_t i = 0; i < n; i++) {
        result.values[n] +=
            (Combination[n - 1][i] * (rhs + 1) - Combination[n][i]) *
            this->values[n - i] * result.values[i] * inv_value;
      }
    }
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable pow(const SingleVariable &self,
                                                    const ValType &rhs) {
    return self.pow(rhs);
  }

  [[nodiscard]] constexpr SingleVariable sqrt() const {
    return this->pow(1. / 2);
  }

  [[nodiscard]] friend constexpr SingleVariable
  sqrt(const SingleVariable &self) {
    return self.sqrt();
  }

  [[nodiscard]] constexpr SingleVariable cbrt() const {
    return this->pow(1. / 3);
  }

  [[nodiscard]] friend constexpr SingleVariable
  cbrt(const SingleVariable &self) {
    return self.cbrt();
  }

  [[nodiscard]] constexpr SingleVariable exp() const {
    SingleVariable result{};
    result.values[0] = std::exp(this->values[0]);
    for (std::size_t n = 1; n < Order + 1; ++n) {
      for (std::size_t i = 0; i < n; i++) {
        result.values[n] +=
            Combination[n - 1][i] * this->values[n - i] * result.values[i];
      }
    }
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  exp(const SingleVariable &self) {
    return self.exp();
  }

  [[nodiscard]] constexpr SingleVariable log() const {
    SingleVariable result{};
    result.values[0] = std::log(this->values[0]);
    for (std::size_t n = 1; n < Order + 1; ++n) {
      result.values[n] = this->values[n];
      for (std::size_t i = 1; i < n; i++) {
        result.values[n] -=
            Combination[n - 1][i - 1] * result.values[i] * this->values[n - i];
      }
      result.values[n] /= this->values[0];
    }
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  log(const SingleVariable &self) {
    return self.log();
  }

  [[nodiscard]] constexpr SingleVariable sin() const {
    SingleVariable result{};
    std::array<std::complex<ValType>, Order + 1> complex_values;
    complex_values[0] = std::exp(std::complex<ValType>(0, this->values[0]));
    for (std::size_t n = 1; n < Order + 1; ++n) {
      for (std::size_t i = 0; i < n; i++) {
        complex_values[n] += std::complex<ValType>(0., 1.) *
                             Combination[n - 1][i] * this->values[n - i] *
                             complex_values[i];
      }
    }
    for (std::size_t i = 0; i < Order + 1; i++) {
      result.values[i] = std::imag(complex_values[i]);
    }
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  sin(const SingleVariable &self) {
    return self.sin();
  }

  [[nodiscard]] constexpr SingleVariable cos() const {
    SingleVariable result{};
    std::array<std::complex<ValType>, Order + 1> complex_values;
    complex_values[0] = std::exp(std::complex<ValType>(0, this->values[0]));
    for (std::size_t n = 1; n < Order + 1; ++n) {
      for (std::size_t i = 0; i < n; i++) {
        complex_values[n] += std::complex<ValType>(0., 1.) *
                             Combination[n - 1][i] * this->values[n - i] *
                             complex_values[i];
      }
    }
    for (std::size_t i = 0; i < Order + 1; i++) {
      result.values[i] = std::real(complex_values[i]);
    }
    return result;
  }

  [[nodiscard]] friend constexpr SingleVariable
  cos(const SingleVariable &self) {
    return self.cos();
  }

  [[nodiscard]] constexpr SingleVariable tan() const {
    SingleVariable nume{};
    SingleVariable deno{};
    std::array<std::complex<ValType>, Order + 1> complex_values;
    complex_values[0] = std::exp(std::complex<ValType>(0, this->values[0]));
    for (std::size_t n = 1; n < Order + 1; ++n) {
      for (std::size_t i = 0; i < n; i++) {
        complex_values[n] += std::complex<ValType>(0., 1.) *
                             Combination[n - 1][i] * this->values[n - i] *
                             complex_values[i];
      }
    }
    for (std::size_t i = 0; i < Order + 1; i++) {
      nume.values[i] = std::imag(complex_values[i]);
    }
    for (std::size_t i = 0; i < Order + 1; i++) {
      deno.values[i] = std::real(complex_values[i]);
    }
    return nume / deno;
  }

  [[nodiscard]] friend constexpr SingleVariable
  tan(const SingleVariable &self) {
    return self.tan();
  }

  [[nodiscard]] constexpr ValType derivative(std::size_t order) const {
    return this->values.at(order);
  }
};

} // namespace Autodiff
