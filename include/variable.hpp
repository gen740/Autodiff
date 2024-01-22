#include <algorithm>
#include <array>
#include <bitset>
#include <span>
#include <vector>

#include "constant.hpp"
#include "generator.hpp"
#include "single_variable.hpp"

namespace Autodiff {

template <size_t I, size_t N> struct Pow {
  static constexpr size_t value = I * Pow<I, N - 1>::value;
};

template <size_t I> struct Pow<I, 0> {
  static constexpr size_t value = 1;
};

template <size_t Order, size_t N> struct InternalNum {
  InternalNum() = default;
  InternalNum(InternalNum &&) noexcept = default;
  InternalNum &operator=(const InternalNum &) = default;
  InternalNum &operator=(InternalNum &&) noexcept = default;

  InternalNum(const InternalNum &other)
      : repr(other.repr), counter(other.counter) {}

  explicit InternalNum(size_t repr) {
    for (size_t i = 0; i < Order; i++) {
      this->repr[i] = repr % (N + 1);
      repr /= (N + 1);
    }
  }

  bool valid() {
    auto prev = Order;
    for (size_t i = 0; i < Order; i++) {
      if (prev < repr[i]) {
        return false;
      }
      prev = repr[i];
    }
    return true;
  }

  ~InternalNum() = default;

  std::array<size_t, Order> repr{};
  size_t counter = 0;

  static constexpr size_t MAX_NUM = std::pow(N, Order);

  [[nodiscard]] std::array<size_t, Order> get_repr_arr() const { return repr; }

  [[nodiscard]] const std::array<size_t, Order> *get_repr_arr_ptr() const {
    return &repr;
  }

  [[nodiscard]] size_t get_repr() const {
    size_t ret = 0;
    for (size_t i = 0; i < Order; i++) {
      ret += repr[i] * std::pow(N + 1, i);
    }
    return ret;
  }

  void set(size_t num) {
    if (counter >= Order) [[unlikely]] {
      throw std::runtime_error("BaseN::set: counter >= Order");
    }
    if (num >= N + 1) [[unlikely]] {
      throw std::runtime_error("BaseN::set: num >= N");
    }
    repr[counter] = num;
    counter++;
  }

  [[nodiscard]] size_t at(size_t i) const {
    if (i >= N + 1) [[unlikely]] {
      throw std::runtime_error("BaseN::set: num >= N");
    }
    return repr[i];
  }

  friend std::ostream &operator<<(std::ostream &os, const InternalNum &num) {
    for (size_t i = 0; i < Order; i++) {
      os << num.repr[i];
    }
    return os;
  }

  void normalize() { std::sort(repr.begin(), repr.end(), std::greater<>()); }
};

template <size_t Deps = 2, size_t Order = 2> class Variable {
public:
  using VecB = std::vector<InternalNum<Order, Deps>>;

  std::array<double, Pow<Deps + 1, Order>::value> repr{};

  [[nodiscard]] constexpr Variable
  operator+([[maybe_unused]] const Variable &rhs) const {
    Variable ret(rhs);
    for (size_t i = 0; i < rhs.repr.size(); i++) {
      ret.repr[i] += this->repr[i];
    }
    return ret;
  }

  [[nodiscard]] friend constexpr Variable
  operator+([[maybe_unused]] double lhs, [[maybe_unused]] const Variable &rhs) {
    Variable ret(rhs);
    ret.repr[0] += lhs;
    return ret;
  }

  [[nodiscard]] Variable operator*([[maybe_unused]] const Variable &rhs) {
    Variable ret;
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      auto ar = std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (size_t j = 0, arr_len = ar.size(),
                  coeff_size = MULT_COEFF.at(arr_len).size();
           j < coeff_size; ++j) {
        auto idx1 = InternalNum<Order, Deps>(0);
        auto idx2 = InternalNum<Order, Deps>(0);
        for (size_t k = 0; k < arr_len; ++k) {
          auto k1 = MULT_COEFF.at(arr_len).at(j).at(k);
          auto k2 = MULT_COEFF.at(arr_len).at(coeff_size - j - 1).at(k);
          if (k1 != 0u) {
            idx1.set(ar[k1 - 1]);
          }
          if (k2 != 0u) {
            idx2.set(ar[k2 - 1]);
          }
        }
        idx1.normalize();
        idx2.normalize();
        ret.repr[i.get_repr()] +=
            this->repr[idx1.get_repr()] * rhs.repr[idx2.get_repr()];
      }
    }
    return ret;
  }

  [[nodiscard]] friend constexpr Variable operator*(double lhs,
                                                    const Variable &rhs) {
    Variable ret(rhs);
    for (auto &&i : ret.repr) {
      i *= lhs;
    }
    return ret;
  }

  [[nodiscard]] Variable inv() const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.inv();
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable inv(const Variable &other) { return other.inv(); }

  [[nodiscard]] Variable sin() const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.sin();
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable sin(const Variable &other) { return other.sin(); }

  [[nodiscard]] Variable cos() const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.cos();
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable cos(const Variable &other) { return other.cos(); }

  [[nodiscard]] Variable tan() const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.tan();
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable tan(const Variable &other) { return other.tan(); }

  [[nodiscard]] Variable exp() const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.exp();
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable exp(const Variable &other) { return other.exp(); }

  [[nodiscard]] Variable log() const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.log();
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable log(const Variable &other) { return other.log(); }

  [[nodiscard]] Variable pow(double p) const {
    Variable ret;
    SingleVariable<Order, double> x;
    x.set_value(this->repr[0], 0);
    x.set_value(1.0, 1);
    x = x.pow(p);
    for (size_t n = 0; n < repr.size(); n++) {
      auto i = InternalNum<Order, Deps>(n);
      if (!i.valid()) {
        continue;
      }
      const auto *arr = i.get_repr_arr_ptr();
      const auto ar =
          std::span(arr->begin(), std::find(arr->begin(), arr->end(), 0));
      for (const auto &j : SINGLE_COEFF.at(ar.size())) {
        auto tmp = 1.0;
        for (const auto &k : j) {
          auto idx = InternalNum<Order, Deps>(0);
          for (const auto &l : k) {
            idx.set(ar[l - 1]);
          }
          idx.normalize();
          auto idx_num = idx.get_repr();
          if (this->repr[idx_num] == 0.) [[unlikely]] {
            goto Escape;
          }
          tmp *= this->repr[idx_num];
        }
        ret.repr[i.get_repr()] += tmp * x.derivative(j.size());
      }
    Escape:
      void();
    }
    return ret;
  }

  friend constexpr Variable pow(const Variable &other, double val) {
    return other.pow(val);
  }

  [[nodiscard]] constexpr Variable sqrt() const { return this->pow(1. / 2.); }

  friend constexpr Variable sqrt(const Variable &other) { return other.sqrt(); }

  [[nodiscard]] constexpr Variable cbrt() const { return this->pow(1. / 3.); }

  friend constexpr Variable cbrt(const Variable &other) { return other.cbrt(); }

  void set(std::vector<size_t> vec, double val) {
    auto num = InternalNum<Order, Deps>();
    if (vec.size() > Order) {
      throw std::runtime_error("set: vec.size() > Order");
    }
    for (auto &&i : vec) {
      num.set(i);
    }
    num.normalize();
    repr[num.get_repr()] = val;
  }

  template <std::integral... Args> double derivative(Args... args) {
    auto num = InternalNum<Order, Deps>();
    return derivative_impl(num, args...);
  }

  template <std::integral Head, std::integral... Tails>
  double derivative_impl(InternalNum<Order, Deps> &num, Head head,
                         Tails... tails) {
    num.set(head);
    return derivative_impl(num, tails...);
  }

  template <std::integral Head>
  double derivative_impl(InternalNum<Order, Deps> &num, Head head) {
    num.set(head);
    num.normalize();
    return repr[num.get_repr()];
  }
};

} // namespace Autodiff
