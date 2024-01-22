#pragma once

#include <coroutine>
#include <exception>
#include <memory>
#include <type_traits>

namespace Generator {
template <typename T> struct is_unique_ptr : std::false_type {};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

/*!
 * 最低限の Generator の実装アルゴリズムとかで使うと良い
 * C++23 でstd に追加されたらそれに変えること
 * */
template <class ValType> struct Generator {
  struct promise_type;
  using handle = std::coroutine_handle<promise_type>;

  struct promise_type {
    ValType current_value;

    static auto get_return_object_on_allocation_failure() {
      return Generator{nullptr};
    }

    auto get_return_object() { return Generator{handle::from_promise(*this)}; }

    auto initial_suspend() { return std::suspend_always{}; }

    auto final_suspend() noexcept { return std::suspend_always{}; }

    void unhandled_exception() { std::terminate(); }

    void return_void() {}

    // void return_value(ValType value) {current_value = value;}
    auto yield_value(ValType value) {
      if constexpr (is_unique_ptr<ValType>::value) {
        current_value = std::move(value);
      } else {
        current_value = value;
      }
      return std::suspend_always{};
    }
  };

  bool next() {
    if (coro) [[likely]] {
      coro();
      return !coro.done();
    }
    return false;
  }

  ValType value() {
    if constexpr (is_unique_ptr<ValType>::value) {
      return std::move(coro.promise().current_value);
    } else {
      return coro.promise().current_value;
    }
  }

  Generator &operator=(const Generator &) = delete;
  Generator &operator=(Generator &&) = delete;
  Generator(Generator const &) = delete;

  Generator(Generator &&rhs) noexcept : coro(rhs.coro) { rhs.coro = nullptr; }

  ~Generator() {
    if (coro) {
      coro.destroy();
    }
  }

private:
  explicit Generator(handle h) : coro(h) {}

  handle coro;

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = ValType;
    using difference_type = std::ptrdiff_t;
    using pointer = ValType *;
    using reference = ValType &;

    Generator *gen;

    bool done = false;

    explicit iterator(Generator *gen, bool done) noexcept
        : gen(gen), done(done) {}

    friend auto operator==(const iterator &lhs, const iterator &rhs) -> bool {
      return (lhs.done == rhs.done) && (lhs.gen == rhs.gen);
    }

    friend auto operator!=(const iterator &lhs, const iterator &rhs) -> bool {
      return !(lhs == rhs);
    }

    auto operator++() -> iterator & {
      done = !gen->next();
      return *this;
    }

    auto operator++(int) -> iterator {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    auto operator*() const -> ValType { return gen->value(); }
  };

public:
  auto begin() -> iterator {
    this->next();
    if (this->coro.done()) {
      return this->end();
    }
    return iterator(this, false);
  }

  auto end() -> iterator { return iterator(this, true); }
};

} // namespace Generator
