#pragma once

#include <type_traits>
#include <utility>

namespace emel {

template <class Signature>
struct callback;

template <class R, class... Args>
struct callback<R(Args...)> {
  using result_type = R;
  using thunk_fn = R (*)(void *, Args...);

  void * object = nullptr;
  thunk_fn thunk = nullptr;

  constexpr callback() noexcept = default;
  constexpr callback(void * obj, thunk_fn fn) noexcept : object(obj), thunk(fn) {}

  constexpr explicit operator bool() const noexcept { return thunk != nullptr; }

  R operator()(Args... args) const noexcept {
    if (thunk == nullptr) {
      if constexpr (!std::is_void_v<R>) {
        return R{};
      } else {
        return;
      }
    }
    if constexpr (std::is_void_v<R>) {
      thunk(object, std::forward<Args>(args)...);
      return;
    } else {
      return thunk(object, std::forward<Args>(args)...);
    }
  }

  template <auto Fn>
  static constexpr callback from() noexcept {
    return callback{
      nullptr,
      [](void *, Args... args) -> R {
        if constexpr (std::is_void_v<R>) {
          Fn(std::forward<Args>(args)...);
          return;
        } else {
          return Fn(std::forward<Args>(args)...);
        }
      },
    };
  }

  template <class T, auto MemFn>
  static constexpr callback from(T * obj) noexcept {
    return callback{
      obj,
      [](void * ptr, Args... args) -> R {
        if constexpr (std::is_void_v<R>) {
          (static_cast<T *>(ptr)->*MemFn)(std::forward<Args>(args)...);
          return;
        } else {
          return (static_cast<T *>(ptr)->*MemFn)(std::forward<Args>(args)...);
        }
      },
    };
  }

  template <class T, auto MemFn>
  static constexpr callback from(const T * obj) noexcept {
    return callback{
      const_cast<T *>(obj),
      [](void * ptr, Args... args) -> R {
        if constexpr (std::is_void_v<R>) {
          (static_cast<const T *>(ptr)->*MemFn)(std::forward<Args>(args)...);
          return;
        } else {
          return (static_cast<const T *>(ptr)->*MemFn)(std::forward<Args>(args)...);
        }
      },
    };
  }
};

}  // namespace emel
