#pragma once

#include <cstddef>
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
  constexpr callback(thunk_fn fn) noexcept : object(nullptr), thunk(fn) {}

  template <class T>
    requires(!std::is_pointer_v<T> &&
             !std::is_same_v<std::remove_cv_t<T>, std::nullptr_t>)
  constexpr callback(T & obj, thunk_fn fn) noexcept : object(static_cast<void *>(&obj)), thunk(fn) {}

  template <class T>
    requires(!std::is_pointer_v<T> &&
             !std::is_same_v<std::remove_cv_t<T>, std::nullptr_t>)
  constexpr callback(const T & obj, thunk_fn fn) noexcept
      : object(const_cast<T *>(&obj)), thunk(fn) {}

  constexpr explicit operator bool() const noexcept { return thunk != nullptr; }

  R operator()(Args... args) const noexcept {
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
    requires(std::is_member_function_pointer_v<decltype(MemFn)>)
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
    requires(std::is_member_function_pointer_v<decltype(MemFn)>)
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

  template <class T, auto Fn>
    requires(!std::is_member_function_pointer_v<decltype(Fn)> &&
             std::is_invocable_r_v<R, decltype(Fn), T &, Args...>)
  static constexpr callback from(T * obj) noexcept {
    return callback{
      obj,
      [](void * ptr, Args... args) -> R {
        if constexpr (std::is_void_v<R>) {
          Fn(*static_cast<T *>(ptr), std::forward<Args>(args)...);
          return;
        } else {
          return Fn(*static_cast<T *>(ptr), std::forward<Args>(args)...);
        }
      },
    };
  }

  template <class T, auto Fn>
    requires(!std::is_member_function_pointer_v<decltype(Fn)> &&
             std::is_invocable_r_v<R, decltype(Fn), const T &, Args...>)
  static constexpr callback from(const T * obj) noexcept {
    return callback{
      const_cast<T *>(obj),
      [](void * ptr, Args... args) -> R {
        if constexpr (std::is_void_v<R>) {
          Fn(*static_cast<const T *>(ptr), std::forward<Args>(args)...);
          return;
        } else {
          return Fn(*static_cast<const T *>(ptr), std::forward<Args>(args)...);
        }
      },
    };
  }

  template <class T, auto Fn>
    requires(!std::is_member_function_pointer_v<decltype(Fn)> &&
             std::is_invocable_r_v<R, decltype(Fn), T &, Args...>)
  static constexpr callback from(T & obj) noexcept {
    return from<T, Fn>(&obj);
  }

  template <class T, auto Fn>
    requires(!std::is_member_function_pointer_v<decltype(Fn)> &&
             std::is_invocable_r_v<R, decltype(Fn), const T &, Args...>)
  static constexpr callback from(const T & obj) noexcept {
    return from<T, Fn>(&obj);
  }
};

}  // namespace emel
