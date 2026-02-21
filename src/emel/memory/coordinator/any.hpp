#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/hybrid/sm.hpp"
#include "emel/memory/coordinator/kv/sm.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"

namespace emel::memory::coordinator {

enum class coordinator_kind : uint8_t {
  recurrent = 0,
  kv = 1,
  hybrid = 2,
};

class any {
 public:
  any() { construct(coordinator_kind::recurrent); }
  explicit any(const coordinator_kind kind) { construct(kind); }

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() { destroy(); }

  void set_kind(const coordinator_kind kind) {
    if (kind_ == kind) {
      return;
    }
    destroy();
    construct(kind);
  }

  coordinator_kind kind() const noexcept { return kind_; }

  bool process_event(const event::prepare_update & ev) {
    return process_update_(storage(), ev);
  }

  bool process_event(const event::prepare_batch & ev) {
    return process_batch_(storage(), ev);
  }

  bool process_event(const event::prepare_full & ev) {
    return process_full_(storage(), ev);
  }

  int32_t last_error() const noexcept { return last_error_(storage_const()); }

  event::memory_status last_status() const noexcept {
    return last_status_(storage_const());
  }

 private:
  using process_update_fn = bool (*)(void *, const event::prepare_update &);
  using process_batch_fn = bool (*)(void *, const event::prepare_batch &);
  using process_full_fn = bool (*)(void *, const event::prepare_full &);
  using last_error_fn = int32_t (*)(const void *) noexcept;
  using last_status_fn = event::memory_status (*)(const void *) noexcept;
  using destroy_fn = void (*)(void *) noexcept;

  static constexpr std::size_t k_max_size =
      sizeof(recurrent::sm) > sizeof(kv::sm)
          ? (sizeof(recurrent::sm) > sizeof(hybrid::sm) ? sizeof(recurrent::sm)
                                                        : sizeof(hybrid::sm))
          : (sizeof(kv::sm) > sizeof(hybrid::sm) ? sizeof(kv::sm) : sizeof(hybrid::sm));
  static constexpr std::size_t k_max_align =
      alignof(recurrent::sm) > alignof(kv::sm)
          ? (alignof(recurrent::sm) > alignof(hybrid::sm) ? alignof(recurrent::sm)
                                                          : alignof(hybrid::sm))
          : (alignof(kv::sm) > alignof(hybrid::sm) ? alignof(kv::sm)
                                                   : alignof(hybrid::sm));

  using storage_t = std::aligned_storage_t<k_max_size, k_max_align>;

  template <class sm>
  static sm * ptr(void * storage) noexcept {
    return std::launder(reinterpret_cast<sm *>(storage));
  }

  template <class sm>
  static const sm * ptr(const void * storage) noexcept {
    return std::launder(reinterpret_cast<const sm *>(storage));
  }

  template <class sm>
  static bool process_update_impl(void * storage, const event::prepare_update & ev) {
    return ptr<sm>(storage)->process_event(ev);
  }

  template <class sm>
  static bool process_batch_impl(void * storage, const event::prepare_batch & ev) {
    return ptr<sm>(storage)->process_event(ev);
  }

  template <class sm>
  static bool process_full_impl(void * storage, const event::prepare_full & ev) {
    return ptr<sm>(storage)->process_event(ev);
  }

  template <class sm>
  static int32_t last_error_impl(const void * storage) noexcept {
    return ptr<sm>(storage)->last_error();
  }

  template <class sm>
  static event::memory_status last_status_impl(const void * storage) noexcept {
    return ptr<sm>(storage)->last_status();
  }

  template <class sm>
  static void destroy_impl(void * storage) noexcept {
    ptr<sm>(storage)->~sm();
  }

  template <class sm>
  void construct_impl(const coordinator_kind kind) {
    new (&storage_) sm();
    kind_ = kind;
    process_update_ = &process_update_impl<sm>;
    process_batch_ = &process_batch_impl<sm>;
    process_full_ = &process_full_impl<sm>;
    last_error_ = &last_error_impl<sm>;
    last_status_ = &last_status_impl<sm>;
    destroy_ = &destroy_impl<sm>;
  }

  void construct(const coordinator_kind kind) {
    switch (kind) {
      case coordinator_kind::recurrent:
        construct_impl<recurrent::sm>(kind);
        return;
      case coordinator_kind::kv:
        construct_impl<kv::sm>(kind);
        return;
      case coordinator_kind::hybrid:
        construct_impl<hybrid::sm>(kind);
        return;
    }
    construct_impl<recurrent::sm>(coordinator_kind::recurrent);
  }

  void destroy() noexcept {
    if (destroy_ != nullptr) {
      destroy_(storage());
    }
    destroy_ = nullptr;
    process_update_ = nullptr;
    process_batch_ = nullptr;
    process_full_ = nullptr;
    last_error_ = nullptr;
    last_status_ = nullptr;
  }

  void * storage() noexcept { return &storage_; }
  const void * storage_const() const noexcept { return &storage_; }

  storage_t storage_{};
  coordinator_kind kind_ = coordinator_kind::recurrent;
  process_update_fn process_update_ = nullptr;
  process_batch_fn process_batch_ = nullptr;
  process_full_fn process_full_ = nullptr;
  last_error_fn last_error_ = nullptr;
  last_status_fn last_status_ = nullptr;
  destroy_fn destroy_ = nullptr;
};

}  // namespace emel::memory::coordinator
