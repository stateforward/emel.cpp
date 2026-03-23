#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/cuda/sm.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/metal/sm.hpp"
#include "emel/kernel/vulkan/sm.hpp"
#include "emel/kernel/wasm/sm.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "emel/sm.hpp"

namespace emel::kernel {

enum class kernel_kind : uint8_t {
  x86_64 = 0,
  aarch64 = 1,
  wasm = 2,
  cuda = 3,
  metal = 4,
  vulkan = 5,
};

class any {
 public:
  any() = default;
  explicit any(const kernel_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const kernel_kind kind) { core_.set_kind(kind); }

  kernel_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::dispatch & ev) { return core_.process_event(ev); }

  template <class event_type>
    requires(::emel::kernel::is_op_event_v<event_type>)
  bool process_event(const event_type & ev) {
    return core_.process_event(ev);
  }

  uint64_t optimized_flash_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_flash_dispatch_count(); }) {
        count = sm.optimized_flash_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t shared_flash_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.shared_flash_dispatch_count(); }) {
        count = sm.shared_flash_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q2_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q2_dispatch_count(); }) {
        count = sm.optimized_q2_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t shared_q2_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.shared_q2_dispatch_count(); }) {
        count = sm.shared_q2_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q3_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q3_dispatch_count(); }) {
        count = sm.optimized_q3_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t shared_q3_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.shared_q3_dispatch_count(); }) {
        count = sm.shared_q3_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_dispatch_count(); }) {
        count = sm.optimized_q6_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t shared_q6_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.shared_q6_dispatch_count(); }) {
        count = sm.shared_q6_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

 private:
  using sm_list = boost::sml::aux::type_list<x86_64::sm, aarch64::sm, wasm::sm, cuda::sm,
                                             metal::sm, vulkan::sm>;
  using event_list = boost::sml::aux::type_list<
      event::dispatch
#define EMEL_KERNEL_ANY_EVENT_TYPE(op_name) , event::op_name
      EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_ANY_EVENT_TYPE)
#undef EMEL_KERNEL_ANY_EVENT_TYPE
      >;

  emel::sm_any<kernel_kind, sm_list, event_list> core_{};
};

}  // namespace emel::kernel
