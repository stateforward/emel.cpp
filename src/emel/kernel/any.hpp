#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "emel/sm.hpp"

namespace emel::kernel {

class any {
 public:
  any() : core_(detect_host_kind()) {}
  explicit any(const kernel_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const kernel_kind kind) { core_.set_kind(kind); }

  kernel_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::dispatch & ev) { return core_.process_event(ev); }

  bool process_event(const event::configure_kind &ev) {
    core_.set_kind(ev.kind);
    return true;
  }

  bool process_event(const event::capture_diagnostics &ev) {
    ev.out.optimized_flash_dispatch_calls = optimized_flash_dispatch_count();
    ev.out.shared_flash_dispatch_calls = shared_flash_dispatch_count();
    ev.out.optimized_q2_dispatch_calls = optimized_q2_dispatch_count();
    ev.out.shared_q2_dispatch_calls = shared_q2_dispatch_count();
    ev.out.optimized_q3_dispatch_calls = optimized_q3_dispatch_count();
    ev.out.shared_q3_dispatch_calls = shared_q3_dispatch_count();
    ev.out.optimized_q4_dispatch_calls = optimized_q4_dispatch_count();
    ev.out.optimized_q4_vector_dispatch_calls =
        optimized_q4_vector_dispatch_count();
    ev.out.optimized_q4_vector_packed_dispatch_calls =
        optimized_q4_vector_packed_dispatch_count();
    ev.out.optimized_q4_vector_packed_q8_rhs_dispatch_calls =
        optimized_q4_vector_packed_q8_rhs_dispatch_count();
    ev.out.shared_q4_dispatch_calls = shared_q4_dispatch_count();
    ev.out.optimized_q6_dispatch_calls = optimized_q6_dispatch_count();
    ev.out.optimized_q6_vector_dispatch_calls =
        optimized_q6_vector_dispatch_count();
    ev.out.optimized_q6_vector_argmax_dispatch_calls =
        optimized_q6_vector_argmax_dispatch_count();
    ev.out.optimized_q6_vector_packed_dispatch_calls =
        optimized_q6_vector_packed_dispatch_count();
    ev.out.optimized_q6_vector_packed_q8_rhs_dispatch_calls =
        optimized_q6_vector_packed_q8_rhs_dispatch_count();
    ev.out.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls =
        optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count();
    ev.out.optimized_q6_vector_prepared_q8_rhs_dispatch_calls =
        optimized_q6_vector_prepared_q8_rhs_dispatch_count();
    ev.out.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls =
        optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count();
    ev.out.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls =
        optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count();
    ev.out.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls =
        optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count();
    ev.out.shared_q6_dispatch_calls = shared_q6_dispatch_count();
    return true;
  }

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

  uint64_t optimized_f16_vector_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_f16_vector_dispatch_count(); }) {
        count = sm.optimized_f16_vector_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_f32_vector_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_f32_vector_dispatch_count(); }) {
        count = sm.optimized_f32_vector_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_conv_transpose_f32_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_conv_transpose_f32_dispatch_count(); }) {
        count = sm.optimized_conv_transpose_f32_dispatch_count();
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

  uint64_t optimized_q4_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q4_dispatch_count(); }) {
        count = sm.optimized_q4_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q4_vector_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q4_vector_dispatch_count(); }) {
        count = sm.optimized_q4_vector_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q4_vector_packed_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q4_vector_packed_dispatch_count(); }) {
        count = sm.optimized_q4_vector_packed_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q4_vector_packed_q8_rhs_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q4_vector_packed_q8_rhs_dispatch_count(); }) {
        count = sm.optimized_q4_vector_packed_q8_rhs_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t shared_q4_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.shared_q4_dispatch_count(); }) {
        count = sm.shared_q4_dispatch_count();
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

  uint64_t optimized_q6_vector_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_dispatch_count(); }) {
        count = sm.optimized_q6_vector_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_argmax_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_argmax_dispatch_count(); }) {
        count = sm.optimized_q6_vector_argmax_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_packed_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_packed_dispatch_count(); }) {
        count = sm.optimized_q6_vector_packed_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_packed_q8_rhs_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_packed_q8_rhs_dispatch_count(); }) {
        count = sm.optimized_q6_vector_packed_q8_rhs_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count(); }) {
        count = sm.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_prepared_q8_rhs_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_prepared_q8_rhs_dispatch_count(); }) {
        count = sm.optimized_q6_vector_prepared_q8_rhs_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count(); }) {
        count = sm.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (
          requires { sm.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count(); }) {
        count = sm.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count();
      } else {
        count = 0u;
      }
    });
    return count;
  }

  uint64_t optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count() const noexcept {
    uint64_t count = 0u;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count(); }) {
        count = sm.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count();
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
  using sm_list = stateforward::sml::aux::type_list<x86_64::sm, aarch64::sm>;
  using event_list = stateforward::sml::aux::type_list<
      event::dispatch
#define EMEL_KERNEL_ANY_EVENT_TYPE(op_name) , event::op_name
      EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_ANY_EVENT_TYPE)
#undef EMEL_KERNEL_ANY_EVENT_TYPE
      >;

  emel::sm_any<kernel_kind, sm_list, event_list> core_{};
};

}  // namespace emel::kernel
