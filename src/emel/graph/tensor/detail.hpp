#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <boost/sml.hpp>
#include <boost/sml/utility/sm_pool.hpp>

#include "emel/error/error.hpp"
#include "emel/graph/tensor/errors.hpp"
#include "emel/graph/tensor/events.hpp"

namespace emel::graph::tensor::detail {

inline constexpr int32_t max_tensors = 65536;

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

template <class value_type>
value_type & bind_or_sink(value_type * ptr, value_type & sink) noexcept {
  value_type * choices[2] = {&sink, ptr};
  return *choices[static_cast<size_t>(ptr != nullptr)];
}

enum class lifecycle_state : uint8_t {
  unallocated = 0u,
  empty = 1u,
  filled = 2u,
  leaf_filled = 3u,
  internal_error = 4u,
};

struct runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  bool accepted = false;
};

struct reserve_tensor_runtime {
  const event::reserve_tensor & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct publish_filled_tensor_runtime {
  const event::publish_filled_tensor & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct release_tensor_ref_runtime {
  const event::release_tensor_ref & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct reset_tensor_epoch_runtime {
  const event::reset_tensor_epoch & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct capture_tensor_state_runtime {
  const event::capture_tensor_state & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct tensor_storage {
  std::array<lifecycle_state, static_cast<size_t>(max_tensors)> lifecycle = {};
  std::array<uint8_t, static_cast<size_t>(max_tensors)> kind = {};
  std::array<uint32_t, static_cast<size_t>(max_tensors)> seed_refs = {};
  std::array<uint32_t, static_cast<size_t>(max_tensors)> live_refs = {};
  std::array<void *, static_cast<size_t>(max_tensors)> buffer = {};
  std::array<uint64_t, static_cast<size_t>(max_tensors)> buffer_bytes = {};

  void reset() noexcept {
    lifecycle.fill(lifecycle_state::unallocated);
    kind.fill(0u);
    seed_refs.fill(0u);
    live_refs.fill(0u);
    buffer.fill(nullptr);
    buffer_bytes.fill(0u);
  }
};

struct op_reserve {
  void * buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  uint32_t consumer_refs = 0u;
  uint8_t is_leaf = 0u;
};
struct op_publish_filled {};
struct op_release_ref {};
struct op_reset_epoch {};
struct router_ready {};

struct tensor_router_model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct can_reserve {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_reserve> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::unallocated;
      }
    };

    struct do_reserve {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_reserve> & ev) const noexcept {
        const uint8_t leaf_flag = static_cast<uint8_t>(ev.event.is_leaf != 0u);
        const uint8_t compute_flag = static_cast<uint8_t>(1u - leaf_flag);
        const std::array<lifecycle_state, 2> reserve_states{
          lifecycle_state::empty,
          lifecycle_state::leaf_filled,
        };

        storage.kind[ev.id] = leaf_flag;
        storage.lifecycle[ev.id] = reserve_states[leaf_flag];
        storage.seed_refs[ev.id] = static_cast<uint32_t>(compute_flag * ev.event.consumer_refs);
        storage.live_refs[ev.id] = static_cast<uint32_t>(compute_flag * ev.event.consumer_refs);
        storage.buffer[ev.id] = ev.event.buffer;
        storage.buffer_bytes[ev.id] = ev.event.buffer_bytes;
      }
    };

    struct can_publish_filled {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_publish_filled> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::empty &&
               storage.kind[ev.id] == 0u;
      }
    };

    struct do_publish_filled {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_publish_filled> & ev) const noexcept {
        storage.lifecycle[ev.id] = lifecycle_state::filled;
        storage.live_refs[ev.id] = storage.seed_refs[ev.id];
      }
    };

    struct can_release_keep_filled {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::filled &&
               storage.live_refs[ev.id] > 1u;
      }
    };

    struct do_release_keep_filled {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        storage.live_refs[ev.id] -= 1u;
      }
    };

    struct can_release_to_empty {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::filled &&
               storage.live_refs[ev.id] == 1u;
      }
    };

    struct do_release_to_empty {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        storage.live_refs[ev.id] = 0u;
        storage.lifecycle[ev.id] = lifecycle_state::empty;
      }
    };

    struct can_release_leaf_noop {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::leaf_filled;
      }
    };

    struct release_leaf_noop {
      void operator()(tensor_storage &, const sml::utility::indexed_event<op_release_ref> &) const noexcept {}
    };

    struct can_reset_filled_compute {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::filled &&
               storage.kind[ev.id] == 0u;
      }
    };

    struct do_reset_filled_compute {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        storage.lifecycle[ev.id] = lifecycle_state::empty;
        storage.live_refs[ev.id] = 0u;
      }
    };

    struct can_reset_empty_compute {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::empty &&
               storage.kind[ev.id] == 0u;
      }
    };

    struct do_reset_empty_compute {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        storage.live_refs[ev.id] = 0u;
      }
    };

    struct can_reset_leaf_noop {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] == lifecycle_state::leaf_filled;
      }
    };

    struct reset_leaf_noop {
      void operator()(tensor_storage &, const sml::utility::indexed_event<op_reset_epoch> &) const noexcept {}
    };

    struct cannot_reserve {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_reserve> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               storage.lifecycle[ev.id] != lifecycle_state::unallocated;
      }
    };

    struct cannot_publish_filled {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_publish_filled> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               !can_publish_filled{}(storage, ev);
      }
    };

    struct cannot_release {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               !can_release_keep_filled{}(storage, ev) &&
               !can_release_to_empty{}(storage, ev) &&
               !can_release_leaf_noop{}(storage, ev);
      }
    };

    struct cannot_reset {
      bool operator()(const tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        return ev.id < storage.lifecycle.size() &&
               !can_reset_filled_compute{}(storage, ev) &&
               !can_reset_empty_compute{}(storage, ev) &&
               !can_reset_leaf_noop{}(storage, ev);
      }
    };

    struct mark_internal_error_from_reserve {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_reserve> & ev) const noexcept {
        storage.lifecycle[ev.id] = lifecycle_state::internal_error;
        storage.live_refs[ev.id] = 0u;
      }
    };

    struct mark_internal_error_from_publish {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_publish_filled> & ev) const noexcept {
        storage.lifecycle[ev.id] = lifecycle_state::internal_error;
        storage.live_refs[ev.id] = 0u;
      }
    };

    struct mark_internal_error_from_release {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_release_ref> & ev) const noexcept {
        storage.lifecycle[ev.id] = lifecycle_state::internal_error;
        storage.live_refs[ev.id] = 0u;
      }
    };

    struct mark_internal_error_from_reset {
      void operator()(tensor_storage & storage,
                      const sml::utility::indexed_event<op_reset_epoch> & ev) const noexcept {
        storage.lifecycle[ev.id] = lifecycle_state::internal_error;
        storage.live_refs[ev.id] = 0u;
      }
    };

    // clang-format off
    return sml::make_transition_table(
        sml::state<router_ready> <= *sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_reserve>> [ can_reserve{} ] / do_reserve{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_reserve>> [ cannot_reserve{} ]
          / mark_internal_error_from_reserve{}

      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_publish_filled>>
          [ can_publish_filled{} ] / do_publish_filled{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_publish_filled>>
          [ cannot_publish_filled{} ] / mark_internal_error_from_publish{}

      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_release_ref>>
          [ can_release_keep_filled{} ] / do_release_keep_filled{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_release_ref>>
          [ can_release_to_empty{} ] / do_release_to_empty{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_release_ref>>
          [ can_release_leaf_noop{} ] / release_leaf_noop{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_release_ref>>
          [ cannot_release{} ] / mark_internal_error_from_release{}

      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_reset_epoch>>
          [ can_reset_filled_compute{} ] / do_reset_filled_compute{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_reset_epoch>>
          [ can_reset_empty_compute{} ] / do_reset_empty_compute{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_reset_epoch>>
          [ can_reset_leaf_noop{} ] / reset_leaf_noop{}
      , sml::state<router_ready> <= sml::state<router_ready>
          + sml::event<sml::utility::indexed_event<op_reset_epoch>>
          [ cannot_reset{} ] / mark_internal_error_from_reset{}
    );
    // clang-format on
  }
};

using tensor_pool = boost::sml::utility::sm_pool<tensor_storage, tensor_router_model>;

}  // namespace emel::graph::tensor::detail
