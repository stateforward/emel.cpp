#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <boost/sml.hpp>
#include <boost/sml/utility/sm_pool.hpp>

#include "emel/error/error.hpp"

namespace emel::memory::recurrent::detail {

inline constexpr int32_t max_sequences = 256;
inline constexpr int32_t invalid_slot = -1;

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline emel::error::type cast_api_error(const emel::error::type err) noexcept {
  return err;
}

inline bool valid_sequence_id(const int32_t max_sequence_count, const int32_t seq_id) noexcept {
  return seq_id >= 0 && seq_id < max_sequence_count;
}

struct slot_activate {};
struct slot_deactivate {};
struct router_ready {};

struct slot_storage {
  std::array<uint8_t, max_sequences> active = {};

  void reset() noexcept { active.fill(0); }
};

struct slot_router_model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct can_activate {
      bool operator()(const slot_storage & storage,
                      const sml::utility::indexed_event<slot_activate> & ev) const noexcept {
        return ev.id < storage.active.size() && storage.active[ev.id] == 0;
      }
    };

    struct do_activate {
      void operator()(slot_storage & storage,
                      const sml::utility::indexed_event<slot_activate> & ev) const noexcept {
        storage.active[ev.id] = 1;
      }
    };

    struct can_deactivate {
      bool operator()(const slot_storage & storage,
                      const sml::utility::indexed_event<slot_deactivate> & ev) const noexcept {
        return ev.id < storage.active.size() && storage.active[ev.id] != 0;
      }
    };

    struct do_deactivate {
      void operator()(slot_storage & storage,
                      const sml::utility::indexed_event<slot_deactivate> & ev) const noexcept {
        storage.active[ev.id] = 0;
      }
    };

    // clang-format off
    return sml::make_transition_table(
        sml::state<router_ready> <= *sml::state<router_ready>
            + sml::event<sml::utility::indexed_event<slot_activate>>
            [ can_activate{} ] / do_activate{}
      , sml::state<router_ready> <= sml::state<router_ready>
            + sml::event<sml::utility::indexed_event<slot_deactivate>>
            [ can_deactivate{} ] / do_deactivate{}
    );
    // clang-format on
  }
};

using slot_pool = boost::sml::utility::sm_pool<slot_storage, slot_router_model>;

}  // namespace emel::memory::recurrent::detail
