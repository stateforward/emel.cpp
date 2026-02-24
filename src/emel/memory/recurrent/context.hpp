#pragma once

#include <array>
#include <cstdint>

#include <boost/sml.hpp>
#include <boost/sml/utility/sm_pool.hpp>

#include "emel/emel.h"

namespace emel::memory::recurrent::action {

inline constexpr int32_t MAX_SEQUENCES = 256;
inline constexpr int32_t INVALID_SLOT = -1;

struct slot_activate {};
struct slot_deactivate {};
struct router_ready {};

struct slot_storage {
  std::array<uint8_t, MAX_SEQUENCES> active = {};

  void reset() noexcept { active.fill(0); }
};

struct slot_router_model {
  auto operator()() const {
    namespace sml = boost::sml;

    const auto can_activate = [](const slot_storage & storage,
                                 const sml::utility::indexed_event<slot_activate> & ev) {
      return ev.id < storage.active.size() && storage.active[ev.id] == 0;
    };

    const auto do_activate = [](slot_storage & storage,
                                const sml::utility::indexed_event<slot_activate> & ev) {
      storage.active[ev.id] = 1;
    };

    const auto can_deactivate = [](const slot_storage & storage,
                                   const sml::utility::indexed_event<slot_deactivate> & ev) {
      return ev.id < storage.active.size() && storage.active[ev.id] != 0;
    };

    const auto do_deactivate = [](slot_storage & storage,
                                  const sml::utility::indexed_event<slot_deactivate> & ev) {
      storage.active[ev.id] = 0;
    };

    // clang-format off
    return sml::make_transition_table(
      *sml::state<router_ready> + sml::event<sml::utility::indexed_event<slot_activate>>[can_activate] / do_activate,
       sml::state<router_ready> + sml::event<sml::utility::indexed_event<slot_deactivate>>[can_deactivate] / do_deactivate
    );
    // clang-format on
  }
};

using slot_pool = boost::sml::utility::sm_pool<slot_storage, slot_router_model>;

struct context {
  int32_t max_sequences = MAX_SEQUENCES;
  int32_t max_slots = MAX_SEQUENCES;

  slot_pool slots = {};
  std::array<int32_t, MAX_SEQUENCES> seq_to_slot = {};
  std::array<int32_t, MAX_SEQUENCES> slot_owner_seq = {};
  std::array<int32_t, MAX_SEQUENCES> sequence_length = {};

  int32_t phase_error = EMEL_OK;
  bool phase_out_of_memory = false;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::memory::recurrent::action
