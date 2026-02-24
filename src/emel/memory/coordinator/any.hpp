#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/hybrid/sm.hpp"
#include "emel/memory/coordinator/kv/sm.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::coordinator {

enum class coordinator_kind : uint8_t {
  recurrent = 0,
  kv = 1,
  hybrid = 2,
};

class any {
 public:
  any() = default;
  explicit any(const coordinator_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const coordinator_kind kind) { core_.set_kind(kind); }

  coordinator_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::reserve & ev) { return core_.process_event(ev); }
  bool process_event(const event::allocate_sequence & ev) { return core_.process_event(ev); }
  bool process_event(const event::allocate_slots & ev) { return core_.process_event(ev); }
  bool process_event(const event::branch_sequence & ev) { return core_.process_event(ev); }
  bool process_event(const event::free_sequence & ev) { return core_.process_event(ev); }
  bool process_event(const event::rollback_slots & ev) { return core_.process_event(ev); }

  int32_t last_error() const noexcept {
    int32_t err = EMEL_ERR_BACKEND;
    core_.visit([&](const auto & sm) { err = sm.last_error(); });
    return err;
  }

  view::any view() const noexcept {
    view::any result = {};
    core_.visit([&](const auto & sm) { result = sm.view(); });
    return result;
  }

 private:
  using sm_list = boost::sml::aux::type_list<recurrent::sm, kv::sm, hybrid::sm>;
  using event_list = boost::sml::aux::type_list<event::reserve,
                                                event::allocate_sequence,
                                                event::allocate_slots,
                                                event::branch_sequence,
                                                event::free_sequence,
                                                event::rollback_slots>;

  emel::sm_any<coordinator_kind, sm_list, event_list> core_{};
};

}  // namespace emel::memory::coordinator
