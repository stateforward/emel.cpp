#pragma once

#include <cstdint>

#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/memory/view.hpp"

namespace emel::memory::coordinator::hybrid {

class sm {
 public:
  using model_type = emel::memory::hybrid::sm::model_type;

  sm() = default;

  bool process_event(const event::reserve & ev) { return core_.process_event(ev); }
  bool process_event(const event::allocate_sequence & ev) { return core_.process_event(ev); }
  bool process_event(const event::allocate_slots & ev) { return core_.process_event(ev); }
  bool process_event(const event::branch_sequence & ev) { return core_.process_event(ev); }
  bool process_event(const event::free_sequence & ev) { return core_.process_event(ev); }
  bool process_event(const event::rollback_slots & ev) { return core_.process_event(ev); }

  int32_t last_error() const noexcept { return core_.last_error(); }
  view::any view() const noexcept { return core_.view(); }

 private:
  emel::memory::hybrid::sm core_{};
};

}  // namespace emel::memory::coordinator::hybrid
