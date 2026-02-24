#pragma once

#include <cstdint>

#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/view.hpp"

namespace emel::memory::coordinator::kv {

class sm {
 public:
  using model_type = emel::memory::kv::sm::model_type;

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
  emel::memory::kv::sm core_{};
};

}  // namespace emel::memory::coordinator::kv
