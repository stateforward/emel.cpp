#pragma once

#include <cstdint>

#include "emel/memory/kv/sm.hpp"

namespace emel::memory::test {

struct recording_kv_actor {
  emel::memory::kv::sm delegate = {};
  int32_t reserve_count = 0;
  int32_t allocate_sequence_count = 0;
  int32_t allocate_slots_count = 0;
  int32_t branch_sequence_count = 0;
  int32_t free_sequence_count = 0;
  int32_t rollback_slots_count = 0;
  int32_t capture_view_count = 0;

  bool process_event(const emel::memory::event::reserve &ev) {
    ++reserve_count;
    return delegate.process_event(ev);
  }

  bool process_event(const emel::memory::event::allocate_sequence &ev) {
    ++allocate_sequence_count;
    return delegate.process_event(ev);
  }

  bool process_event(const emel::memory::event::allocate_slots &ev) {
    ++allocate_slots_count;
    return delegate.process_event(ev);
  }

  bool process_event(const emel::memory::event::branch_sequence &ev) {
    ++branch_sequence_count;
    return delegate.process_event(ev);
  }

  bool process_event(const emel::memory::event::free_sequence &ev) {
    ++free_sequence_count;
    return delegate.process_event(ev);
  }

  bool process_event(const emel::memory::event::rollback_slots &ev) {
    ++rollback_slots_count;
    return delegate.process_event(ev);
  }

  bool process_event(const emel::memory::event::capture_view &ev) {
    ++capture_view_count;
    return delegate.process_event(ev);
  }
};

} // namespace emel::memory::test
