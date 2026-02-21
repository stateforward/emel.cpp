#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/hybrid/sm.hpp"
#include "emel/memory/coordinator/kv/sm.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"
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

  bool process_event(const event::prepare_update & ev) {
    return core_.process_event(ev);
  }

  bool process_event(const event::prepare_batch & ev) {
    return core_.process_event(ev);
  }

  bool process_event(const event::prepare_full & ev) {
    return core_.process_event(ev);
  }

  int32_t last_error() const noexcept {
    int32_t err = EMEL_ERR_BACKEND;
    core_.visit([&](const auto & sm) { err = sm.last_error(); });
    return err;
  }

  event::memory_status last_status() const noexcept {
    event::memory_status status = {};
    core_.visit([&](const auto & sm) { status = sm.last_status(); });
    return status;
  }

 private:
  using sm_list = boost::sml::aux::type_list<recurrent::sm, kv::sm, hybrid::sm>;
  using event_list = boost::sml::aux::type_list<event::prepare_update,
                                                event::prepare_batch,
                                                event::prepare_full>;

  emel::sm_any<coordinator_kind, sm_list, event_list> core_{};
};

}  // namespace emel::memory::coordinator
