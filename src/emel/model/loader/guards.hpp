#pragma once

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::guard {

struct no_error {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.err == EMEL_OK;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.err != EMEL_OK;
  }
};

struct has_arch_validate {
  bool operator()(const events::structure_validated & ev) const {
    return ev.request != nullptr && ev.request->validate_architecture;
  }
};

struct no_arch_validate {
  bool operator()(const events::structure_validated & ev) const {
    return ev.request == nullptr ? true : !ev.request->validate_architecture;
  }
};

struct should_load_weights {
  bool operator()(const events::parsing_done & ev) const {
    return ev.request != nullptr && !ev.request->vocab_only;
  }
};

struct skip_weights {
  bool operator()(const events::parsing_done & ev) const {
    return ev.request == nullptr || ev.request->vocab_only;
  }
};

}  // namespace emel::model::loader::guard
