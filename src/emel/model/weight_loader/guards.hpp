#pragma once

#include "emel/emel.h"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

struct use_mmap_selected {
  bool operator()(const events::strategy_selected & ev) const {
    return ev.use_mmap;
  }
};

struct use_stream_selected {
  bool operator()(const events::strategy_selected & ev) const {
    return !ev.use_mmap;
  }
};

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

}  // namespace emel::model::weight_loader::guard
