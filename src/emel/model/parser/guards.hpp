#pragma once

#include "emel/emel.h"
#include "emel/model/parser/events.hpp"

namespace emel::model::parser::guard {

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

}  // namespace emel::model::parser::guard
