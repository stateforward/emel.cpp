#pragma once

#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"

namespace emel::memory::hybrid::action {

struct context {
  emel::memory::kv::sm kv = {};
  emel::memory::recurrent::sm recurrent = {};
};

}  // namespace emel::memory::hybrid::action
