#pragma once

#include <cstdint>
#include <memory>

#include "emel/emel.h"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"
#include "emel/memory/view.hpp"

namespace emel::memory::hybrid::action {

struct context {
  emel::memory::kv::sm kv = {};
  emel::memory::recurrent::sm recurrent = {};
  std::unique_ptr<emel::memory::view::snapshot> kv_snapshot =
      std::make_unique<emel::memory::view::snapshot>();
  std::unique_ptr<emel::memory::view::snapshot> recurrent_snapshot =
      std::make_unique<emel::memory::view::snapshot>();

  int32_t phase_error = EMEL_OK;
  bool phase_out_of_memory = false;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::memory::hybrid::action
