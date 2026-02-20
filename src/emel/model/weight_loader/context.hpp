#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::action {

struct context {
  const event::load_weights * request = nullptr;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  bool use_mmap = false;
  bool use_direct_io = false;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::model::weight_loader::action
