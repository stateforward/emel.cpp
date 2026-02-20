#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"

namespace emel::memory::coordinator::hybrid::action {

namespace event = emel::memory::coordinator::event;

enum class request_kind : int32_t {
  none = 0,
  update = 1,
  batch = 2,
  full = 3,
};

struct context {
  bool has_pending_update = false;
  int32_t update_apply_count = 0;
  int32_t batch_prepare_count = 0;
  int32_t full_prepare_count = 0;

  request_kind active_request = request_kind::none;
  event::prepare_update update_request = {};
  event::prepare_batch batch_request = {};
  event::prepare_full full_request = {};

  event::memory_status prepared_status = event::memory_status::success;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::memory::coordinator::hybrid::action
