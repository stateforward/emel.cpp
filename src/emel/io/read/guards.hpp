#pragma once

#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"

namespace emel::io::read::guard {

// Boundary acceptance predicate for Phase 212. The boundary admits every
// trigger event without inspecting byte_size, file_path, file_index,
// file_offset, layout, target_buffer, or platform; that is Phase 213 work.
// Phase 212 only needs a deterministic forward edge from the request
// decision state to a fail-closed leg.
struct guard_request_accepted {
  bool operator()(const detail::read_tensor_runtime &,
                  const action::context &) const noexcept {
    return true;
  }
};

struct error_callback_present {
  bool operator()(const detail::read_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::read_tensor_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

} // namespace emel::io::read::guard
