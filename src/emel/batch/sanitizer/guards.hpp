#pragma once

#include "emel/batch/sanitizer/context.hpp"

namespace emel::batch::sanitizer::guard {

struct valid_request {
  bool operator()(const event::sanitize_decode & ev, const action::context &) const noexcept {
    if (ev.error_out == nullptr) {
      return false;
    }
    if (ev.n_tokens <= 0 || ev.token_ids == nullptr) {
      return false;
    }
    if (ev.seq_primary_ids_out == nullptr || ev.seq_primary_ids_capacity < ev.n_tokens) {
      return false;
    }
    if (ev.seq_masks_out == nullptr || ev.seq_masks_capacity < ev.n_tokens) {
      return false;
    }
    if (ev.positions_out == nullptr || ev.positions_capacity < ev.n_tokens) {
      return false;
    }
    if (ev.output_mask_out == nullptr || ev.output_mask_capacity < ev.n_tokens) {
      return false;
    }
    return true;
  }
};

struct invalid_request {
  bool operator()(const event::sanitize_decode & ev, const action::context & ctx) const noexcept {
    return !valid_request{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

}  // namespace emel::batch::sanitizer::guard
