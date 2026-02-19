#pragma once

#include "emel/decoder/ubatch_executor/actions.hpp"
#include "emel/kv/cache/actions.hpp"

namespace emel::decoder::ubatch_executor::guard {

inline constexpr auto phase_ok = [](const action::context & ctx) {
  return ctx.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & ctx) {
  return ctx.phase_error != EMEL_OK;
};

inline constexpr auto outputs_produced_invalid = [](const action::context & ctx) {
  return ctx.phase_error == EMEL_OK && ctx.outputs_produced != ctx.expected_outputs;
};

inline constexpr auto always = [](const action::context &) {
  return true;
};

struct valid_execute_request {
  bool operator()(const event::execute & ev, const action::context &) const noexcept {
    if (ev.memory_coordinator_sm == nullptr || ev.kv_cache_sm == nullptr) {
      return false;
    }
    if (ev.ubatch_index < 0 || ev.ubatch_size <= 0) {
      return false;
    }
    if (ev.expected_outputs < 0 || ev.expected_outputs > ev.ubatch_size) {
      return false;
    }
    if (ev.positions != nullptr) {
      if (ev.positions_count < ev.ubatch_size) {
        return false;
      }
      if (ev.positions_count > ev.ubatch_size &&
          ev.positions_count < ev.ubatch_size * 3) {
        return false;
      }
    }
    if (ev.seq_masks != nullptr) {
      if (ev.seq_mask_words <= 0 ||
          ev.seq_mask_words > emel::kv::cache::action::SEQ_WORDS) {
        return false;
      }
      if (ev.seq_masks_count < ev.ubatch_size) {
        return false;
      }
    }
    if (ev.seq_primary_ids != nullptr && ev.seq_primary_ids_count < ev.ubatch_size) {
      return false;
    }
    return true;
  }
};

struct invalid_execute_request {
  bool operator()(const event::execute & ev, const action::context & ctx) const noexcept {
    return !valid_execute_request{}(ev, ctx);
  }
};

}  // namespace emel::decoder::ubatch_executor::guard
