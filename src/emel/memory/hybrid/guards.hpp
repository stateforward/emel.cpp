#pragma once

#include "emel/memory/hybrid/actions.hpp"

namespace emel::memory::hybrid::guard {

struct phase_ok {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

struct has_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return action::has_capacity(ctx);
  }
};

struct no_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return !action::has_capacity(ctx);
  }
};

struct valid_reserve_context {
  bool operator()(const action::context &ctx) const noexcept {
    const event::reserve &request = ctx.reserve_request;
    if (request.kv_size <= 0 ||
        request.kv_size > emel::memory::kv::action::MAX_KV_CELLS) {
      return false;
    }
    if (request.recurrent_slot_capacity <= 0 ||
        request.recurrent_slot_capacity > action::MAX_SEQ) {
      return false;
    }
    if (request.n_stream <= 0 ||
        request.n_stream > emel::memory::kv::action::MAX_STREAMS) {
      return false;
    }
    return true;
  }
};

struct invalid_reserve_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_reserve_context{}(ctx);
  }
};

struct valid_allocate_context {
  bool operator()(const action::context &ctx) const noexcept {
    const event::allocate_sequence &request = ctx.allocate_request;
    if (request.seq_id < 0 || request.seq_id >= action::MAX_SEQ) {
      return false;
    }
    if (request.slot_count <= 0) {
      return false;
    }
    return !action::has_sequence(ctx, request.seq_id);
  }
};

struct invalid_allocate_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_allocate_context{}(ctx);
  }
};

struct valid_branch_context {
  bool operator()(const action::context &ctx) const noexcept {
    const event::branch_sequence &request = ctx.branch_request;
    if (request.seq_id_src < 0 || request.seq_id_src >= action::MAX_SEQ ||
        request.seq_id_dst < 0 || request.seq_id_dst >= action::MAX_SEQ ||
        request.seq_id_src == request.seq_id_dst) {
      return false;
    }
    if (!action::has_sequence(ctx, request.seq_id_src)) {
      return false;
    }
    return !action::has_sequence(ctx, request.seq_id_dst);
  }
};

struct invalid_branch_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_branch_context{}(ctx);
  }
};

struct valid_free_context {
  bool operator()(const action::context &ctx) const noexcept {
    const event::free_sequence &request = ctx.free_request;
    if (request.seq_id < 0 || request.seq_id >= action::MAX_SEQ) {
      return false;
    }
    return action::has_sequence(ctx, request.seq_id);
  }
};

struct invalid_free_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_free_context{}(ctx);
  }
};

} // namespace emel::memory::hybrid::guard
