#pragma once

#include "emel/memory/recurrent/actions.hpp"

namespace emel::memory::recurrent::guard {

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
    const int32_t capacity = ctx.reserve_request.slot_capacity;
    return capacity > 0 && capacity <= action::MAX_SEQ;
  }
};

struct invalid_reserve_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_reserve_context{}(ctx);
  }
};

struct valid_allocate_context {
  bool operator()(const action::context &ctx) const noexcept {
    const int32_t seq_id = ctx.allocate_request.seq_id;
    if (seq_id < 0 || seq_id >= action::MAX_SEQ) {
      return false;
    }
    if (!action::has_capacity(ctx)) {
      return false;
    }
    if (action::sequence_exists(ctx, seq_id)) {
      return false;
    }
    return true;
  }
};

struct invalid_allocate_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_allocate_context{}(ctx);
  }
};

struct valid_branch_context {
  bool operator()(const action::context &ctx) const noexcept {
    const int32_t seq_src = ctx.branch_request.seq_id_src;
    const int32_t seq_dst = ctx.branch_request.seq_id_dst;
    if (seq_src < 0 || seq_src >= action::MAX_SEQ || seq_dst < 0 ||
        seq_dst >= action::MAX_SEQ || seq_src == seq_dst) {
      return false;
    }
    if (!action::has_capacity(ctx)) {
      return false;
    }
    if (!action::sequence_exists(ctx, seq_src) ||
        action::sequence_exists(ctx, seq_dst)) {
      return false;
    }
    return true;
  }
};

struct invalid_branch_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_branch_context{}(ctx);
  }
};

struct valid_free_context {
  bool operator()(const action::context &ctx) const noexcept {
    const int32_t seq_id = ctx.free_request.seq_id;
    if (seq_id < 0 || seq_id >= action::MAX_SEQ) {
      return false;
    }
    return action::sequence_exists(ctx, seq_id);
  }
};

struct invalid_free_context {
  bool operator()(const action::context &ctx) const noexcept {
    return !valid_free_context{}(ctx);
  }
};

} // namespace emel::memory::recurrent::guard
