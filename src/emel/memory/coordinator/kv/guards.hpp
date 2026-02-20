#pragma once

#include <cstdint>

#include "emel/memory/coordinator/kv/actions.hpp"
#include "emel/memory/coordinator/events.hpp"

namespace emel::memory::coordinator::kv::guard {

namespace event = emel::memory::coordinator::event;

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

struct valid_update_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::update;
  }
};

struct invalid_update_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request != action::request_kind::update;
  }
};

struct valid_batch_context {
  bool operator()(const action::context & ctx) const noexcept {
    if (ctx.active_request != action::request_kind::batch) {
      return false;
    }
    return ctx.batch_request.n_ubatch > 0 && ctx.batch_request.n_ubatches_total > 0;
  }
};

struct invalid_batch_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_batch_context{}(ctx);
  }
};

struct valid_full_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::full;
  }
};

struct invalid_full_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request != action::request_kind::full;
  }
};

struct prepare_update_success {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.prepared_status == event::memory_status::success;
  }
};

struct prepare_update_no_update {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.prepared_status == event::memory_status::no_update;
  }
};

struct prepare_update_invalid_status {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.prepared_status != event::memory_status::success &&
      ctx.prepared_status != event::memory_status::no_update;
  }
};

struct apply_update_ready {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::update &&
      ctx.prepared_status == event::memory_status::success &&
      (ctx.has_pending_update || ctx.update_request.optimize);
  }
};

struct apply_update_backend_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::update &&
      ctx.prepared_status == event::memory_status::success &&
      !ctx.has_pending_update && !ctx.update_request.optimize;
  }
};

struct apply_update_invalid_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request != action::request_kind::update ||
      ctx.prepared_status != event::memory_status::success;
  }
};

struct valid_publish_update_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::update;
  }
};

struct invalid_publish_update_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request != action::request_kind::update;
  }
};

struct valid_publish_batch_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::batch;
  }
};

struct invalid_publish_batch_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request != action::request_kind::batch;
  }
};

struct valid_publish_full_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request == action::request_kind::full;
  }
};

struct invalid_publish_full_context {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.active_request != action::request_kind::full;
  }
};

}  // namespace emel::memory::coordinator::kv::guard
