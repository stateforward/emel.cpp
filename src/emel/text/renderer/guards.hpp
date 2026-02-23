#pragma once

#include "emel/text/renderer/context.hpp"
#include "emel/text/renderer/events.hpp"

namespace emel::text::renderer::guard {

struct valid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    if (ev.vocab == nullptr || ev.detokenizer_sm == nullptr ||
        ev.dispatch_detokenizer_bind == nullptr ||
        ev.dispatch_detokenizer_detokenize == nullptr) {
      return false;
    }
    if (ev.stop_sequence_count > 0 && ev.stop_sequences == nullptr) {
      return false;
    }
    return true;
  }
};

struct invalid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct valid_render {
  bool operator()(const event::render & ev,
                  const action::context & ctx) const noexcept {
    if (!ctx.is_bound || ctx.vocab == nullptr) {
      return false;
    }
    if (ev.sequence_id < 0 ||
        static_cast<size_t>(ev.sequence_id) >= action::k_max_sequences) {
      return false;
    }
    if (ev.output == nullptr && ev.output_capacity > 0) {
      return false;
    }
    return ev.output_length_out != nullptr && ev.status_out != nullptr &&
           ev.error_out != nullptr;
  }
};

struct invalid_render {
  bool operator()(const event::render & ev,
                  const action::context & ctx) const noexcept {
    return !valid_render{}(ev, ctx);
  }
};

struct valid_flush {
  bool operator()(const event::flush & ev,
                  const action::context & ctx) const noexcept {
    if (!ctx.is_bound || ctx.vocab == nullptr) {
      return false;
    }
    if (ev.sequence_id < 0 ||
        static_cast<size_t>(ev.sequence_id) >= action::k_max_sequences) {
      return false;
    }
    if (ev.output == nullptr && ev.output_capacity > 0) {
      return false;
    }
    return ev.output_length_out != nullptr && ev.status_out != nullptr &&
           ev.error_out != nullptr;
  }
};

struct invalid_flush {
  bool operator()(const event::flush & ev,
                  const action::context & ctx) const noexcept {
    return !valid_flush{}(ev, ctx);
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

}  // namespace emel::text::renderer::guard
