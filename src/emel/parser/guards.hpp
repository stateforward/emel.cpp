#pragma once

#include "emel/parser/actions.hpp"
#include "emel/parser/events.hpp"

namespace emel::parser::guard {

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

struct valid_parse_request {
  bool operator()(const event::parse_model & ev) const noexcept {
    return ev.model != nullptr && ev.format_ctx != nullptr;
  }
};

struct invalid_parse_request {
  bool operator()(const event::parse_model & ev) const noexcept {
    return !valid_parse_request{}(ev);
  }
};

struct skip_map_tensors {
  bool operator()(const action::context & ctx) const noexcept {
    return !ctx.request.map_tensors;
  }
};

struct should_map_tensors {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request.map_tensors;
  }
};

struct phase_ok_and_skip_map_tensors {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK && !ctx.request.map_tensors;
  }
};

struct phase_ok_and_map_tensors {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK && ctx.request.map_tensors;
  }
};

}  // namespace emel::parser::guard
