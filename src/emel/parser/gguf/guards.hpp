#pragma once

#include "emel/parser/gguf/context.hpp"
#include "emel/parser/gguf/events.hpp"

namespace emel::parser::gguf::guard {

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.last_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.last_error != EMEL_OK;
  }
};

struct valid_probe {
  bool operator()(const event::probe & ev) const noexcept {
    return ev.file_image != nullptr && ev.size > 0;
  }
};

struct invalid_probe {
  bool operator()(const event::probe & ev) const noexcept {
    return !valid_probe{}(ev);
  }
};

struct valid_bind {
  bool operator()(const event::bind_storage & ev) const noexcept {
    return ev.kv_arena != nullptr && ev.kv_arena_size > 0 && ev.tensors != nullptr &&
           ev.tensor_capacity > 0;
  }
};

struct invalid_bind {
  bool operator()(const event::bind_storage & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct valid_parse {
  bool operator()(const event::parse & ev) const noexcept {
    return ev.file_image != nullptr && ev.size > 0;
  }
};

struct invalid_parse {
  bool operator()(const event::parse & ev) const noexcept {
    return !valid_parse{}(ev);
  }
};

}  // namespace emel::parser::gguf::guard
