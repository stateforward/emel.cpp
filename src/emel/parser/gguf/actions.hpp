#pragma once

#include "emel/parser/gguf/context.hpp"
#include "emel/parser/gguf/events.hpp"

namespace emel::parser::gguf::action {

inline void set_error(context & ctx, int32_t err) noexcept {
  ctx.last_error = err;
}

struct clear_error {
  void operator()(context & ctx) const noexcept { ctx.last_error = EMEL_OK; }
};

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_INVALID_ARGUMENT); }
};

struct run_probe {
  void operator()(const event::probe & ev, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
    ctx.probed_ok = true;
    ctx.probed = {};
    if (ev.requirements_out != nullptr) {
      *ev.requirements_out = ctx.probed;
    }
  }
};

struct run_bind_storage {
  void operator()(const event::bind_storage & ev, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
    ctx.bound_ok = true;
    ctx.kv_arena = ev.kv_arena;
    ctx.kv_arena_size = ev.kv_arena_size;
    ctx.kv_entries = ev.kv_entries;
    ctx.kv_entry_capacity = ev.kv_entry_capacity;
    ctx.tensors = ev.tensors;
    ctx.tensor_capacity = ev.tensor_capacity;
  }
};

struct run_parse {
  void operator()(const event::parse &, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

inline constexpr clear_error clear_error{};
inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr run_probe run_probe{};
inline constexpr run_bind_storage run_bind_storage{};
inline constexpr run_parse run_parse{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::parser::gguf::action
