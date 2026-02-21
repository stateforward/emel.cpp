#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/emel.h"
#include "emel/jinja/ast.hpp"
#include "emel/jinja/value.hpp"

namespace emel::jinja::event {
struct render;
}  // namespace emel::jinja::event

namespace emel::jinja::renderer::action {

inline constexpr size_t k_max_scopes = 16;
inline constexpr size_t k_scope_capacity = 128;
inline constexpr size_t k_max_array_items = 2048;
inline constexpr size_t k_max_object_entries = 2048;
inline constexpr size_t k_max_callables = 64;
inline constexpr size_t k_max_string_bytes = 8192;
inline constexpr size_t k_max_capture_depth = 4;
inline constexpr size_t k_max_capture_bytes = 2048;
inline constexpr size_t k_max_steps = 200000;

struct scope_state {
  emel::jinja::object_value locals = {};
};

struct macro_callable {
  const emel::jinja::macro_statement * stmt = nullptr;
};

struct caller_callable {
  const emel::jinja::ast_list * body = nullptr;
  const emel::jinja::ast_list * params = nullptr;
};

struct callable_slot {
  emel::jinja::function_kind kind = emel::jinja::function_kind::builtin;
  macro_callable macro = {};
  caller_callable caller = {};
};

struct context {
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
  size_t error_pos = 0;
  uint32_t steps_remaining = k_max_steps;
  const emel::jinja::event::render * request = nullptr;
  const emel::jinja::object_value * globals = nullptr;
  const emel::jinja::ast_list * statements = nullptr;
  size_t statement_index = 0;
  const emel::jinja::ast_node * pending_expr = nullptr;
  emel::jinja::value pending_value = {};
  bool pending_value_ready = false;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t output_length = 0;
  bool output_truncated = false;

  std::array<scope_state, k_max_scopes> scopes = {};
  size_t scope_count = 0;

  std::array<emel::jinja::value, k_max_array_items> array_items = {};
  size_t array_items_used = 0;

  std::array<emel::jinja::object_entry, k_max_object_entries> object_entries = {};
  size_t object_entries_used = 0;

  std::array<char, k_max_string_bytes> string_buffer = {};
  size_t string_buffer_used = 0;

  std::array<char, k_max_capture_depth * k_max_capture_bytes> capture_buffer = {};

  std::array<callable_slot, k_max_callables> callables = {};
  size_t callable_count = 0;
};

}  // namespace emel::jinja::renderer::action
