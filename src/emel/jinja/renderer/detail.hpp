#pragma once

#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "emel/emel.h"
#include "emel/jinja/ast.hpp"
#include "emel/jinja/value.hpp"
#include "emel/jinja/renderer/context.hpp"
#include "emel/jinja/renderer/events.hpp"

namespace emel::jinja::renderer::detail {

inline constexpr size_t k_max_call_args = 16;

struct keyword_arg {
  std::string_view key = {};
  emel::jinja::value val = {};
};

struct call_args {
  std::array<emel::jinja::value, k_max_call_args> pos = {};
  size_t pos_count = 0;
  std::array<keyword_arg, k_max_call_args> kw = {};
  size_t kw_count = 0;
};

struct writer_state {
  char * data = nullptr;
  size_t capacity = 0;
  size_t length = 0;
};

struct render_io {
  std::array<writer_state, action::k_max_capture_depth + 1> writers = {};
  size_t depth = 0;
};

enum class control_flow : uint8_t {
  none = 0,
  break_loop = 1,
  continue_loop = 2
};

using builtin_fn = bool (*)(action::context &, const call_args &, emel::jinja::value &) noexcept;
using filter_fn = bool (*)(action::context &, const emel::jinja::value &, const call_args &, emel::jinja::value &) noexcept;
using test_fn = bool (*)(action::context &, const emel::jinja::value &, const call_args &, emel::jinja::value &) noexcept;

struct builtin_entry {
  std::string_view name = {};
  builtin_fn fn = nullptr;
};

struct filter_entry {
  std::string_view name = {};
  filter_fn fn = nullptr;
};

struct test_entry {
  std::string_view name = {};
  test_fn fn = nullptr;
};

inline void set_error(action::context & ctx, const int32_t err, const size_t pos) noexcept {
  if (ctx.phase_error == EMEL_OK) {
    ctx.phase_error = err;
    ctx.last_error = err;
    ctx.error_pos = pos;
  }
}

inline emel::jinja::value make_undefined(std::string_view hint = {}) noexcept {
  emel::jinja::value v;
  v.type = emel::jinja::value_type::undefined;
  v.hint = hint;
  return v;
}

inline emel::jinja::value make_none() noexcept {
  emel::jinja::value v;
  v.type = emel::jinja::value_type::none;
  return v;
}

inline emel::jinja::value make_bool(const bool v) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::boolean;
  out.bool_v = v;
  return out;
}

inline emel::jinja::value make_int(const int64_t v) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::integer;
  out.int_v = v;
  out.float_v = static_cast<double>(v);
  return out;
}

inline emel::jinja::value make_float(const double v) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::floating;
  out.float_v = v;
  out.int_v = static_cast<int64_t>(v);
  return out;
}

inline emel::jinja::value make_string(std::string_view view, const bool is_input = false) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::string;
  out.string_v.view = view;
  out.string_v.is_input = is_input;
  return out;
}

inline emel::jinja::value make_function(const emel::jinja::function_ref ref) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::function;
  out.func_v = ref;
  return out;
}

inline bool ensure_steps(action::context & ctx, const size_t pos) noexcept {
  if (ctx.steps_remaining == 0) {
    set_error(ctx, EMEL_ERR_BACKEND, pos);
    return false;
  }
  ctx.steps_remaining -= 1;
  return true;
}

inline writer_state & current_writer(render_io & io) noexcept {
  return io.writers[io.depth];
}

inline void init_writer(render_io & io, char * output, const size_t capacity) noexcept {
  io.depth = 0;
  io.writers[0].data = output;
  io.writers[0].capacity = capacity;
  io.writers[0].length = 0;
}

inline bool write_text(action::context & ctx, render_io & io, std::string_view text) noexcept {
  writer_state & writer = current_writer(io);
  if (writer.length + text.size() > writer.capacity) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    if (writer.length < writer.capacity && !text.empty()) {
      const size_t avail = writer.capacity - writer.length;
      if (avail > 0) {
        std::memcpy(writer.data + writer.length, text.data(), avail);
        writer.length += avail;
      }
    }
    return false;
  }
  if (!text.empty()) {
    std::memcpy(writer.data + writer.length, text.data(), text.size());
    writer.length += text.size();
  }
  return true;
}

inline std::string_view store_string(action::context & ctx, std::string_view text) noexcept {
  if (text.empty()) {
    return {};
  }
  const size_t remaining = action::k_max_string_bytes - ctx.string_buffer_used;
  if (text.size() > remaining) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return {};
  }
  char * dest = ctx.string_buffer.data() + ctx.string_buffer_used;
  std::memcpy(dest, text.data(), text.size());
  ctx.string_buffer_used += text.size();
  return std::string_view(dest, text.size());
}

inline bool begin_capture(action::context & ctx, render_io & io) noexcept {
  if (io.depth + 1 > action::k_max_capture_depth) {
    set_error(ctx, EMEL_ERR_BACKEND, 0);
    return false;
  }
  const size_t capture_index = io.depth;
  char * base = ctx.capture_buffer.data() + capture_index * action::k_max_capture_bytes;
  io.depth += 1;
  io.writers[io.depth].data = base;
  io.writers[io.depth].capacity = action::k_max_capture_bytes;
  io.writers[io.depth].length = 0;
  return true;
}

inline bool end_capture(action::context & ctx, render_io & io, emel::jinja::value & out) noexcept {
  if (io.depth == 0) {
    set_error(ctx, EMEL_ERR_BACKEND, 0);
    return false;
  }
  writer_state finished = io.writers[io.depth];
  io.depth -= 1;
  const std::string_view stored = store_string(ctx, std::string_view(finished.data, finished.length));
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  out = make_string(stored, false);
  return true;
}

inline emel::jinja::value make_array(action::context & ctx,
                                    const emel::jinja::value * items,
                                    const size_t count) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::array;
  if (count == 0) {
    out.array_v.items = nullptr;
    out.array_v.count = 0;
    out.array_v.capacity = 0;
    return out;
  }
  const size_t remaining = action::k_max_array_items - ctx.array_items_used;
  if (count > remaining) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return make_undefined("array_overflow");
  }
  emel::jinja::value * dest = ctx.array_items.data() + ctx.array_items_used;
  for (size_t i = 0; i < count; ++i) {
    dest[i] = items[i];
  }
  ctx.array_items_used += count;
  out.array_v.items = dest;
  out.array_v.count = count;
  out.array_v.capacity = count;
  return out;
}

inline emel::jinja::value make_object(action::context & ctx,
                                     const emel::jinja::object_entry * entries,
                                     const size_t count,
                                     const size_t capacity,
                                     const bool has_builtins) noexcept {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::object;
  if (capacity == 0) {
    out.object_v.entries = nullptr;
    out.object_v.count = 0;
    out.object_v.capacity = 0;
    out.object_v.has_builtins = has_builtins;
    return out;
  }
  const size_t remaining = action::k_max_object_entries - ctx.object_entries_used;
  if (capacity > remaining) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return make_undefined("object_overflow");
  }
  emel::jinja::object_entry * dest = ctx.object_entries.data() + ctx.object_entries_used;
  for (size_t i = 0; i < count && i < capacity; ++i) {
    dest[i] = entries[i];
  }
  ctx.object_entries_used += capacity;
  out.object_v.entries = dest;
  out.object_v.count = count;
  out.object_v.capacity = capacity;
  out.object_v.has_builtins = has_builtins;
  return out;
}

inline bool push_scope(action::context & ctx) noexcept {
  if (ctx.scope_count >= action::k_max_scopes) {
    set_error(ctx, EMEL_ERR_BACKEND, 0);
    return false;
  }
  const size_t remaining = action::k_max_object_entries - ctx.object_entries_used;
  if (action::k_scope_capacity > remaining) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  action::scope_state & scope = ctx.scopes[ctx.scope_count++];
  emel::jinja::object_entry * entries = ctx.object_entries.data() + ctx.object_entries_used;
  ctx.object_entries_used += action::k_scope_capacity;
  scope.locals.entries = entries;
  scope.locals.count = 0;
  scope.locals.capacity = action::k_scope_capacity;
  scope.locals.has_builtins = false;
  return true;
}

inline void pop_scope(action::context & ctx) noexcept {
  if (ctx.scope_count == 0) {
    return;
  }
  ctx.scope_count -= 1;
}

inline bool value_is_truthy(const emel::jinja::value & v) noexcept {
  switch (v.type) {
    case emel::jinja::value_type::undefined:
    case emel::jinja::value_type::none:
      return false;
    case emel::jinja::value_type::boolean:
      return v.bool_v;
    case emel::jinja::value_type::integer:
      return v.int_v != 0;
    case emel::jinja::value_type::floating:
      return v.float_v != 0.0;
    case emel::jinja::value_type::string:
      return !v.string_v.view.empty();
    case emel::jinja::value_type::array:
      return v.array_v.count > 0;
    case emel::jinja::value_type::object:
      return v.object_v.count > 0;
    case emel::jinja::value_type::function:
      return true;
  }
  return false;
}

inline bool value_is_number(const emel::jinja::value & v) noexcept {
  return v.type == emel::jinja::value_type::integer ||
         v.type == emel::jinja::value_type::floating;
}

inline bool value_equal(const emel::jinja::value & a, const emel::jinja::value & b) noexcept {
  if (a.type != b.type) {
    if (value_is_number(a) && value_is_number(b)) {
      const double da = a.type == emel::jinja::value_type::integer ?
        static_cast<double>(a.int_v) : a.float_v;
      const double db = b.type == emel::jinja::value_type::integer ?
        static_cast<double>(b.int_v) : b.float_v;
      return da == db;
    }
    return false;
  }
  switch (a.type) {
    case emel::jinja::value_type::undefined:
    case emel::jinja::value_type::none:
      return true;
    case emel::jinja::value_type::boolean:
      return a.bool_v == b.bool_v;
    case emel::jinja::value_type::integer:
      return a.int_v == b.int_v;
    case emel::jinja::value_type::floating:
      return a.float_v == b.float_v;
    case emel::jinja::value_type::string:
      return a.string_v.view == b.string_v.view;
    case emel::jinja::value_type::array:
      if (a.array_v.count != b.array_v.count) {
        return false;
      }
      for (size_t i = 0; i < a.array_v.count; ++i) {
        if (!value_equal(a.array_v.items[i], b.array_v.items[i])) {
          return false;
        }
      }
      return true;
    case emel::jinja::value_type::object:
      if (a.object_v.count != b.object_v.count) {
        return false;
      }
      for (size_t i = 0; i < a.object_v.count; ++i) {
        const auto & entry = a.object_v.entries[i];
        bool found = false;
        for (size_t j = 0; j < b.object_v.count; ++j) {
          if (value_equal(entry.key, b.object_v.entries[j].key)) {
            if (!value_equal(entry.val, b.object_v.entries[j].val)) {
              return false;
            }
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    case emel::jinja::value_type::function:
      return a.func_v.data == b.func_v.data && a.func_v.kind == b.func_v.kind;
  }
  return false;
}

inline const emel::jinja::object_entry * find_object_entry(
    const emel::jinja::object_value & object,
    const std::string_view key) noexcept {
  for (size_t i = 0; i < object.count; ++i) {
    const emel::jinja::object_entry & entry = object.entries[i];
    if (entry.key.type == emel::jinja::value_type::string &&
        entry.key.string_v.view == key) {
      return &entry;
    }
  }
  return nullptr;
}

inline emel::jinja::object_entry * find_object_entry_mut(
    emel::jinja::object_value & object,
    const std::string_view key) noexcept {
  for (size_t i = 0; i < object.count; ++i) {
    emel::jinja::object_entry & entry = object.entries[i];
    if (entry.key.type == emel::jinja::value_type::string &&
        entry.key.string_v.view == key) {
      return &entry;
    }
  }
  return nullptr;
}

inline bool set_object_value(action::context & ctx,
                             emel::jinja::object_value & object,
                             const std::string_view key,
                             const emel::jinja::value & val) noexcept {
  emel::jinja::object_entry * existing = find_object_entry_mut(object, key);
  if (existing != nullptr) {
    existing->val = val;
    return true;
  }
  if (object.count >= object.capacity) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  object.entries[object.count].key = make_string(key, false);
  object.entries[object.count].val = val;
  object.count += 1;
  return true;
}

inline emel::jinja::value lookup_identifier(
    action::context & ctx,
    const emel::jinja::object_value * globals,
    const std::string_view name,
    const builtin_entry * builtins,
    const size_t builtin_count) noexcept {
  for (size_t i = ctx.scope_count; i > 0; --i) {
    emel::jinja::object_value & scope_obj = ctx.scopes[i - 1].locals;
    const emel::jinja::object_entry * entry = find_object_entry(scope_obj, name);
    if (entry != nullptr) {
      return entry->val;
    }
  }
  if (globals != nullptr) {
    const emel::jinja::object_entry * entry = find_object_entry(*globals, name);
    if (entry != nullptr) {
      return entry->val;
    }
  }
  for (size_t i = 0; i < builtin_count; ++i) {
    if (builtins[i].name == name) {
      emel::jinja::function_ref ref;
      ref.kind = emel::jinja::function_kind::builtin;
      ref.data = &builtins[i];
      return make_function(ref);
    }
  }
  return make_undefined(name);
}

inline std::string_view value_to_string(action::context & ctx, const emel::jinja::value & v) noexcept {
  if (v.type == emel::jinja::value_type::string) {
    return v.string_v.view;
  }
  if (v.type == emel::jinja::value_type::none || v.type == emel::jinja::value_type::undefined) {
    return {};
  }
  char buffer[64] = {};
  int written = 0;
  switch (v.type) {
    case emel::jinja::value_type::boolean:
      return v.bool_v ? "true" : "false";
    case emel::jinja::value_type::integer:
      written = std::snprintf(buffer, sizeof(buffer), "%lld",
                              static_cast<long long>(v.int_v));
      break;
    case emel::jinja::value_type::floating:
      written = std::snprintf(buffer, sizeof(buffer), "%.15g", v.float_v);
      break;
    case emel::jinja::value_type::array:
    case emel::jinja::value_type::object:
    case emel::jinja::value_type::function:
    case emel::jinja::value_type::undefined:
    case emel::jinja::value_type::none:
    case emel::jinja::value_type::string:
      written = 0;
      break;
  }
  if (written <= 0) {
    return {};
  }
  const std::string_view stored = store_string(ctx, std::string_view(buffer, static_cast<size_t>(written)));
  return stored;
}

inline bool write_value(action::context & ctx, render_io & io, const emel::jinja::value & v) noexcept {
  if (v.type == emel::jinja::value_type::none || v.type == emel::jinja::value_type::undefined) {
    return true;
  }
  if (v.type == emel::jinja::value_type::array) {
    for (size_t i = 0; i < v.array_v.count; ++i) {
      if (!write_value(ctx, io, v.array_v.items[i])) {
        return false;
      }
    }
    return true;
  }
  if (v.type == emel::jinja::value_type::object) {
    return true;
  }
  const std::string_view text = value_to_string(ctx, v);
  return write_text(ctx, io, text);
}

inline bool add_pos(call_args & args, const emel::jinja::value & val) noexcept {
  if (args.pos_count >= k_max_call_args) {
    return false;
  }
  args.pos[args.pos_count++] = val;
  return true;
}

inline bool add_kw(call_args & args, const std::string_view key, const emel::jinja::value & val) noexcept {
  if (args.kw_count >= k_max_call_args) {
    return false;
  }
  args.kw[args.kw_count++] = keyword_arg{key, val};
  return true;
}

inline const emel::jinja::value * find_kw(const call_args & args, std::string_view key) noexcept {
  for (size_t i = 0; i < args.kw_count; ++i) {
    if (args.kw[i].key == key) {
      return &args.kw[i].val;
    }
  }
  return nullptr;
}

inline const emel::jinja::value * get_pos(const call_args & args, const size_t index) noexcept {
  if (index >= args.pos_count) {
    return nullptr;
  }
  return &args.pos[index];
}

inline bool value_is_string(const emel::jinja::value & v) noexcept {
  return v.type == emel::jinja::value_type::string;
}

inline bool value_is_array(const emel::jinja::value & v) noexcept {
  return v.type == emel::jinja::value_type::array;
}

inline bool value_is_object(const emel::jinja::value & v) noexcept {
  return v.type == emel::jinja::value_type::object;
}

inline bool builtin_default(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * val = get_pos(args, 0);
  const emel::jinja::value * def = get_pos(args, 1);
  const emel::jinja::value * use_when_false = get_pos(args, 2);
  if (val == nullptr || def == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  bool apply = (val->type == emel::jinja::value_type::undefined ||
                val->type == emel::jinja::value_type::none);
  if (!apply && use_when_false != nullptr && use_when_false->type == emel::jinja::value_type::boolean) {
    if (use_when_false->bool_v) {
      apply = !value_is_truthy(*val);
    }
  }
  out = apply ? *def : *val;
  return true;
}

inline bool builtin_length(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * val = get_pos(args, 0);
  if (val == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  switch (val->type) {
    case emel::jinja::value_type::string:
      out = make_int(static_cast<int64_t>(val->string_v.view.size()));
      return true;
    case emel::jinja::value_type::array:
      out = make_int(static_cast<int64_t>(val->array_v.count));
      return true;
    case emel::jinja::value_type::object:
      out = make_int(static_cast<int64_t>(val->object_v.count));
      return true;
    default:
      out = make_int(0);
      return true;
  }
}

inline bool builtin_range(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  if (args.pos_count < 1 || args.pos_count > 3) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  int64_t start = 0;
  int64_t stop = 0;
  int64_t step = 1;
  if (args.pos_count == 1) {
    const emel::jinja::value & v0 = args.pos[0];
    if (v0.type != emel::jinja::value_type::integer) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return false;
    }
    stop = v0.int_v;
  } else if (args.pos_count == 2) {
    const emel::jinja::value & v0 = args.pos[0];
    const emel::jinja::value & v1 = args.pos[1];
    if (v0.type != emel::jinja::value_type::integer ||
        v1.type != emel::jinja::value_type::integer) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return false;
    }
    start = v0.int_v;
    stop = v1.int_v;
  } else {
    const emel::jinja::value & v0 = args.pos[0];
    const emel::jinja::value & v1 = args.pos[1];
    const emel::jinja::value & v2 = args.pos[2];
    if (v0.type != emel::jinja::value_type::integer ||
        v1.type != emel::jinja::value_type::integer ||
        v2.type != emel::jinja::value_type::integer) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return false;
    }
    start = v0.int_v;
    stop = v1.int_v;
    step = v2.int_v;
  }
  if (step == 0) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  std::array<emel::jinja::value, action::k_max_array_items> tmp = {};
  size_t count = 0;
  if (step > 0) {
    for (int64_t i = start; i < stop && count < tmp.size(); i += step) {
      tmp[count++] = make_int(i);
    }
  } else {
    for (int64_t i = start; i > stop && count < tmp.size(); i += step) {
      tmp[count++] = make_int(i);
    }
  }
  if (count >= tmp.size()) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  out = make_array(ctx, tmp.data(), count);
  return ctx.phase_error == EMEL_OK;
}

inline bool builtin_namespace(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  if (args.kw_count == 0) {
    emel::jinja::object_entry empty = {};
    out = make_object(ctx, &empty, 0, 0, false);
    return ctx.phase_error == EMEL_OK;
  }
  std::array<emel::jinja::object_entry, k_max_call_args> tmp = {};
  for (size_t i = 0; i < args.kw_count; ++i) {
    tmp[i].key = make_string(args.kw[i].key, false);
    tmp[i].val = args.kw[i].val;
  }
  out = make_object(ctx, tmp.data(), args.kw_count, args.kw_count, false);
  return ctx.phase_error == EMEL_OK;
}

inline bool builtin_upper(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * val = get_pos(args, 0);
  if (val == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  const std::string_view input = value_to_string(ctx, *val);
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  std::array<char, action::k_max_string_bytes> buffer = {};
  if (input.size() > buffer.size()) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = 0; i < input.size(); ++i) {
    buffer[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(input[i])));
  }
  out = make_string(store_string(ctx, std::string_view(buffer.data(), input.size())), false);
  return ctx.phase_error == EMEL_OK;
}

inline bool builtin_lower(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * val = get_pos(args, 0);
  if (val == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  const std::string_view input = value_to_string(ctx, *val);
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  std::array<char, action::k_max_string_bytes> buffer = {};
  if (input.size() > buffer.size()) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = 0; i < input.size(); ++i) {
    buffer[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(input[i])));
  }
  out = make_string(store_string(ctx, std::string_view(buffer.data(), input.size())), false);
  return ctx.phase_error == EMEL_OK;
}

inline bool builtin_trim(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * val = get_pos(args, 0);
  if (val == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  std::string_view input = value_to_string(ctx, *val);
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  size_t start = 0;
  size_t end = input.size();
  while (start < end && std::isspace(static_cast<unsigned char>(input[start]))) {
    ++start;
  }
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  out = make_string(store_string(ctx, input.substr(start, end - start)), false);
  return ctx.phase_error == EMEL_OK;
}

inline bool builtin_replace(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * input_val = get_pos(args, 0);
  const emel::jinja::value * old_val = get_pos(args, 1);
  const emel::jinja::value * new_val = get_pos(args, 2);
  if (input_val == nullptr || old_val == nullptr || new_val == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  const std::string_view input = value_to_string(ctx, *input_val);
  const std::string_view old_str = value_to_string(ctx, *old_val);
  const std::string_view new_str = value_to_string(ctx, *new_val);
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  if (old_str.empty()) {
    out = make_string(store_string(ctx, input), false);
    return ctx.phase_error == EMEL_OK;
  }
  std::array<char, action::k_max_string_bytes> buffer = {};
  size_t offset = 0;
  size_t pos = 0;
  while (pos <= input.size()) {
    const size_t found = input.find(old_str, pos);
    if (found == std::string_view::npos) {
      const size_t tail = input.size() - pos;
      if (offset + tail > buffer.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
        return false;
      }
      std::memcpy(buffer.data() + offset, input.data() + pos, tail);
      offset += tail;
      break;
    }
    const size_t chunk = found - pos;
    if (offset + chunk + new_str.size() > buffer.size()) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return false;
    }
    if (chunk > 0) {
      std::memcpy(buffer.data() + offset, input.data() + pos, chunk);
      offset += chunk;
    }
    if (!new_str.empty()) {
      std::memcpy(buffer.data() + offset, new_str.data(), new_str.size());
      offset += new_str.size();
    }
    pos = found + old_str.size();
  }
  out = make_string(store_string(ctx, std::string_view(buffer.data(), offset)), false);
  return ctx.phase_error == EMEL_OK;
}

inline bool builtin_join(action::context & ctx, const call_args & args, emel::jinja::value & out) noexcept {
  const emel::jinja::value * input = get_pos(args, 0);
  const emel::jinja::value * delim = get_pos(args, 1);
  if (input == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  if (input->type != emel::jinja::value_type::array) {
    out = make_string(value_to_string(ctx, *input), false);
    return ctx.phase_error == EMEL_OK;
  }
  const std::string_view sep = delim != nullptr ? value_to_string(ctx, *delim) : std::string_view(" ");
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  std::array<char, action::k_max_string_bytes> buffer = {};
  size_t offset = 0;
  for (size_t i = 0; i < input->array_v.count; ++i) {
    const std::string_view part = value_to_string(ctx, input->array_v.items[i]);
    if (ctx.phase_error != EMEL_OK) {
      return false;
    }
    if (i > 0 && !sep.empty()) {
      if (offset + sep.size() > buffer.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
        return false;
      }
      std::memcpy(buffer.data() + offset, sep.data(), sep.size());
      offset += sep.size();
    }
    if (offset + part.size() > buffer.size()) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return false;
    }
    if (!part.empty()) {
      std::memcpy(buffer.data() + offset, part.data(), part.size());
      offset += part.size();
    }
  }
  out = make_string(store_string(ctx, std::string_view(buffer.data(), offset)), false);
  return ctx.phase_error == EMEL_OK;
}

inline bool filter_default(action::context & ctx, const emel::jinja::value & input,
                           const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  combined.pos[0] = input;
  if (combined.pos_count == 0) {
    combined.pos_count = 1;
  } else {
    for (size_t i = combined.pos_count; i > 0; --i) {
      combined.pos[i] = combined.pos[i - 1];
    }
    combined.pos[0] = input;
    combined.pos_count += 1;
  }
  return builtin_default(ctx, combined, out);
}

inline bool filter_length(action::context & ctx, const emel::jinja::value & input,
                          const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  if (combined.pos_count >= k_max_call_args) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = combined.pos_count; i > 0; --i) {
    combined.pos[i] = combined.pos[i - 1];
  }
  combined.pos[0] = input;
  combined.pos_count += 1;
  return builtin_length(ctx, combined, out);
}

inline bool filter_upper(action::context & ctx, const emel::jinja::value & input,
                         const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  if (combined.pos_count >= k_max_call_args) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = combined.pos_count; i > 0; --i) {
    combined.pos[i] = combined.pos[i - 1];
  }
  combined.pos[0] = input;
  combined.pos_count += 1;
  return builtin_upper(ctx, combined, out);
}

inline bool filter_lower(action::context & ctx, const emel::jinja::value & input,
                         const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  if (combined.pos_count >= k_max_call_args) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = combined.pos_count; i > 0; --i) {
    combined.pos[i] = combined.pos[i - 1];
  }
  combined.pos[0] = input;
  combined.pos_count += 1;
  return builtin_lower(ctx, combined, out);
}

inline bool filter_trim(action::context & ctx, const emel::jinja::value & input,
                        const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  if (combined.pos_count >= k_max_call_args) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = combined.pos_count; i > 0; --i) {
    combined.pos[i] = combined.pos[i - 1];
  }
  combined.pos[0] = input;
  combined.pos_count += 1;
  return builtin_trim(ctx, combined, out);
}

inline bool filter_replace(action::context & ctx, const emel::jinja::value & input,
                           const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  if (combined.pos_count >= k_max_call_args) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = combined.pos_count; i > 0; --i) {
    combined.pos[i] = combined.pos[i - 1];
  }
  combined.pos[0] = input;
  combined.pos_count += 1;
  return builtin_replace(ctx, combined, out);
}

inline bool filter_join(action::context & ctx, const emel::jinja::value & input,
                        const call_args & args, emel::jinja::value & out) noexcept {
  call_args combined = args;
  if (combined.pos_count >= k_max_call_args) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    return false;
  }
  for (size_t i = combined.pos_count; i > 0; --i) {
    combined.pos[i] = combined.pos[i - 1];
  }
  combined.pos[0] = input;
  combined.pos_count += 1;
  return builtin_join(ctx, combined, out);
}

inline bool test_defined(action::context &, const emel::jinja::value & input,
                         const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type != emel::jinja::value_type::undefined);
  return true;
}

inline bool test_undefined(action::context &, const emel::jinja::value & input,
                           const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::undefined);
  return true;
}

inline bool test_none(action::context &, const emel::jinja::value & input,
                      const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::none);
  return true;
}

inline bool test_string(action::context &, const emel::jinja::value & input,
                        const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::string);
  return true;
}

inline bool test_boolean(action::context &, const emel::jinja::value & input,
                         const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::boolean);
  return true;
}

inline bool test_number(action::context &, const emel::jinja::value & input,
                        const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(value_is_number(input));
  return true;
}

inline bool test_iterable(action::context &, const emel::jinja::value & input,
                          const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::array ||
                  input.type == emel::jinja::value_type::string ||
                  input.type == emel::jinja::value_type::object);
  return true;
}

inline bool test_mapping(action::context &, const emel::jinja::value & input,
                         const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::object);
  return true;
}

inline bool test_even(action::context &, const emel::jinja::value & input,
                      const call_args &, emel::jinja::value & out) noexcept {
  if (input.type != emel::jinja::value_type::integer) {
    out = make_bool(false);
    return true;
  }
  out = make_bool((input.int_v % 2) == 0);
  return true;
}

inline bool test_odd(action::context &, const emel::jinja::value & input,
                     const call_args &, emel::jinja::value & out) noexcept {
  if (input.type != emel::jinja::value_type::integer) {
    out = make_bool(false);
    return true;
  }
  out = make_bool((input.int_v % 2) != 0);
  return true;
}

inline bool test_true(action::context &, const emel::jinja::value & input,
                      const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::boolean && input.bool_v);
  return true;
}

inline bool test_false(action::context &, const emel::jinja::value & input,
                       const call_args &, emel::jinja::value & out) noexcept {
  out = make_bool(input.type == emel::jinja::value_type::boolean && !input.bool_v);
  return true;
}

inline const builtin_entry * builtin_table(size_t & count) noexcept {
  static const std::array<builtin_entry, 8> table = {{
      {"default", builtin_default},
      {"length", builtin_length},
      {"range", builtin_range},
      {"namespace", builtin_namespace},
      {"upper", builtin_upper},
      {"lower", builtin_lower},
      {"trim", builtin_trim},
      {"replace", builtin_replace}
  }};
  count = table.size();
  return table.data();
}

inline const filter_entry * filter_table(size_t & count) noexcept {
  static const std::array<filter_entry, 7> table = {{
      {"default", filter_default},
      {"length", filter_length},
      {"upper", filter_upper},
      {"lower", filter_lower},
      {"trim", filter_trim},
      {"replace", filter_replace},
      {"join", filter_join}
  }};
  count = table.size();
  return table.data();
}

inline const test_entry * test_table(size_t & count) noexcept {
  static const std::array<test_entry, 12> table = {{
      {"defined", test_defined},
      {"undefined", test_undefined},
      {"none", test_none},
      {"string", test_string},
      {"boolean", test_boolean},
      {"number", test_number},
      {"iterable", test_iterable},
      {"mapping", test_mapping},
      {"even", test_even},
      {"odd", test_odd},
      {"true", test_true},
      {"false", test_false}
  }};
  count = table.size();
  return table.data();
}

inline const builtin_entry * find_builtin(std::string_view name) noexcept {
  size_t count = 0;
  const builtin_entry * table = builtin_table(count);
  for (size_t i = 0; i < count; ++i) {
    if (table[i].name == name) {
      return &table[i];
    }
  }
  return nullptr;
}

inline const filter_entry * find_filter(std::string_view name) noexcept {
  size_t count = 0;
  const filter_entry * table = filter_table(count);
  for (size_t i = 0; i < count; ++i) {
    if (table[i].name == name) {
      return &table[i];
    }
  }
  return nullptr;
}

inline const test_entry * find_test(std::string_view name) noexcept {
  size_t count = 0;
  const test_entry * table = test_table(count);
  for (size_t i = 0; i < count; ++i) {
    if (table[i].name == name) {
      return &table[i];
    }
  }
  return nullptr;
}

inline bool collect_call_args(action::context & ctx,
                              const emel::jinja::ast_list & args,
                              call_args & out,
                              const emel::jinja::object_value * globals,
                              render_io & io);

inline emel::jinja::value eval_expr(action::context & ctx,
                                   const emel::jinja::ast_node * node,
                                   const emel::jinja::object_value * globals,
                                   render_io & io);

inline bool render_statements(action::context & ctx,
                              const emel::jinja::ast_list & statements,
                              const emel::jinja::object_value * globals,
                              render_io & io,
                              bool allow_control,
                              control_flow & flow);

inline bool collect_call_args(action::context & ctx,
                              const emel::jinja::ast_list & args,
                              call_args & out,
                              const emel::jinja::object_value * globals,
                              render_io & io) {
  out.pos_count = 0;
  out.kw_count = 0;
  for (const auto & arg : args) {
    if (ctx.phase_error != EMEL_OK) {
      return false;
    }
    if (!arg) {
      continue;
    }
    if (auto * spread = dynamic_cast<emel::jinja::spread_expression *>(arg.get())) {
      emel::jinja::value val = eval_expr(ctx, spread->operand.get(), globals, io);
      if (val.type != emel::jinja::value_type::array) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, spread->pos);
        return false;
      }
      for (size_t i = 0; i < val.array_v.count; ++i) {
        if (!add_pos(out, val.array_v.items[i])) {
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, spread->pos);
          return false;
        }
      }
      continue;
    }
    if (auto * kw = dynamic_cast<emel::jinja::keyword_argument_expression *>(arg.get())) {
      auto * key_id = dynamic_cast<emel::jinja::identifier *>(kw->key.get());
      if (key_id == nullptr) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, kw->pos);
        return false;
      }
      emel::jinja::value val = eval_expr(ctx, kw->value.get(), globals, io);
      if (!add_kw(out, key_id->name, val)) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, kw->pos);
        return false;
      }
      continue;
    }
    emel::jinja::value val = eval_expr(ctx, arg.get(), globals, io);
    if (!add_pos(out, val)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, arg->pos);
      return false;
    }
  }
  return ctx.phase_error == EMEL_OK;
}

inline emel::jinja::value invoke_function(action::context & ctx,
                                          const emel::jinja::function_ref & fn,
                                          const call_args & args,
                                          const emel::jinja::object_value * globals,
                                          render_io & io) noexcept {
  if (fn.kind == emel::jinja::function_kind::builtin) {
    const auto * entry = static_cast<const builtin_entry *>(fn.data);
    if (entry == nullptr || entry->fn == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return make_undefined("builtin_missing");
    }
    emel::jinja::value out = make_undefined();
    if (!entry->fn(ctx, args, out)) {
      return make_undefined("builtin_failed");
    }
    return out;
  }
  if (fn.kind == emel::jinja::function_kind::macro) {
    const auto * slot = static_cast<const action::callable_slot *>(fn.data);
    if (slot == nullptr || slot->macro.stmt == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return make_undefined("macro_missing");
    }
    const emel::jinja::macro_statement * stmt = slot->macro.stmt;
    if (!push_scope(ctx)) {
      return make_undefined("macro_scope");
    }
    size_t arg_index = 0;
    for (size_t i = 0; i < stmt->args.size(); ++i) {
      const auto & param = stmt->args[i];
      auto * param_id = dynamic_cast<emel::jinja::identifier *>(param.get());
      auto * param_kw = dynamic_cast<emel::jinja::keyword_argument_expression *>(param.get());
      std::string_view name = {};
    emel::jinja::value param_value = make_undefined();
      if (param_id != nullptr) {
        name = param_id->name;
      } else if (param_kw != nullptr) {
        auto * key_id = dynamic_cast<emel::jinja::identifier *>(param_kw->key.get());
        if (key_id != nullptr) {
          name = key_id->name;
        }
      }
      if (name.empty()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
        pop_scope(ctx);
        return make_undefined("macro_param");
      }
      const emel::jinja::value * kw_val = find_kw(args, name);
      if (kw_val != nullptr) {
        param_value = *kw_val;
      } else if (arg_index < args.pos_count) {
        param_value = args.pos[arg_index++];
      } else if (param_kw != nullptr) {
        param_value = eval_expr(ctx, param_kw->value.get(), globals, io);
      } else {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
        pop_scope(ctx);
        return make_undefined("macro_args");
      }
      set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, name, param_value);
      if (ctx.phase_error != EMEL_OK) {
        pop_scope(ctx);
        return make_undefined("macro_bind");
      }
    }
    std::array<char, action::k_max_capture_bytes> buffer = {};
    render_io macro_io = {};
    init_writer(macro_io, buffer.data(), buffer.size());
    control_flow flow = control_flow::none;
    bool ok = render_statements(ctx, stmt->body, globals, macro_io, false, flow);
    if (flow != control_flow::none) {
      ok = false;
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    }
    emel::jinja::value out = make_string(
      store_string(ctx, std::string_view(macro_io.writers[0].data, macro_io.writers[0].length)), false);
    pop_scope(ctx);
    if (!ok || ctx.phase_error != EMEL_OK) {
      return make_undefined("macro_failed");
    }
    return out;
  }
  if (fn.kind == emel::jinja::function_kind::caller) {
    const auto * slot = static_cast<const action::callable_slot *>(fn.data);
    if (slot == nullptr || slot->caller.body == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return make_undefined("caller_missing");
    }
    if (!push_scope(ctx)) {
      return make_undefined("caller_scope");
    }
    if (slot->caller.params != nullptr) {
      for (size_t i = 0; i < slot->caller.params->size(); ++i) {
        const auto & param = (*slot->caller.params)[i];
        auto * param_id = dynamic_cast<emel::jinja::identifier *>(param.get());
        if (param_id == nullptr) {
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
          break;
        }
        emel::jinja::value arg_val = i < args.pos_count ? args.pos[i] : make_undefined();
        set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, param_id->name, arg_val);
      }
    }
    std::array<char, action::k_max_capture_bytes> buffer = {};
    render_io caller_io = {};
    init_writer(caller_io, buffer.data(), buffer.size());
    control_flow flow = control_flow::none;
    render_statements(ctx, *slot->caller.body, globals, caller_io, false, flow);
    if (flow != control_flow::none) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
    }
    emel::jinja::value out = make_string(
      store_string(ctx, std::string_view(caller_io.writers[0].data, caller_io.writers[0].length)), false);
    pop_scope(ctx);
    return out;
  }
  set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
  return make_undefined("function_kind");
}

inline emel::jinja::value eval_member(action::context & ctx,
                                     const emel::jinja::member_expression * node,
                                     const emel::jinja::object_value * globals,
                                     render_io & io) {
  emel::jinja::value object = eval_expr(ctx, node->object.get(), globals, io);
  if (object.type == emel::jinja::value_type::undefined) {
    return make_undefined("member_object");
  }
  emel::jinja::value property = make_undefined();
  if (node->computed) {
    if (auto * slice = dynamic_cast<emel::jinja::slice_expression *>(node->property.get())) {
      int64_t start = 0;
      int64_t stop = 0;
      int64_t step = 1;
      if (slice->start) {
        emel::jinja::value start_val = eval_expr(ctx, slice->start.get(), globals, io);
        if (start_val.type == emel::jinja::value_type::integer) {
          start = start_val.int_v;
        }
      }
      if (slice->stop) {
        emel::jinja::value stop_val = eval_expr(ctx, slice->stop.get(), globals, io);
        if (stop_val.type == emel::jinja::value_type::integer) {
          stop = stop_val.int_v;
        }
      } else if (object.type == emel::jinja::value_type::array) {
        stop = static_cast<int64_t>(object.array_v.count);
      } else if (object.type == emel::jinja::value_type::string) {
        stop = static_cast<int64_t>(object.string_v.view.size());
      }
      if (slice->step) {
        emel::jinja::value step_val = eval_expr(ctx, slice->step.get(), globals, io);
        if (step_val.type == emel::jinja::value_type::integer) {
          step = step_val.int_v;
        }
      }
      if (step == 0) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("slice_step");
      }
      if (object.type == emel::jinja::value_type::string) {
        const std::string_view s = object.string_v.view;
        int64_t len = static_cast<int64_t>(s.size());
        auto norm = [len](int64_t v) { return v < 0 ? v + len : v; };
        int64_t s_start = norm(start);
        int64_t s_stop = norm(stop);
        if (s_start < 0) s_start = 0;
        if (s_stop > len) s_stop = len;
        if (s_start > s_stop) s_start = s_stop;
        std::string_view slice_view = s.substr(static_cast<size_t>(s_start),
                                               static_cast<size_t>(s_stop - s_start));
        return make_string(slice_view, object.string_v.is_input);
      }
      if (object.type == emel::jinja::value_type::array) {
        const int64_t len = static_cast<int64_t>(object.array_v.count);
        auto norm = [len](int64_t v) { return v < 0 ? v + len : v; };
        int64_t a_start = norm(start);
        int64_t a_stop = norm(stop);
        if (a_start < 0) a_start = 0;
        if (a_stop > len) a_stop = len;
        if (a_start > a_stop) a_start = a_stop;
        std::array<emel::jinja::value, action::k_max_array_items> tmp = {};
        size_t count = 0;
        for (int64_t i = a_start; i < a_stop && count < tmp.size(); i += step) {
          tmp[count++] = object.array_v.items[i];
        }
        return make_array(ctx, tmp.data(), count);
      }
      return make_undefined("slice_type");
    }
    property = eval_expr(ctx, node->property.get(), globals, io);
  } else {
    auto * prop_id = dynamic_cast<emel::jinja::identifier *>(node->property.get());
    if (prop_id == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("member_prop");
    }
    property = make_string(prop_id->name, false);
  }
  if (object.type == emel::jinja::value_type::array) {
    if (property.type == emel::jinja::value_type::integer) {
      const int64_t idx = property.int_v;
      if (idx >= 0 && static_cast<size_t>(idx) < object.array_v.count) {
        return object.array_v.items[idx];
      }
      return make_undefined("array_index");
    }
    return make_undefined("array_prop");
  }
  if (object.type == emel::jinja::value_type::string) {
    if (property.type == emel::jinja::value_type::integer) {
      const int64_t idx = property.int_v;
      if (idx >= 0 && static_cast<size_t>(idx) < object.string_v.view.size()) {
        const char c = object.string_v.view[static_cast<size_t>(idx)];
        return make_string(std::string_view(&c, 1), object.string_v.is_input);
      }
      return make_undefined("string_index");
    }
    return make_undefined("string_prop");
  }
  if (object.type == emel::jinja::value_type::object) {
    if (property.type == emel::jinja::value_type::string) {
      const emel::jinja::object_entry * entry = find_object_entry(object.object_v, property.string_v.view);
      if (entry != nullptr) {
        return entry->val;
      }
    }
    for (size_t i = 0; i < object.object_v.count; ++i) {
      if (value_equal(object.object_v.entries[i].key, property)) {
        return object.object_v.entries[i].val;
      }
    }
    return make_undefined("object_prop");
  }
  return make_undefined("member_type");
}

inline emel::jinja::value eval_expr(action::context & ctx,
                                   const emel::jinja::ast_node * node,
                                   const emel::jinja::object_value * globals,
                                   render_io & io) {
  if (ctx.phase_error != EMEL_OK) {
    return make_undefined();
  }
  if (node == nullptr) {
    return make_undefined();
  }
  if (auto * literal = dynamic_cast<const emel::jinja::string_literal *>(node)) {
    return make_string(literal->value, false);
  }
  if (auto * ident = dynamic_cast<const emel::jinja::identifier *>(node)) {
    size_t builtin_count = 0;
    const builtin_entry * builtins = builtin_table(builtin_count);
    return lookup_identifier(ctx, globals, ident->name, builtins, builtin_count);
  }
  if (auto * literal = dynamic_cast<const emel::jinja::integer_literal *>(node)) {
    return make_int(literal->value);
  }
  if (auto * literal = dynamic_cast<const emel::jinja::float_literal *>(node)) {
    return make_float(literal->value);
  }
  if (auto * tuple = dynamic_cast<const emel::jinja::tuple_literal *>(node)) {
    std::array<emel::jinja::value, action::k_max_array_items> tmp = {};
    size_t count = 0;
    for (const auto & entry : tuple->values) {
      if (count >= tmp.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("tuple_overflow");
      }
      tmp[count++] = eval_expr(ctx, entry.get(), globals, io);
    }
    return make_array(ctx, tmp.data(), count);
  }
  if (auto * arr = dynamic_cast<const emel::jinja::array_literal *>(node)) {
    std::array<emel::jinja::value, action::k_max_array_items> tmp = {};
    size_t count = 0;
    for (const auto & entry : arr->values) {
      if (count >= tmp.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("array_overflow");
      }
      tmp[count++] = eval_expr(ctx, entry.get(), globals, io);
    }
    return make_array(ctx, tmp.data(), count);
  }
  if (auto * obj = dynamic_cast<const emel::jinja::object_literal *>(node)) {
    std::array<emel::jinja::object_entry, action::k_max_object_entries> tmp = {};
    size_t count = 0;
    for (const auto & entry : obj->pairs) {
      if (count >= tmp.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("object_overflow");
      }
      tmp[count].key = eval_expr(ctx, entry.first.get(), globals, io);
      tmp[count].val = eval_expr(ctx, entry.second.get(), globals, io);
      count += 1;
    }
    return make_object(ctx, tmp.data(), count, count, true);
  }
  if (auto * unary = dynamic_cast<const emel::jinja::unary_expression *>(node)) {
    emel::jinja::value operand = eval_expr(ctx, unary->operand.get(), globals, io);
    if (unary->op.value == "not") {
      return make_bool(!value_is_truthy(operand));
    }
    if (unary->op.value == "+") {
      if (operand.type == emel::jinja::value_type::integer) {
        return operand;
      }
      if (operand.type == emel::jinja::value_type::floating) {
        return operand;
      }
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("unary_plus");
    }
    if (unary->op.value == "-") {
      if (operand.type == emel::jinja::value_type::integer) {
        return make_int(-operand.int_v);
      }
      if (operand.type == emel::jinja::value_type::floating) {
        return make_float(-operand.float_v);
      }
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("unary_minus");
    }
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
    return make_undefined("unary_op");
  }
  if (auto * binary = dynamic_cast<const emel::jinja::binary_expression *>(node)) {
    const std::string_view op = binary->op.value;
    if (op == "or") {
      emel::jinja::value left = eval_expr(ctx, binary->left.get(), globals, io);
      if (value_is_truthy(left)) {
        return left;
      }
      return eval_expr(ctx, binary->right.get(), globals, io);
    }
    if (op == "and") {
      emel::jinja::value left = eval_expr(ctx, binary->left.get(), globals, io);
      if (!value_is_truthy(left)) {
        return left;
      }
      return eval_expr(ctx, binary->right.get(), globals, io);
    }
    emel::jinja::value left = eval_expr(ctx, binary->left.get(), globals, io);
    emel::jinja::value right = eval_expr(ctx, binary->right.get(), globals, io);
    if (op == "+") {
      if (value_is_number(left) && value_is_number(right)) {
        const double lv = left.type == emel::jinja::value_type::integer ?
          static_cast<double>(left.int_v) : left.float_v;
        const double rv = right.type == emel::jinja::value_type::integer ?
          static_cast<double>(right.int_v) : right.float_v;
        if (left.type == emel::jinja::value_type::integer && right.type == emel::jinja::value_type::integer) {
          return make_int(static_cast<int64_t>(lv + rv));
        }
        return make_float(lv + rv);
      }
      const std::string_view l = value_to_string(ctx, left);
      const std::string_view r = value_to_string(ctx, right);
      if (ctx.phase_error != EMEL_OK) {
        return make_undefined("concat");
      }
      std::array<char, action::k_max_string_bytes> buffer = {};
      if (l.size() + r.size() > buffer.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("concat_overflow");
      }
      std::memcpy(buffer.data(), l.data(), l.size());
      std::memcpy(buffer.data() + l.size(), r.data(), r.size());
      return make_string(store_string(ctx, std::string_view(buffer.data(), l.size() + r.size())), false);
    }
    if (op == "-") {
      if (value_is_number(left) && value_is_number(right)) {
        const double lv = left.type == emel::jinja::value_type::integer ?
          static_cast<double>(left.int_v) : left.float_v;
        const double rv = right.type == emel::jinja::value_type::integer ?
          static_cast<double>(right.int_v) : right.float_v;
        if (left.type == emel::jinja::value_type::integer && right.type == emel::jinja::value_type::integer) {
          return make_int(static_cast<int64_t>(lv - rv));
        }
        return make_float(lv - rv);
      }
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("sub");
    }
    if (op == "~") {
      const std::string_view l = value_to_string(ctx, left);
      const std::string_view r = value_to_string(ctx, right);
      if (ctx.phase_error != EMEL_OK) {
        return make_undefined("concat");
      }
      std::array<char, action::k_max_string_bytes> buffer = {};
      if (l.size() + r.size() > buffer.size()) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("concat_overflow");
      }
      std::memcpy(buffer.data(), l.data(), l.size());
      std::memcpy(buffer.data() + l.size(), r.data(), r.size());
      return make_string(store_string(ctx, std::string_view(buffer.data(), l.size() + r.size())), false);
    }
    if (op == "*") {
      if (value_is_number(left) && value_is_number(right)) {
        const double lv = left.type == emel::jinja::value_type::integer ?
          static_cast<double>(left.int_v) : left.float_v;
        const double rv = right.type == emel::jinja::value_type::integer ?
          static_cast<double>(right.int_v) : right.float_v;
        if (left.type == emel::jinja::value_type::integer && right.type == emel::jinja::value_type::integer) {
          return make_int(static_cast<int64_t>(lv * rv));
        }
        return make_float(lv * rv);
      }
      if (left.type == emel::jinja::value_type::string && right.type == emel::jinja::value_type::integer) {
        const int64_t times = right.int_v;
        if (times <= 0) {
          return make_string({}, false);
        }
        const size_t total = left.string_v.view.size() * static_cast<size_t>(times);
        std::array<char, action::k_max_string_bytes> buffer = {};
        if (total > buffer.size()) {
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
          return make_undefined("repeat_overflow");
        }
        size_t offset = 0;
        for (int64_t i = 0; i < times; ++i) {
          std::memcpy(buffer.data() + offset, left.string_v.view.data(), left.string_v.view.size());
          offset += left.string_v.view.size();
        }
        return make_string(store_string(ctx, std::string_view(buffer.data(), total)), false);
      }
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("mul");
    }
    if (op == "/") {
      if (!value_is_number(left) || !value_is_number(right)) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("div");
      }
      const double lv = left.type == emel::jinja::value_type::integer ?
        static_cast<double>(left.int_v) : left.float_v;
      const double rv = right.type == emel::jinja::value_type::integer ?
        static_cast<double>(right.int_v) : right.float_v;
      if (rv == 0.0) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("div_zero");
      }
      return make_float(lv / rv);
    }
    if (op == "%") {
      if (left.type != emel::jinja::value_type::integer ||
          right.type != emel::jinja::value_type::integer) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("mod");
      }
      if (right.int_v == 0) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("mod_zero");
      }
      return make_int(left.int_v % right.int_v);
    }
    if (op == "==") {
      return make_bool(value_equal(left, right));
    }
    if (op == "!=") {
      return make_bool(!value_equal(left, right));
    }
    if (op == "<" || op == "<=" || op == ">" || op == ">=") {
      if (value_is_number(left) && value_is_number(right)) {
        const double lv = left.type == emel::jinja::value_type::integer ?
          static_cast<double>(left.int_v) : left.float_v;
        const double rv = right.type == emel::jinja::value_type::integer ?
          static_cast<double>(right.int_v) : right.float_v;
        if (op == "<") return make_bool(lv < rv);
        if (op == "<=") return make_bool(lv <= rv);
        if (op == ">") return make_bool(lv > rv);
        return make_bool(lv >= rv);
      }
      const std::string_view l = value_to_string(ctx, left);
      const std::string_view r = value_to_string(ctx, right);
      if (op == "<") return make_bool(l < r);
      if (op == "<=") return make_bool(l <= r);
      if (op == ">") return make_bool(l > r);
      return make_bool(l >= r);
    }
    if (op == "in" || op == "not in") {
      bool contains = false;
      if (right.type == emel::jinja::value_type::array) {
        for (size_t i = 0; i < right.array_v.count; ++i) {
          if (value_equal(left, right.array_v.items[i])) {
            contains = true;
            break;
          }
        }
      } else if (right.type == emel::jinja::value_type::object) {
        for (size_t i = 0; i < right.object_v.count; ++i) {
          if (value_equal(left, right.object_v.entries[i].key)) {
            contains = true;
            break;
          }
        }
      } else if (right.type == emel::jinja::value_type::string &&
                 left.type == emel::jinja::value_type::string) {
        contains = right.string_v.view.find(left.string_v.view) != std::string_view::npos;
      } else if (right.type == emel::jinja::value_type::undefined) {
        contains = false;
      }
      if (op == "not in") {
        contains = !contains;
      }
      return make_bool(contains);
    }
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
    return make_undefined("binary_op");
  }
  if (auto * ternary = dynamic_cast<const emel::jinja::ternary_expression *>(node)) {
    emel::jinja::value test = eval_expr(ctx, ternary->test.get(), globals, io);
    if (value_is_truthy(test)) {
      return eval_expr(ctx, ternary->true_expr.get(), globals, io);
    }
    return eval_expr(ctx, ternary->false_expr.get(), globals, io);
  }
  if (auto * select = dynamic_cast<const emel::jinja::select_expression *>(node)) {
    emel::jinja::value test = eval_expr(ctx, select->test.get(), globals, io);
    if (value_is_truthy(test)) {
      return eval_expr(ctx, select->value.get(), globals, io);
    }
    return make_undefined("select");
  }
  if (auto * test_expr = dynamic_cast<const emel::jinja::test_expression *>(node)) {
    emel::jinja::value operand = eval_expr(ctx, test_expr->operand.get(), globals, io);
    std::string_view test_name = {};
    call_args args = {};
    if (auto * id = dynamic_cast<emel::jinja::identifier *>(test_expr->test.get())) {
      test_name = id->name;
    } else if (auto * call = dynamic_cast<emel::jinja::call_expression *>(test_expr->test.get())) {
      auto * id = dynamic_cast<emel::jinja::identifier *>(call->callee.get());
      if (id == nullptr) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("test_name");
      }
      test_name = id->name;
      collect_call_args(ctx, call->args, args, globals, io);
    }
    const test_entry * entry = find_test(test_name);
    if (entry == nullptr || entry->fn == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("test_missing");
    }
    emel::jinja::value result = make_bool(false);
    entry->fn(ctx, operand, args, result);
    if (test_expr->negate) {
      result = make_bool(!value_is_truthy(result));
    }
    return result;
  }
  if (auto * filter_expr = dynamic_cast<const emel::jinja::filter_expression *>(node)) {
    emel::jinja::value operand = eval_expr(ctx, filter_expr->operand.get(), globals, io);
    std::string_view name = {};
    call_args args = {};
    if (auto * id = dynamic_cast<emel::jinja::identifier *>(filter_expr->filter.get())) {
      name = id->name;
    } else if (auto * call = dynamic_cast<emel::jinja::call_expression *>(filter_expr->filter.get())) {
      auto * id = dynamic_cast<emel::jinja::identifier *>(call->callee.get());
      if (id == nullptr) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        return make_undefined("filter_name");
      }
      name = id->name;
      collect_call_args(ctx, call->args, args, globals, io);
    }
    const filter_entry * entry = find_filter(name);
    if (entry == nullptr || entry->fn == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("filter_missing");
    }
    emel::jinja::value out = make_undefined();
    entry->fn(ctx, operand, args, out);
    return out;
  }
  if (auto * call = dynamic_cast<const emel::jinja::call_expression *>(node)) {
    emel::jinja::value callee = eval_expr(ctx, call->callee.get(), globals, io);
    call_args args = {};
    collect_call_args(ctx, call->args, args, globals, io);
    if (callee.type != emel::jinja::value_type::function) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return make_undefined("call_target");
    }
    return invoke_function(ctx, callee.func_v, args, globals, io);
  }
  if (auto * member = dynamic_cast<const emel::jinja::member_expression *>(node)) {
    return eval_member(ctx, member, globals, io);
  }
  return make_undefined("expr_unknown");
}

inline bool render_if(action::context & ctx,
                      const emel::jinja::if_statement * stmt,
                      const emel::jinja::object_value * globals,
                      render_io & io,
                      const bool allow_control,
                      control_flow & flow) {
  emel::jinja::value test = eval_expr(ctx, stmt->test.get(), globals, io);
  const bool truthy = value_is_truthy(test);
  return render_statements(ctx, truthy ? stmt->body : stmt->alternate,
                           globals, io, allow_control, flow);
}

inline bool bind_loop_var(action::context & ctx,
                          const emel::jinja::ast_node * loop_var,
                          const emel::jinja::value & item) {
  if (auto * id = dynamic_cast<const emel::jinja::identifier *>(loop_var)) {
    return set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, id->name, item);
  }
  if (auto * tuple = dynamic_cast<const emel::jinja::tuple_literal *>(loop_var)) {
    if (item.type != emel::jinja::value_type::array) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, loop_var->pos);
      return false;
    }
    if (tuple->values.size() != item.array_v.count) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, loop_var->pos);
      return false;
    }
    for (size_t i = 0; i < tuple->values.size(); ++i) {
      auto * id = dynamic_cast<emel::jinja::identifier *>(tuple->values[i].get());
      if (id == nullptr) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, loop_var->pos);
        return false;
      }
      if (!set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, id->name, item.array_v.items[i])) {
        return false;
      }
    }
    return true;
  }
  set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, loop_var->pos);
  return false;
}

inline bool render_for(action::context & ctx,
                       const emel::jinja::for_statement * stmt,
                       const emel::jinja::object_value * globals,
                       render_io & io,
                       const bool allow_control,
                       control_flow & flow) {
  const emel::jinja::ast_node * iter_expr = stmt->iterable.get();
  const emel::jinja::ast_node * test_expr = nullptr;
  if (auto * select = dynamic_cast<const emel::jinja::select_expression *>(iter_expr)) {
    iter_expr = select->value.get();
    test_expr = select->test.get();
  }
  emel::jinja::value iterable = eval_expr(ctx, iter_expr, globals, io);
  if (iterable.type == emel::jinja::value_type::undefined) {
    iterable = make_array(ctx, nullptr, 0);
  }
  std::array<emel::jinja::value, action::k_max_array_items> items = {};
  size_t item_count = 0;
  if (iterable.type == emel::jinja::value_type::array) {
    for (size_t i = 0; i < iterable.array_v.count && item_count < items.size(); ++i) {
      items[item_count++] = iterable.array_v.items[i];
    }
  } else if (iterable.type == emel::jinja::value_type::object) {
    for (size_t i = 0; i < iterable.object_v.count && item_count < items.size(); ++i) {
      if (auto * tuple = dynamic_cast<const emel::jinja::tuple_literal *>(stmt->loop_var.get())) {
        if (tuple->values.size() >= 2) {
          emel::jinja::value tmp_items[2] = {
              iterable.object_v.entries[i].key,
              iterable.object_v.entries[i].val
          };
          items[item_count++] = make_array(ctx, tmp_items, 2);
          continue;
        }
      }
      items[item_count++] = iterable.object_v.entries[i].key;
    }
  } else {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    return false;
  }
  if (item_count >= items.size()) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    return false;
  }
  std::array<emel::jinja::value, action::k_max_array_items> filtered = {};
  size_t filtered_count = 0;
  for (size_t i = 0; i < item_count && filtered_count < filtered.size(); ++i) {
    if (test_expr == nullptr) {
      filtered[filtered_count++] = items[i];
      continue;
    }
    if (!push_scope(ctx)) {
      return false;
    }
    if (!bind_loop_var(ctx, stmt->loop_var.get(), items[i])) {
      pop_scope(ctx);
      return false;
    }
    emel::jinja::value test_val = eval_expr(ctx, test_expr, globals, io);
    const bool pass = value_is_truthy(test_val);
    pop_scope(ctx);
    if (pass) {
      filtered[filtered_count++] = items[i];
    }
  }
  if (filtered_count == 0) {
    return render_statements(ctx, stmt->alternate, globals, io, allow_control, flow);
  }
  for (size_t i = 0; i < filtered_count; ++i) {
    if (!push_scope(ctx)) {
      return false;
    }
    if (!bind_loop_var(ctx, stmt->loop_var.get(), filtered[i])) {
      pop_scope(ctx);
      return false;
    }
    if (!render_statements(ctx, stmt->body, globals, io, true, flow)) {
      pop_scope(ctx);
      return false;
    }
    pop_scope(ctx);
    if (flow == control_flow::break_loop) {
      flow = control_flow::none;
      break;
    }
    if (flow == control_flow::continue_loop) {
      flow = control_flow::none;
      continue;
    }
  }
  return ctx.phase_error == EMEL_OK;
}

inline bool render_set(action::context & ctx,
                       const emel::jinja::set_statement * stmt,
                       const emel::jinja::object_value * globals,
                       render_io & io) {
  emel::jinja::value rhs = make_undefined();
  if (!stmt->body.empty()) {
    if (!begin_capture(ctx, io)) {
      return false;
    }
    control_flow flow = control_flow::none;
    render_statements(ctx, stmt->body, globals, io, false, flow);
    if (!end_capture(ctx, io, rhs)) {
      return false;
    }
  } else if (stmt->value) {
    rhs = eval_expr(ctx, stmt->value.get(), globals, io);
  }
  if (auto * id = dynamic_cast<const emel::jinja::identifier *>(stmt->left.get())) {
    return set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, id->name, rhs);
  }
  if (auto * tuple = dynamic_cast<const emel::jinja::tuple_literal *>(stmt->left.get())) {
    if (rhs.type != emel::jinja::value_type::array) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
      return false;
    }
    if (tuple->values.size() != rhs.array_v.count) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
      return false;
    }
    for (size_t i = 0; i < tuple->values.size(); ++i) {
      auto * id = dynamic_cast<emel::jinja::identifier *>(tuple->values[i].get());
      if (id == nullptr) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
        return false;
      }
      if (!set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, id->name, rhs.array_v.items[i])) {
        return false;
      }
    }
    return true;
  }
  if (auto * member = dynamic_cast<const emel::jinja::member_expression *>(stmt->left.get())) {
    if (member->computed) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
      return false;
    }
    auto * prop_id = dynamic_cast<emel::jinja::identifier *>(member->property.get());
    if (prop_id == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
      return false;
    }
    emel::jinja::value target = eval_expr(ctx, member->object.get(), globals, io);
    if (target.type != emel::jinja::value_type::object) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
      return false;
    }
    return set_object_value(ctx, target.object_v, prop_id->name, rhs);
  }
  set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
  return false;
}

inline bool render_filter_statement(action::context & ctx,
                                    const emel::jinja::filter_statement * stmt,
                                    const emel::jinja::object_value * globals,
                                    render_io & io) {
  emel::jinja::value body_val = make_undefined();
  if (!begin_capture(ctx, io)) {
    return false;
  }
  control_flow flow = control_flow::none;
  if (!render_statements(ctx, stmt->body, globals, io, false, flow)) {
    return false;
  }
  if (!end_capture(ctx, io, body_val)) {
    return false;
  }
  std::string_view name = {};
  call_args args = {};
  if (auto * id = dynamic_cast<emel::jinja::identifier *>(stmt->filter_node.get())) {
    name = id->name;
  } else if (auto * call = dynamic_cast<emel::jinja::call_expression *>(stmt->filter_node.get())) {
    auto * id = dynamic_cast<emel::jinja::identifier *>(call->callee.get());
    if (id == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
      return false;
    }
    name = id->name;
    collect_call_args(ctx, call->args, args, globals, io);
  }
  const filter_entry * entry = find_filter(name);
  if (entry == nullptr || entry->fn == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    return false;
  }
  emel::jinja::value filtered = make_undefined();
  entry->fn(ctx, body_val, args, filtered);
  return write_value(ctx, io, filtered);
}

inline bool render_call_statement(action::context & ctx,
                                  const emel::jinja::call_statement * stmt,
                                  const emel::jinja::object_value * globals,
                                  render_io & io) {
  auto * call_expr = dynamic_cast<emel::jinja::call_expression *>(stmt->call_expr.get());
  if (call_expr == nullptr) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    return false;
  }
  emel::jinja::value callee = eval_expr(ctx, call_expr->callee.get(), globals, io);
  if (callee.type != emel::jinja::value_type::function) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    return false;
  }
  if (ctx.callable_count >= action::k_max_callables) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt->pos);
    return false;
  }
  action::callable_slot & slot = ctx.callables[ctx.callable_count++];
  slot.kind = emel::jinja::function_kind::caller;
  slot.caller.body = &stmt->body;
  slot.caller.params = &stmt->caller_args;
  emel::jinja::function_ref caller_ref;
  caller_ref.kind = emel::jinja::function_kind::caller;
  caller_ref.data = &slot;
  if (!push_scope(ctx)) {
    return false;
  }
  set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, "caller", make_function(caller_ref));
  call_args args = {};
  collect_call_args(ctx, call_expr->args, args, globals, io);
  emel::jinja::value result = invoke_function(ctx, callee.func_v, args, globals, io);
  pop_scope(ctx);
  return write_value(ctx, io, result);
}

inline bool render_statement(action::context & ctx,
                             const emel::jinja::ast_node * node,
                             const emel::jinja::object_value * globals,
                             render_io & io,
                             const bool allow_control,
                             control_flow & flow) {
  if (ctx.phase_error != EMEL_OK) {
    return false;
  }
  if (!ensure_steps(ctx, node != nullptr ? node->pos : 0)) {
    return false;
  }
  if (node == nullptr) {
    return true;
  }
  if (dynamic_cast<const emel::jinja::comment_statement *>(node) != nullptr) {
    return true;
  }
  if (dynamic_cast<const emel::jinja::noop_statement *>(node) != nullptr) {
    return true;
  }
  if (dynamic_cast<const emel::jinja::break_statement *>(node) != nullptr) {
    flow = control_flow::break_loop;
    return true;
  }
  if (dynamic_cast<const emel::jinja::continue_statement *>(node) != nullptr) {
    flow = control_flow::continue_loop;
    return true;
  }
  if (auto * stmt = dynamic_cast<const emel::jinja::if_statement *>(node)) {
    return render_if(ctx, stmt, globals, io, allow_control, flow);
  }
  if (auto * stmt = dynamic_cast<const emel::jinja::for_statement *>(node)) {
    return render_for(ctx, stmt, globals, io, allow_control, flow);
  }
  if (auto * stmt = dynamic_cast<const emel::jinja::set_statement *>(node)) {
    return render_set(ctx, stmt, globals, io);
  }
  if (auto * stmt = dynamic_cast<const emel::jinja::macro_statement *>(node)) {
    if (ctx.callable_count >= action::k_max_callables) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return false;
    }
    auto * name_id = dynamic_cast<emel::jinja::identifier *>(stmt->name.get());
    if (name_id == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return false;
    }
    action::callable_slot & slot = ctx.callables[ctx.callable_count++];
    slot.kind = emel::jinja::function_kind::macro;
    slot.macro.stmt = stmt;
    emel::jinja::function_ref ref;
    ref.kind = emel::jinja::function_kind::macro;
    ref.data = &slot;
    return set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, name_id->name, make_function(ref));
  }
  if (auto * stmt = dynamic_cast<const emel::jinja::filter_statement *>(node)) {
    return render_filter_statement(ctx, stmt, globals, io);
  }
  if (auto * stmt = dynamic_cast<const emel::jinja::call_statement *>(node)) {
    return render_call_statement(ctx, stmt, globals, io);
  }
  emel::jinja::value val = eval_expr(ctx, node, globals, io);
  return write_value(ctx, io, val);
}

inline bool render_statements(action::context & ctx,
                              const emel::jinja::ast_list & statements,
                              const emel::jinja::object_value * globals,
                              render_io & io,
                              bool allow_control,
                              control_flow & flow) {
  for (const auto & stmt : statements) {
    if (!render_statement(ctx, stmt.get(), globals, io, allow_control, flow)) {
      return false;
    }
    if (flow != control_flow::none) {
      if (!allow_control) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, stmt ? stmt->pos : 0);
        flow = control_flow::none;
        return false;
      }
      return true;
    }
  }
  return ctx.phase_error == EMEL_OK;
}

inline bool run_render(action::context & ctx,
                       const emel::jinja::event::render & ev) noexcept {
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.error_pos = 0;
  ctx.steps_remaining = action::k_max_steps;
  ctx.scope_count = 0;
  ctx.array_items_used = 0;
  ctx.object_entries_used = 0;
  ctx.string_buffer_used = 0;
  ctx.callable_count = 0;

  if (!push_scope(ctx)) {
    return false;
  }

  render_io io = {};
  init_writer(io, ev.output, ev.output_capacity);

  control_flow flow = control_flow::none;
  if (ev.program != nullptr) {
    render_statements(ctx, ev.program->body, ev.globals, io, false, flow);
  } else {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
  }

  if (ev.output_length != nullptr) {
    *ev.output_length = io.writers[0].length;
  }
  if (ev.output_truncated != nullptr) {
    *ev.output_truncated = ctx.phase_error != EMEL_OK;
  }
  if (ev.error_out != nullptr) {
    *ev.error_out = ctx.phase_error;
  }
  if (ev.error_pos_out != nullptr) {
    *ev.error_pos_out = ctx.error_pos;
  }
  return ctx.phase_error == EMEL_OK;
}

}  // namespace emel::jinja::renderer::detail
