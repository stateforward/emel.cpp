#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace emel::jinja {

enum class value_type : uint8_t {
  undefined = 0,
  none = 1,
  boolean = 2,
  integer = 3,
  floating = 4,
  string = 5,
  array = 6,
  object = 7,
  function = 8
};

struct value;
struct object_entry;

struct string_value {
  std::string_view view = {};
  bool is_input = false;
};

struct array_value {
  value * items = nullptr;
  size_t count = 0;
  size_t capacity = 0;
};

struct object_value {
  object_entry * entries = nullptr;
  size_t count = 0;
  size_t capacity = 0;
  bool has_builtins = true;
};

enum class function_kind : uint8_t {
  builtin = 0,
  macro = 1,
  caller = 2
};

struct function_ref {
  function_kind kind = function_kind::builtin;
  const void * data = nullptr;
};

struct value {
  value_type type = value_type::undefined;
  bool bool_v = false;
  int64_t int_v = 0;
  double float_v = 0.0;
  string_value string_v = {};
  array_value array_v = {};
  object_value object_v = {};
  function_ref func_v = {};
  std::string_view hint = {};
};

struct object_entry {
  value key = {};
  value val = {};
};

}  // namespace emel::jinja
