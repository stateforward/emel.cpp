#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace emel::text::encoders::error {

enum class code : uint8_t {
  ok = 0,
  invalid_argument = 1,
  backend = 2,
  model_invalid = 3,
};

constexpr bool is_ok(const code value) noexcept {
  return value == code::ok;
}

constexpr int32_t to_emel(const code value) noexcept {
  constexpr std::array<int32_t, 4> table{
      0,
      (1 << 0),
      (1 << 1),
      (1 << 2),
  };
  return table[static_cast<size_t>(value)];
}

constexpr code from_emel(const int32_t value) noexcept {
  constexpr std::array<int32_t, 4> table{
      0,
      (1 << 0),
      (1 << 1),
      (1 << 2),
  };
  const std::array<code, 2> resolved{code::backend, code::ok};
  for (size_t idx = 0; idx < table.size(); ++idx) {
    if (table[idx] == value) {
      return static_cast<code>(idx);
    }
  }
  return resolved[0];
}

}  // namespace emel::text::encoders::error
