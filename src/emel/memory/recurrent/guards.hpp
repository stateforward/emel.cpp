#pragma once

#include "emel/memory/recurrent/context.hpp"

namespace emel::memory::recurrent::guard {

inline constexpr auto phase_ok = [](const action::context & ctx) {
  return ctx.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & ctx) {
  return ctx.phase_error != EMEL_OK && !ctx.phase_out_of_memory;
};

inline constexpr auto phase_out_of_memory = [](const action::context & ctx) {
  return ctx.phase_out_of_memory;
};

}  // namespace emel::memory::recurrent::guard
