#pragma once

#include "emel/decoder/compute_executor/actions.hpp"

namespace emel::decoder::compute_executor::guard {

inline constexpr auto has_outputs = [](const action::context & ctx) {
  return ctx.outputs_produced > 0;
};

}  // namespace emel::decoder::compute_executor::guard
