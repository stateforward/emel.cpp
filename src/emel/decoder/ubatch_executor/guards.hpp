#pragma once

#include "emel/decoder/ubatch_executor/actions.hpp"

namespace emel::decoder::ubatch_executor::guard {

inline constexpr auto rollback_required = [](const action::context & ctx) {
  return !ctx.rollback_attempted;
};

}  // namespace emel::decoder::ubatch_executor::guard
