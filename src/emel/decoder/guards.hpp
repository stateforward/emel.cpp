#pragma once

#include "emel/decoder/actions.hpp"

namespace emel::decoder::guard {

inline constexpr auto has_more_ubatches = [](const action::context & ctx) {
  return ctx.ubatches_processed < ctx.ubatches_total;
};

inline constexpr auto no_more_ubatches = [](const action::context & ctx) {
  return !has_more_ubatches(ctx);
};

}  // namespace emel::decoder::guard
