#pragma once

#include "emel/decoder/actions.hpp"

namespace emel::decoder::guard {

inline constexpr auto has_more_ubatches = [](const action::context & ctx) {
  return ctx.ubatches_processed < ctx.ubatches_total;
};

inline constexpr auto no_more_ubatches = [](const action::context & ctx) {
  return !has_more_ubatches(ctx);
};

inline constexpr auto has_valid_ubatch_size = [](const action::context & ctx) {
  if (ctx.ubatches_processed < 0 || ctx.ubatches_processed >= ctx.ubatches_total) {
    return true;
  }
  return ctx.ubatch_sizes[ctx.ubatches_processed] > 0;
};

inline constexpr auto invalid_ubatch_size = [](const action::context & ctx) {
  return !has_valid_ubatch_size(ctx);
};

inline constexpr auto valid_token_inputs = [](const action::context & ctx) {
  return ctx.n_tokens > 0 && ctx.token_ids != nullptr;
};

inline constexpr auto invalid_token_inputs = [](const action::context & ctx) {
  return !valid_token_inputs(ctx);
};

inline constexpr auto valid_outputs_total = [](const action::context & ctx) {
  return ctx.outputs_total >= 0;
};

inline constexpr auto invalid_outputs_total = [](const action::context & ctx) {
  return !valid_outputs_total(ctx);
};

inline constexpr auto has_valid_ubatch_index = [](const action::context & ctx) {
  return ctx.ubatches_processed >= 0 && ctx.ubatches_processed < ctx.ubatches_total;
};

inline constexpr auto invalid_ubatch_index = [](const action::context & ctx) {
  return !has_valid_ubatch_index(ctx);
};

inline constexpr auto can_process_ubatch = [](const action::context & ctx) {
  return has_valid_ubatch_index(ctx) && ctx.ubatch_sizes[ctx.ubatches_processed] > 0;
};

inline constexpr auto cannot_process_ubatch = [](const action::context & ctx) {
  return !can_process_ubatch(ctx);
};

inline constexpr auto outputs_match = [](const action::context & ctx) {
  return ctx.outputs_processed == ctx.outputs_total;
};

inline constexpr auto outputs_mismatch = [](const action::context & ctx) {
  return !outputs_match(ctx);
};

}  // namespace emel::decoder::guard
