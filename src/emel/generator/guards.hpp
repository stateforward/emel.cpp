#pragma once

#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"

namespace emel::generator::guard {

inline constexpr auto valid_generate = [](const event::generate & ev,
                                          const action::context &) noexcept {
  return ev.prompt.data() != nullptr && ev.max_tokens > 0 &&
         ev.max_tokens <= action::MAX_GENERATION_STEPS;
};

inline constexpr auto invalid_generate = [](const event::generate & ev,
                                            const action::context & ctx) noexcept {
  return !valid_generate(ev, ctx);
};

inline constexpr auto phase_ok = [](const action::context & ctx) noexcept {
  return ctx.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & ctx) noexcept {
  return ctx.phase_error != EMEL_OK;
};

inline constexpr auto should_continue_decode = [](const action::context & ctx) noexcept {
  return phase_ok(ctx) && ctx.tokens_generated < ctx.max_tokens;
};

inline constexpr auto stop_condition_met = [](const action::context & ctx) noexcept {
  return phase_ok(ctx) && ctx.tokens_generated >= ctx.max_tokens;
};

}  // namespace emel::generator::guard
