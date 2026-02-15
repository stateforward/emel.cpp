#pragma once

#include "emel/kv/cache/actions.hpp"

namespace emel::kv::cache::guard {

inline constexpr auto is_prepare = [](const action::context & ctx) {
  return ctx.op == action::operation::prepare;
};

inline constexpr auto is_apply = [](const action::context & ctx) {
  return ctx.op == action::operation::apply;
};

inline constexpr auto is_rollback = [](const action::context & ctx) {
  return ctx.op == action::operation::rollback;
};

}  // namespace emel::kv::cache::guard
