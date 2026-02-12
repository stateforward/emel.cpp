#pragma once

namespace emel::generator::action {

inline constexpr auto initialize = [](auto&&...) {};
inline constexpr auto tokenize_prompt = [](auto&&...) {};
inline constexpr auto run_prefill = [](auto&&...) {};
inline constexpr auto run_decode_step = [](auto&&...) {};
inline constexpr auto dispatch_generation_done_to_owner = [](auto&&...) {};
inline constexpr auto dispatch_generation_error_to_owner = [](auto&&...) {};

}  // namespace emel::generator::action
