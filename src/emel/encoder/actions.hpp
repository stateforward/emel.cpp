#pragma once

namespace emel::encoder::action {

inline constexpr auto on_encode_requested = [](auto&&...) {};
inline constexpr auto on_pretokenizing_done = [](auto&&...) {};
inline constexpr auto on_pretokenizing_error = [](auto&&...) {};
inline constexpr auto on_algorithm_step_done = [](auto&&...) {};
inline constexpr auto on_algorithm_step_error = [](auto&&...) {};
inline constexpr auto on_emission_done = [](auto&&...) {};
inline constexpr auto on_emission_error = [](auto&&...) {};
inline constexpr auto on_postrules_done = [](auto&&...) {};
inline constexpr auto on_postrules_error = [](auto&&...) {};
inline constexpr auto dispatch_encoding_done_to_owner = [](auto&&...) {};
inline constexpr auto dispatch_encoding_error_to_owner = [](auto&&...) {};

}  // namespace emel::encoder::action
