#pragma once

namespace emel::model::parser::action {

inline constexpr auto parse_architecture = [](auto&&...) {};
inline constexpr auto map_architecture = [](auto&&...) {};
inline constexpr auto parse_hparams = [](auto&&...) {};
inline constexpr auto parse_vocab = [](auto&&...) {};
inline constexpr auto map_tensors = [](auto&&...) {};
inline constexpr auto dispatch_parsing_done_to_owner = [](auto&&...) {};
inline constexpr auto dispatch_parsing_error_to_owner = [](auto&&...) {};

}  // namespace emel::model::parser::action
