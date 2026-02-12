#pragma once

namespace emel::model::loader::action {

inline constexpr auto reset = [](auto&&...) {};
inline constexpr auto map_parser = [](auto&&...) {};
inline constexpr auto parse = [](auto&&...) {};
inline constexpr auto map_layers = [](auto&&...) {};
inline constexpr auto validate_structure = [](auto&&...) {};
inline constexpr auto validate_architecture = [](auto&&...) {};
inline constexpr auto forward_error_event_to_owner = [](auto&&...) {};

}  // namespace emel::model::loader::action
