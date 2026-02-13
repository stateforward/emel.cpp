#pragma once

namespace emel::model::weight_loader::action {

inline constexpr auto map_weights_mmap = [](const auto &) {};
inline constexpr auto load_weights_streamed = [](const auto &) {};
inline constexpr auto dispatch_loading_done_to_owner = [](const auto &) {};
inline constexpr auto dispatch_loading_error_to_owner = [](const auto &) {};

}  // namespace emel::model::weight_loader::action
