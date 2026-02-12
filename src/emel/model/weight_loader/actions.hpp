#pragma once

namespace emel::model::weight_loader::action {

inline constexpr auto map_weights_mmap = [](auto&&...) {};
inline constexpr auto load_weights_streamed = [](auto&&...) {};
inline constexpr auto dispatch_loading_done_to_owner = [](auto&&...) {};
inline constexpr auto dispatch_loading_error_to_owner = [](auto&&...) {};

}  // namespace emel::model::weight_loader::action
