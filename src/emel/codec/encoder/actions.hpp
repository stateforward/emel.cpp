#pragma once

namespace emel::codec::encoder::action {

inline constexpr auto tokenize = [](auto&&...) {};
inline constexpr auto dispatch_encoding_done_to_owner = [](auto&&...) {};
inline constexpr auto dispatch_encoding_error_to_owner = [](auto&&...) {};

}  // namespace emel::codec::encoder::action
