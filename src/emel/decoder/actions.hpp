#pragma once

namespace emel::decoder::action {

inline constexpr auto detokenize = [](auto&&...) {};
inline constexpr auto dispatch_decoding_done_to_owner = [](auto&&...) {};
inline constexpr auto dispatch_decoding_error_to_owner = [](auto&&...) {};

}  // namespace emel::decoder::action
