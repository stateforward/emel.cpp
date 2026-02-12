#pragma once

namespace emel::model::parser::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };
inline constexpr auto has_error = [] { return error(); };

// Kept as explicit parser validation hooks used by step actions to emit _done/_error.
inline constexpr auto arch_supported = [] { return true; };
inline constexpr auto hparams_valid = [] { return true; };

}  // namespace emel::model::parser::guard
