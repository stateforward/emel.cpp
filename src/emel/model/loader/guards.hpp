#pragma once

namespace emel::model::loader::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };
inline constexpr auto has_error = [] { return error(); };
inline constexpr auto has_arch_validate = [] { return false; };
inline constexpr auto no_error_and_has_arch_validate = [] {
  return no_error() && has_arch_validate();
};
inline constexpr auto no_error_and_no_arch_validate = [] {
  return no_error() && !has_arch_validate();
};

}  // namespace emel::model::loader::guard
