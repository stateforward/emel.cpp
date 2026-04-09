#pragma once

#include <array>
#include <span>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model {

struct architecture {
  std::string_view name = {};
  bool (*load_hparams)(const emel::model::detail::hparam_loader &,
                       emel::model::data &) noexcept = nullptr;
  emel::error::type (*validate_data)(const emel::model::data &) noexcept = nullptr;
};

using architectures = std::span<const architecture>;

extern const std::array<architecture, 4> default_architectures;

architectures default_architecture_span() noexcept;

const architecture * resolve_architecture(std::string_view name,
                                          architectures available_architectures) noexcept;

}  // namespace emel::model
