#pragma once

#include "emel/model/builder/detail.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::gemma4::detail {

bool is_execution_architecture(std::string_view architecture) noexcept;

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

emel::error::type validate_builder_contract(const emel::model::data & model_data) noexcept;

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept;

emel::error::type build_view(const emel::model::data & model_data,
                             emel::model::builder::detail::view & view_out) noexcept;

}  // namespace emel::model::gemma4::detail
