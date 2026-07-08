#pragma once

#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"
#include "emel/model/sortformer/any.hpp"

namespace emel::model::sortformer::detail {

using tensor_view = emel::model::sortformer::tensor_view;
using family_view = emel::model::sortformer::family_view;
using execution_contract = emel::model::sortformer::execution_contract;

bool is_execution_architecture(std::string_view architecture) noexcept;

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type build_execution_contract(const emel::model::data & model_data,
                                           execution_contract & contract_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept;

}  // namespace emel::model::sortformer::detail
