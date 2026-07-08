#pragma once

#include "emel/model/generation/any.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::qwen3::detail {

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

emel::error::type build_generation_contract(
    const emel::model::data & model_data,
    emel::model::generation::contract & contract_out) noexcept;

}  // namespace emel::model::qwen3::detail
