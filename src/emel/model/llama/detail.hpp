#pragma once

#include "emel/model/builder/detail.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::llama::detail {

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

emel::error::type build_view(const emel::model::data & model_data,
                             emel::model::builder::detail::view & view_out) noexcept;

emel::error::type build_execution_view(
    const emel::model::data & model_data,
    emel::model::builder::detail::execution_view & view_out) noexcept;

}  // namespace emel::model::llama::detail
