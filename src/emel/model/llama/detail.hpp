#pragma once

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::llama::detail {

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

}  // namespace emel::model::llama::detail
