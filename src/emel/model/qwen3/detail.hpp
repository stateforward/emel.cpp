#pragma once

#include "emel/model/loader/detail.hpp"

namespace emel::model::qwen3::detail {

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

}  // namespace emel::model::qwen3::detail
