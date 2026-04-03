#pragma once

#include <cstdint>

#include "emel/model/architecture/detail.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::detail {

inline constexpr int32_t k_token_type_undefined = 0;
inline constexpr int32_t k_token_type_normal = 1;
inline constexpr int32_t k_token_type_unknown = 2;
inline constexpr int32_t k_token_type_control = 3;

bool load_hparams_from_gguf(const kv_binding & binding,
                            emel::model::architectures available_architectures,
                            emel::model::data & model_out) noexcept;

bool load_hparams_from_gguf(const kv_binding & binding,
                            emel::model::data & model_out) noexcept;

void mark_special_token_type(emel::model::data::vocab & vocab,
                             int32_t token_id,
                             int32_t token_type) noexcept;

bool load_vocab_from_gguf(const kv_binding & binding,
                          emel::model::data::vocab & vocab_out) noexcept;

}  // namespace emel::model::detail
