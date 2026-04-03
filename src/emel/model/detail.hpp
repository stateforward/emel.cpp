#pragma once

#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::detail {

inline constexpr int32_t k_token_type_undefined = 0;
inline constexpr int32_t k_token_type_normal = 1;
inline constexpr int32_t k_token_type_unknown = 2;
inline constexpr int32_t k_token_type_control = 3;

bool load_hparams_from_gguf(const kv_binding & binding,
                            emel::model::data & model_out) noexcept;

emel::model::data::tokenizer_model tokenizer_model_from_name(std::string_view name) noexcept;

emel::model::data::tokenizer_pre tokenizer_pre_profile_from_name(
    std::string_view name) noexcept;

void apply_tokenizer_model_defaults(std::string_view name,
                                    emel::model::data::vocab & vocab) noexcept;

void apply_tokenizer_pre_defaults(std::string_view name,
                                  emel::model::data::vocab & vocab) noexcept;

void mark_special_token_type(emel::model::data::vocab & vocab,
                             int32_t token_id,
                             int32_t token_type) noexcept;

bool load_vocab_from_gguf(const kv_binding & binding,
                          emel::model::data::vocab & vocab_out) noexcept;

}  // namespace emel::model::detail
