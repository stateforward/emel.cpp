#pragma once

#include "emel/model/detail.hpp"

namespace emel::model {

using kv_binding = detail::kv_binding;

inline bool load_hparams_from_gguf(const kv_binding & binding,
                                   emel::model::data & model_out) noexcept {
  return detail::load_hparams_from_gguf(binding, model_out);
}

inline bool load_vocab_from_gguf(const kv_binding & binding,
                                 emel::model::data::vocab & vocab_out) noexcept {
  return detail::load_vocab_from_gguf(binding, vocab_out);
}

}  // namespace emel::model
