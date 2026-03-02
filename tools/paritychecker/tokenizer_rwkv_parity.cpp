#include "tokenizer_parity.hpp"

namespace emel::paritychecker {

int run_tokenizer_rwkv_parity(const parity_options & opts,
                              const llama_vocab & llama_vocab_ref,
                              const emel::model::data::vocab & emel_vocab) {
  return run_tokenizer_variant_parity(
      opts,
      llama_vocab_ref,
      emel_vocab,
      emel::text::tokenizer::preprocessor::preprocessor_kind::rwkv,
      emel::text::encoders::encoder_kind::rwkv,
      "rwkv");
}

}  // namespace emel::paritychecker
