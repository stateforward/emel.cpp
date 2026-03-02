#pragma once

#include "parity_runner.hpp"

#include "emel/model/data.hpp"
#include "emel/text/encoders/any.hpp"
#include "emel/text/tokenizer/preprocessor/any.hpp"

struct llama_vocab;

namespace emel::paritychecker {

int run_tokenizer_variant_parity(
    const parity_options & opts,
    const llama_vocab & llama_vocab_ref,
    const emel::model::data::vocab & emel_vocab,
    emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant,
    emel::text::encoders::encoder_kind encoder_variant,
    const char * variant_name);

int run_tokenizer_spm_parity(const parity_options & opts,
                             const llama_vocab & llama_vocab_ref,
                             const emel::model::data::vocab & emel_vocab);

int run_tokenizer_bpe_parity(const parity_options & opts,
                             const llama_vocab & llama_vocab_ref,
                             const emel::model::data::vocab & emel_vocab);

int run_tokenizer_wpm_parity(const parity_options & opts,
                             const llama_vocab & llama_vocab_ref,
                             const emel::model::data::vocab & emel_vocab);

int run_tokenizer_ugm_parity(const parity_options & opts,
                             const llama_vocab & llama_vocab_ref,
                             const emel::model::data::vocab & emel_vocab);

int run_tokenizer_rwkv_parity(const parity_options & opts,
                              const llama_vocab & llama_vocab_ref,
                              const emel::model::data::vocab & emel_vocab);

int run_tokenizer_plamo2_parity(const parity_options & opts,
                                const llama_vocab & llama_vocab_ref,
                                const emel::model::data::vocab & emel_vocab);

int run_tokenizer_fallback_parity(const parity_options & opts,
                                  const llama_vocab & llama_vocab_ref,
                                  const emel::model::data::vocab & emel_vocab);

}  // namespace emel::paritychecker
