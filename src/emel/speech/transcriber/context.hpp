#pragma once

#include <string_view>

#include "emel/speech/decoder/any.hpp"
#include "emel/speech/encoder/any.hpp"
#include "emel/speech/tokenizer/any.hpp"

namespace emel::speech::transcriber {

// Injected component contracts (text/generator dependencies pattern): pure data
// bound at the variant boundary after model binding. The transcriber engine has
// no model-family knowledge; which variant runs is decided by the kinds, and
// how it runs is decided by the contracts and decode policy the caller injects.
// The default value models "no components injected" and initialize rejects it.
struct dependencies {
  speech::encoder::encoder_kind encoder_kind =
      speech::encoder::encoder_kind::unsupported;
  speech::decoder::decoder_kind decoder_kind =
      speech::decoder::decoder_kind::unsupported;
  speech::tokenizer::tokenizer_kind tokenizer_kind =
      speech::tokenizer::tokenizer_kind::unsupported;
  speech::encoder::execution_contract encoder_contract = {};
  speech::decoder::execution_contract decoder_contract = {};
  speech::tokenizer::asr_decode_policy decode_policy = {};
  std::string_view tokenizer_sha256 = {};
};

} // namespace emel::speech::transcriber

namespace emel::speech::transcriber::action {

// Persistent actor-owned state: the injected dependencies and the component
// actors the sm wrapper owns for the machine's lifetime. The pointers are wired
// once at construction, before any process_event dispatch.
struct context {
  dependencies deps = {};
  speech::encoder::any *encoder = nullptr;
  speech::decoder::any *decoder = nullptr;
  speech::tokenizer::any *tokenizer = nullptr;
};

} // namespace emel::speech::transcriber::action
