#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/actions.hpp"
#include "emel/text/tokenizer/preprocessor/any.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"
#include "emel/text/encoders/any.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

int32_t add_token(emel::model::data::vocab & vocab, const char * text,
                  float score = 0.0f, int32_t type = 0) {
  const uint32_t len = static_cast<uint32_t>(std::strlen(text));
  const uint32_t offset = vocab.token_bytes_used;
  std::memcpy(vocab.token_storage.data() + offset, text, len);
  const uint32_t id = vocab.n_tokens;
  vocab.entries[id].text_offset = offset;
  vocab.entries[id].text_length = len;
  vocab.entries[id].score = score;
  vocab.entries[id].type = type;
  vocab.token_bytes_used += len;
  vocab.n_tokens = id + 1;
  return static_cast<int32_t>(id);
}

void add_all_plamo2_byte_tokens(emel::model::data::vocab & vocab) {
  char token[7] = {};
  for (int value = 0; value < 256; ++value) {
    std::snprintf(token, sizeof(token), "<0x%02X>", value);
    (void)add_token(vocab, token, 0.0f, 6);
  }
}

void init_bpe_vocab(emel::model::data::vocab & vocab) {
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  vocab.ignore_merges = true;
  vocab.add_bos = true;
  vocab.add_eos = true;
  (void)add_token(vocab, "hello");
  (void)add_token(vocab, "\xC4\xA0" "world");
  vocab.bos_id = add_token(vocab, "<bos>");
  vocab.eos_id = add_token(vocab, "<eos>");
}

void init_spm_vocab(emel::model::data::vocab & vocab) {
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  vocab.add_bos = true;
  vocab.add_eos = true;
  (void)add_token(vocab, "a");
  vocab.bos_id = add_token(vocab, "<bos>");
  vocab.eos_id = add_token(vocab, "<eos>");
}

void init_ugm_vocab(emel::model::data::vocab & vocab) {
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::UGM;
  vocab.escape_whitespaces = false;
  vocab.remove_extra_whitespaces = false;
  vocab.treat_whitespace_as_suffix = false;
  vocab.add_space_prefix = false;
  vocab.add_bos = true;
  vocab.add_eos = true;
  vocab.unk_id = add_token(vocab, "<unk>", 0.0f, 2);
  (void)add_token(vocab, "a");
  vocab.bos_id = add_token(vocab, "<bos>");
  vocab.eos_id = add_token(vocab, "<eos>");
}

void init_wpm_vocab(emel::model::data::vocab & vocab) {
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::WPM;
  vocab.add_sep = true;
  vocab.unk_id = add_token(vocab, "<unk>", 0.0f, 2);
  (void)add_token(vocab, "a");
  vocab.sep_id = add_token(vocab, "<sep>");
}

void init_rwkv_vocab(emel::model::data::vocab & vocab) {
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::RWKV;
  vocab.add_eos = true;
  (void)add_token(vocab, "a");
  vocab.eos_id = add_token(vocab, "<eos>");
}

void init_plamo2_vocab(emel::model::data::vocab & vocab) {
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;
  vocab.add_eos = true;
  (void)add_token(vocab, "<unk>", 0.0f, 2);
  add_all_plamo2_byte_tokens(vocab);
  (void)add_token(vocab, "a");
  vocab.eos_id = add_token(vocab, "<eos>");
}

emel::text::encoders::encoder_kind encoder_kind_for_model(
    const emel::model::data::tokenizer_model model) {
  switch (model) {
    case emel::model::data::tokenizer_model::SPM:
      return emel::text::encoders::encoder_kind::spm;
    case emel::model::data::tokenizer_model::BPE:
      return emel::text::encoders::encoder_kind::bpe;
    case emel::model::data::tokenizer_model::WPM:
      return emel::text::encoders::encoder_kind::wpm;
    case emel::model::data::tokenizer_model::UGM:
      return emel::text::encoders::encoder_kind::ugm;
    case emel::model::data::tokenizer_model::RWKV:
      return emel::text::encoders::encoder_kind::rwkv;
    case emel::model::data::tokenizer_model::PLAMO2:
      return emel::text::encoders::encoder_kind::plamo2;
    case emel::model::data::tokenizer_model::NONE:
    case emel::model::data::tokenizer_model::UNKNOWN:
    default:
      return emel::text::encoders::encoder_kind::fallback;
  }
}

emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_kind_for_model(
    const emel::model::data::tokenizer_model model) {
  switch (model) {
    case emel::model::data::tokenizer_model::SPM:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
    case emel::model::data::tokenizer_model::BPE:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
    case emel::model::data::tokenizer_model::WPM:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
    case emel::model::data::tokenizer_model::UGM:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::ugm;
    case emel::model::data::tokenizer_model::RWKV:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::rwkv;
    case emel::model::data::tokenizer_model::PLAMO2:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::plamo2;
    case emel::model::data::tokenizer_model::NONE:
    case emel::model::data::tokenizer_model::UNKNOWN:
    default:
      return emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  }
}

bool reference_tokenize(const emel::model::data::vocab & vocab,
                        const std::string_view text,
                        const bool add_special,
                        const bool parse_special,
                        int32_t * token_ids,
                        const int32_t token_capacity,
                        int32_t & token_count,
                        int32_t & err) {
  token_count = 0;
  err = EMEL_OK;

  emel::text::tokenizer::preprocessor::any preprocessor;
  preprocessor.set_kind(preprocessor_kind_for_model(vocab.tokenizer_model_id));

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments> fragments = {};
  size_t fragment_count = 0;
  bool preprocessed = false;
  emel::text::tokenizer::preprocessor::event::preprocess pre_ev(
      vocab, text, parse_special,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments),
      fragment_count, err);
  pre_ev.preprocessed_out = &preprocessed;
  if (!preprocessor.process_event(pre_ev) || err != EMEL_OK) {
    return false;
  }

  auto push_token = [&](const int32_t token) -> bool {
    if (token < 0 || token_ids == nullptr) {
      err = EMEL_ERR_INVALID_ARGUMENT;
      return false;
    }
    if (token_count >= token_capacity) {
      err = EMEL_ERR_INVALID_ARGUMENT;
      return false;
    }
    token_ids[token_count++] = token;
    return true;
  };

  if (add_special && vocab.add_bos) {
    if (vocab.bos_id < 0 || !push_token(vocab.bos_id)) {
      return false;
    }
  }

  emel::text::encoders::any encoder;
  encoder.set_kind(encoder_kind_for_model(vocab.tokenizer_model_id));

  for (size_t idx = 0; idx < fragment_count; ++idx) {
    const auto & frag = fragments[idx];
    if (frag.kind == emel::text::tokenizer::preprocessor::fragment_kind::token) {
      if (!push_token(frag.token)) {
        return false;
      }
      continue;
    }
    if (frag.text.empty()) {
      continue;
    }
    int32_t fragment_tokens = 0;
    emel::text::encoders::event::encode enc_ev = {};
    enc_ev.vocab = &vocab;
    enc_ev.text = frag.text;
    enc_ev.preprocessed = preprocessed;
    enc_ev.token_ids = token_ids + token_count;
    enc_ev.token_capacity = token_capacity - token_count;
    enc_ev.token_count_out = &fragment_tokens;
    enc_ev.error_out = &err;
    if (!encoder.process_event(enc_ev) || err != EMEL_OK) {
      return false;
    }
    token_count += fragment_tokens;
  }

  if (add_special) {
    if (vocab.tokenizer_model_id == emel::model::data::tokenizer_model::WPM &&
        vocab.add_sep) {
      if (vocab.sep_id < 0 || !push_token(vocab.sep_id)) {
        return false;
      }
    } else if (vocab.tokenizer_model_id != emel::model::data::tokenizer_model::WPM &&
               vocab.add_eos) {
      if (vocab.eos_id < 0 || !push_token(vocab.eos_id)) {
        return false;
      }
    }
  }

  return err == EMEL_OK;
}

void run_parity_case(const emel::model::data::vocab & vocab,
                     const std::string_view text,
                     const bool add_special,
                     const bool parse_special) {
  emel::text::tokenizer::sm machine{};
  int32_t bind_err = EMEL_OK;
  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.preprocessor_variant = preprocessor_kind_for_model(vocab.tokenizer_model_id);
  bind_ev.encoder_variant = encoder_kind_for_model(vocab.tokenizer_model_id);
  bind_ev.error_out = &bind_err;
  REQUIRE(machine.process_event(bind_ev));
  REQUIRE(bind_err == EMEL_OK);

  std::array<int32_t, 32> tokens = {};
  int32_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = text;
  tok_ev.add_special = add_special;
  tok_ev.parse_special = parse_special;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &count;
  tok_ev.error_out = &err;
  REQUIRE(machine.process_event(tok_ev));
  REQUIRE(err == EMEL_OK);

  std::array<int32_t, 32> reference_tokens = {};
  int32_t reference_count = 0;
  int32_t reference_err = EMEL_OK;
  REQUIRE(reference_tokenize(vocab, text, add_special, parse_special,
                             reference_tokens.data(),
                             static_cast<int32_t>(reference_tokens.size()),
                             reference_count, reference_err));
  REQUIRE(reference_err == EMEL_OK);
  REQUIRE(reference_count == count);
  for (int32_t idx = 0; idx < count; ++idx) {
    CHECK(reference_tokens[static_cast<size_t>(idx)] ==
          tokens[static_cast<size_t>(idx)]);
  }
}

}  // namespace

TEST_CASE("tokenizer_parity_basic_models") {
  auto bpe_vocab = std::make_unique<emel::model::data::vocab>();
  init_bpe_vocab(*bpe_vocab);
  run_parity_case(*bpe_vocab, "hello world", true, false);

  auto spm_vocab = std::make_unique<emel::model::data::vocab>();
  init_spm_vocab(*spm_vocab);
  run_parity_case(*spm_vocab, "a", true, false);

  auto ugm_vocab = std::make_unique<emel::model::data::vocab>();
  init_ugm_vocab(*ugm_vocab);
  run_parity_case(*ugm_vocab, "a", true, false);

  auto wpm_vocab = std::make_unique<emel::model::data::vocab>();
  init_wpm_vocab(*wpm_vocab);
  run_parity_case(*wpm_vocab, "a", true, false);

  auto rwkv_vocab = std::make_unique<emel::model::data::vocab>();
  init_rwkv_vocab(*rwkv_vocab);
  run_parity_case(*rwkv_vocab, "a", true, false);

  auto plamo2_vocab = std::make_unique<emel::model::data::vocab>();
  init_plamo2_vocab(*plamo2_vocab);
  run_parity_case(*plamo2_vocab, "a", true, false);
}
