#include "bench_cases.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/errors.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

constexpr size_t k_token_capacity = 4096;
constexpr int32_t k_error_none =
    emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);

int32_t add_token(emel::model::data::vocab & vocab,
                  const char * text,
                  const uint32_t len,
                  float score,
                  int32_t type) {
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

int32_t add_token(emel::model::data::vocab & vocab,
                  const char * text,
                  float score,
                  int32_t type) {
  return add_token(vocab, text, static_cast<uint32_t>(std::strlen(text)), score, type);
}

void add_all_plamo2_byte_tokens(emel::model::data::vocab & vocab) {
  char token[7] = {};
  for (int value = 0; value < 256; ++value) {
    std::snprintf(token, sizeof(token), "<0x%02X>", value);
    (void)add_token(vocab, token, 0.0f, 6);
  }
}

std::unique_ptr<emel::model::data::vocab> make_bpe_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  vocab->ignore_merges = true;

  for (int value = 0; value < 256; ++value) {
    const char byte = static_cast<char>(value);
    (void)add_token(*vocab, &byte, 1, 0.0f, 6);
  }
  (void)add_token(*vocab, "hello", 0.5f, 1);
  (void)add_token(*vocab, "\xC4\xA0" "hello", 0.5f, 1);
  (void)add_token(*vocab, "\xC4\xA0" "world", 0.5f, 1);
  return vocab;
}

std::unique_ptr<emel::model::data::vocab> make_spm_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  vocab->add_space_prefix = true;
  add_all_plamo2_byte_tokens(*vocab);
  (void)add_token(*vocab, "\xE2\x96\x81" "hello", 0.5f, 1);
  (void)add_token(*vocab, "\xE2\x96\x81" "world", 0.5f, 1);
  return vocab;
}

std::unique_ptr<emel::model::data::vocab> make_ugm_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::UGM;
  vocab->add_space_prefix = true;
  const int32_t unk_id = add_token(*vocab, "<unk>", 0.0f, 2);
  vocab->unk_id = unk_id;
  (void)add_token(*vocab, "\xE2\x96\x81" "hello", 0.5f, 1);
  (void)add_token(*vocab, "\xE2\x96\x81" "world", 0.5f, 1);
  return vocab;
}

std::unique_ptr<emel::model::data::vocab> make_wpm_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::WPM;
  const int32_t unk_id = add_token(*vocab, "<unk>", 0.0f, 2);
  vocab->unk_id = unk_id;
  (void)add_token(*vocab, "\xE2\x96\x81" "hello", 0.5f, 1);
  (void)add_token(*vocab, "\xE2\x96\x81" "world", 0.5f, 1);
  return vocab;
}

std::unique_ptr<emel::model::data::vocab> make_rwkv_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::RWKV;
  const int32_t unk_id = add_token(*vocab, "<unk>", 0.0f, 2);
  vocab->unk_id = unk_id;
  (void)add_token(*vocab, "hello", 0.5f, 1);
  (void)add_token(*vocab, "world", 0.5f, 1);
  return vocab;
}

std::unique_ptr<emel::model::data::vocab> make_plamo2_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;
  (void)add_token(*vocab, "<unk>", 0.0f, 2);
  add_all_plamo2_byte_tokens(*vocab);
  (void)add_token(*vocab, "hello", 0.5f, 1);
  (void)add_token(*vocab, "world", 0.5f, 1);
  return vocab;
}

std::string make_repeated_text(const int repeats) {
  std::string out;
  out.reserve(static_cast<size_t>(repeats) * 12);
  for (int i = 0; i < repeats; ++i) {
    if (i > 0) {
      out += ' ';
    }
    out += "hello world";
  }
  return out;
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

bool bind_tokenizer(emel::text::tokenizer::sm & machine,
                    const emel::model::data::vocab & vocab) {
  int32_t err = k_error_none;
  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.preprocessor_variant = preprocessor_kind_for_model(vocab.tokenizer_model_id);
  bind_ev.encoder_variant = encoder_kind_for_model(vocab.tokenizer_model_id);
  bind_ev.error_out = &err;
  if (!machine.process_event(bind_ev) || err != k_error_none) {
    return false;
  }
  return true;
}

bool tokenize_once(emel::text::tokenizer::sm & machine,
                   const emel::model::data::vocab & vocab,
                   const std::string_view text,
                   std::array<int32_t, k_token_capacity> & tokens,
                   int32_t & token_count,
                   int32_t & err) {
  err = k_error_none;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = text;
  tok_ev.add_special = false;
  tok_ev.parse_special = false;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &token_count;
  tok_ev.error_out = &err;
  const bool accepted = machine.process_event(tok_ev);
  return accepted && err == k_error_none;
}

void ensure_tokenizes(emel::text::tokenizer::sm & machine,
                      const emel::model::data::vocab & vocab,
                      const std::string_view text,
                      const char * label) {
  std::array<int32_t, k_token_capacity> tokens = {};
  int32_t token_count = 0;
  int32_t err = k_error_none;
  if (!tokenize_once(machine, vocab, text, tokens, token_count, err)) {
    std::fprintf(stderr,
                 "error: tokenizer failed to process text (%s, err=%d)\n",
                 label,
                 err);
    std::abort();
  }
}

struct tokenizer_case {
  const char * name = nullptr;
  std::unique_ptr<emel::model::data::vocab> (*build_vocab)() = nullptr;
  int short_repeats = 1;
  int long_repeats = 64;
};

}  // namespace

namespace emel::bench {

void append_emel_tokenizer_cases(std::vector<result> & results, const config & cfg) {
  const tokenizer_case cases[] = {
    {"tokenizer/full_bpe", make_bpe_vocab, 1, 64},
    {"tokenizer/full_spm", make_spm_vocab, 1, 64},
    {"tokenizer/full_ugm", make_ugm_vocab, 1, 64},
    {"tokenizer/full_wpm", make_wpm_vocab, 1, 64},
    {"tokenizer/full_rwkv", make_rwkv_vocab, 16, 64},
    {"tokenizer/full_plamo2", make_plamo2_vocab, 1, 64},
  };

  for (const auto & entry : cases) {
    const std::string short_text = make_repeated_text(entry.short_repeats);
    const std::string long_text = make_repeated_text(entry.long_repeats);
    auto vocab = entry.build_vocab();
    emel::text::tokenizer::sm machine{};
    if (!bind_tokenizer(machine, *vocab)) {
      std::fprintf(stderr, "error: tokenizer bind failed\n");
      std::abort();
    }
    ensure_tokenizes(machine, *vocab, short_text, entry.name);
    ensure_tokenizes(machine, *vocab, long_text, entry.name);

    std::array<int32_t, k_token_capacity> tokens = {};
    int32_t token_count = 0;
    int32_t err = k_error_none;
    emel::text::tokenizer::event::tokenize short_ev = {};
    short_ev.vocab = vocab.get();
    short_ev.text = short_text;
    short_ev.add_special = false;
    short_ev.parse_special = false;
    short_ev.token_ids_out = tokens.data();
    short_ev.token_capacity = static_cast<int32_t>(tokens.size());
    short_ev.token_count_out = &token_count;
    short_ev.error_out = &err;

    auto short_fn = [&]() { (void)machine.process_event(short_ev); };
    const std::string short_name = std::string(entry.name) + "_short";
    results.push_back(measure_case(short_name.c_str(), cfg, short_fn));

    emel::text::tokenizer::event::tokenize long_ev = short_ev;
    long_ev.text = long_text;
    auto long_fn = [&]() { (void)machine.process_event(long_ev); };
    const std::string long_name = std::string(entry.name) + "_long";
    results.push_back(measure_case(long_name.c_str(), cfg, long_fn));
  }
}

void append_reference_tokenizer_cases(std::vector<result> & results, const config & cfg) {
  // Reference tokenizer benchmarks reuse the EMEL pipeline until llama.cpp parity is wired.
  append_emel_tokenizer_cases(results, cfg);
}

}  // namespace emel::bench
