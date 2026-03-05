#pragma once

#include "../../bench_common.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/unicode.hpp"

namespace emel::bench::encoder_bench {

constexpr size_t k_token_capacity = 4096;

enum class bench_error : int32_t {
  none = 0,
  invalid_argument = (1 << 0),
  backend = (1 << 1),
  model_invalid = (1 << 2),
};

constexpr int32_t error_code(const bench_error code) noexcept {
  return static_cast<int32_t>(code);
}

constexpr int32_t k_error_none = error_code(bench_error::none);

inline int32_t add_token(emel::model::data::vocab & vocab,
                         const char * text,
                         const uint32_t len,
                         const float score,
                         const int32_t type) {
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

inline int32_t add_token(emel::model::data::vocab & vocab,
                         const char * text,
                         const float score,
                         const int32_t type) {
  return add_token(vocab, text, static_cast<uint32_t>(std::strlen(text)), score, type);
}

inline int32_t add_byte_token(emel::model::data::vocab & vocab, const uint8_t value) {
  const std::string token = emel::text::unicode_byte_to_utf8(value);
  return add_token(vocab,
                   token.c_str(),
                   static_cast<uint32_t>(token.size()),
                   0.0f,
                   6);
}

inline int32_t add_raw_byte_token(emel::model::data::vocab & vocab, const uint8_t value) {
  const char byte = static_cast<char>(value);
  return add_token(vocab, &byte, 1u, 0.0f, 6);
}

inline int32_t add_plamo2_byte_token(emel::model::data::vocab & vocab, const uint8_t value) {
  char token[7] = {};
  std::snprintf(token, sizeof(token), "<0x%02X>", value);
  return add_token(vocab, token, 0.0f, 6);
}

inline void add_all_byte_tokens(emel::model::data::vocab & vocab) {
  for (int value = 0; value < 256; ++value) {
    (void)add_byte_token(vocab, static_cast<uint8_t>(value));
  }
}

inline void add_all_raw_byte_tokens(emel::model::data::vocab & vocab) {
  for (int value = 0; value < 256; ++value) {
    (void)add_raw_byte_token(vocab, static_cast<uint8_t>(value));
  }
}

inline void add_all_plamo2_byte_tokens(emel::model::data::vocab & vocab) {
  for (int value = 0; value < 256; ++value) {
    (void)add_plamo2_byte_token(vocab, static_cast<uint8_t>(value));
  }
}

inline std::string make_repeated_text(const int repeats) {
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

template <class machine_type>
inline bool run_encode(machine_type & machine,
                       emel::text::encoders::event::encode & request,
                       int32_t & token_count,
                       int32_t & err) {
  token_count = 0;
  err = k_error_none;
  const bool accepted = machine.process_event(request);
  return accepted && err == k_error_none;
}

template <class machine_type>
inline void ensure_encodes(machine_type & machine,
                           emel::text::encoders::event::encode & request,
                           const char * label) {
  int32_t token_count = 0;
  int32_t err = k_error_none;
  if (!run_encode(machine, request, token_count, err)) {
    std::fprintf(stderr,
                 "error: encoder failed to process text (%s, err=%d)\n",
                 label,
                 err);
    std::abort();
  }
}

template <class machine_type, class build_vocab_fn>
inline void append_emel_encoder_cases_with_text(std::vector<result> & results,
                                                const config & cfg,
                                                const char * short_name,
                                                const char * long_name,
                                                build_vocab_fn build_vocab,
                                                const bool preprocessed,
                                                const std::string_view short_text,
                                                const std::string_view long_text) {
  auto vocab = build_vocab();

  machine_type machine{};
  std::array<int32_t, k_token_capacity> tokens = {};
  int32_t token_count = 0;
  int32_t err = k_error_none;

  emel::text::encoders::event::encode short_request{
    .vocab = *vocab,
    .text = short_text,
    .preprocessed = preprocessed,
    .token_ids = std::span<int32_t>(tokens.data(), tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  emel::text::encoders::event::encode long_request{
    .vocab = *vocab,
    .text = long_text,
    .preprocessed = preprocessed,
    .token_ids = std::span<int32_t>(tokens.data(), tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  ensure_encodes(machine, short_request, short_name);
  ensure_encodes(machine, long_request, long_name);

  auto short_fn = [&]() { (void)run_encode(machine, short_request, token_count, err); };
  auto long_fn = [&]() { (void)run_encode(machine, long_request, token_count, err); };

  results.push_back(measure_case(short_name, cfg, short_fn));
  results.push_back(measure_case(long_name, cfg, long_fn));
}

template <class machine_type, class build_vocab_fn>
inline void append_emel_encoder_cases(std::vector<result> & results,
                                      const config & cfg,
                                      const char * short_name,
                                      const char * long_name,
                                      build_vocab_fn build_vocab,
                                      const bool preprocessed,
                                      const int short_repeats = 1,
                                      const int long_repeats = 64) {
  const std::string short_text = make_repeated_text(short_repeats);
  const std::string long_text = make_repeated_text(long_repeats);
  append_emel_encoder_cases_with_text<machine_type>(
    results,
    cfg,
    short_name,
    long_name,
    build_vocab,
    preprocessed,
    short_text,
    long_text);
}

}  // namespace emel::bench::encoder_bench
