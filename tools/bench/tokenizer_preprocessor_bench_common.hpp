#pragma once

#include "bench_common.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/tokenizer/preprocessor/detail.hpp"
#include "emel/tokenizer/preprocessor/events.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::bench::tokenizer_preprocessor {

using emel::tokenizer::preprocessor::fragment;
using emel::tokenizer::preprocessor::fragment_kind;

constexpr size_t k_fragment_capacity = emel::tokenizer::preprocessor::k_max_fragments;

struct special_case {
  const char * name = nullptr;
  std::string text;
  bool parse_special = false;
};

struct reference_fragments {
  std::string text;
  std::vector<std::string> storage;
  std::vector<fragment> fragments;
};

inline std::unique_ptr<emel::model::data::vocab> make_simple_vocab(
    const emel::model::data::tokenizer_model model_id,
    const bool set_strip_flags) {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->n_tokens = 2;
  vocab->tokenizer_model_id = model_id;
  vocab->entries[0].text_offset = 0;
  vocab->entries[0].text_length = 1;
  vocab->entries[0].type = 4;
  vocab->entries[1].text_offset = 2;
  vocab->entries[1].text_length = 3;
  vocab->entries[1].type = 3;
  vocab->token_storage[0] = 'A';
  vocab->token_storage[2] = 'B';
  vocab->token_storage[3] = 'B';
  vocab->token_storage[4] = 'B';
  if (set_strip_flags) {
    vocab->lstrip_flags[0] = 0x01;
    vocab->rstrip_flags[0] = 0x01;
  }
  return vocab;
}

inline std::string make_specials_long_text() {
  std::string out;
  out.reserve(2048);
  for (int i = 0; i < 32; ++i) {
    out += " hello ";
    out += "A";
    out += " world ";
    out += "BBB";
    out += "!";
  }
  return out;
}

inline std::array<special_case, 2> make_special_cases(const char * short_name,
                                                      const char * long_name) {
  return {{{short_name, "hello A  BBB world", true},
           {long_name, make_specials_long_text(), true}}};
}

inline reference_fragments build_reference_special_fragments(
    const emel::model::data::vocab & vocab,
    const std::string & text,
    const bool parse_special,
    const char * label) {
  reference_fragments out;
  out.text = text;

  emel::tokenizer::preprocessor::special_token_cache cache = {};
  if (!emel::tokenizer::preprocessor::detail::build_special_tokens(cache, vocab)) {
    std::fprintf(stderr, "error: %s reference special token build failed\n", label);
    std::abort();
  }

  std::array<fragment, k_fragment_capacity> fragments = {};
  size_t count = 0;
  if (!emel::tokenizer::preprocessor::detail::partition_with_specials(
          out.text, cache, parse_special, fragments.data(), fragments.size(), &count)) {
    std::fprintf(stderr, "error: %s reference partition failed\n", label);
    std::abort();
  }

  out.fragments.assign(fragments.begin(),
                       fragments.begin() + static_cast<std::ptrdiff_t>(count));
  return out;
}

template <class machine_type>
bool collect_emel_fragments(machine_type & machine,
                            const emel::model::data::vocab & vocab,
                            const std::string & text,
                            const bool parse_special,
                            std::array<fragment, k_fragment_capacity> & fragments,
                            size_t & count,
                            int32_t & err) {
  count = 0;
  err = EMEL_OK;
  emel::tokenizer::preprocessor::event::preprocess request = {};
  request.vocab = &vocab;
  request.text = text;
  request.parse_special = parse_special;
  request.fragments_out = fragments.data();
  request.fragment_capacity = fragments.size();
  request.fragment_count_out = &count;
  request.error_out = &err;
  return machine.process_event(request);
}

template <class machine_type>
void ensure_special_preprocessor_parity(const char * label,
                                        const emel::model::data::vocab & vocab,
                                        const std::string & text,
                                        const bool parse_special) {
  machine_type machine{};
  std::array<fragment, k_fragment_capacity> fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  if (!collect_emel_fragments(machine, vocab, text, parse_special, fragments, count, err) ||
      err != EMEL_OK) {
    std::fprintf(stderr, "error: %s preprocessor failed for parity check: %d\n",
                 label,
                 err);
    std::abort();
  }

  const reference_fragments reference =
      build_reference_special_fragments(vocab, text, parse_special, label);
  if (count != reference.fragments.size()) {
    std::fprintf(stderr,
                 "error: %s preprocessor parity mismatch count %zu vs %zu\n",
                 label,
                 count,
                 reference.fragments.size());
    std::abort();
  }

  for (size_t i = 0; i < count; ++i) {
    const auto & lhs = fragments[i];
    const auto & rhs = reference.fragments[i];
    if (lhs.kind != rhs.kind || lhs.text != rhs.text || lhs.token != rhs.token) {
      std::fprintf(stderr,
                   "error: %s preprocessor parity mismatch at %zu\n",
                   label,
                   i);
      std::abort();
    }
  }
}

template <class machine_type>
void append_emel_special_preprocessor_cases(std::vector<result> & results,
                                            const config & cfg,
                                            const char * short_name,
                                            const char * long_name,
                                            const emel::model::data::tokenizer_model model_id,
                                            const bool set_strip_flags,
                                            const char * label) {
  const auto cases = make_special_cases(short_name, long_name);

  for (const auto & entry : cases) {
    const auto vocab = make_simple_vocab(model_id, set_strip_flags);
    ensure_special_preprocessor_parity<machine_type>(label, *vocab, entry.text,
                                                     entry.parse_special);

    machine_type machine{};
    std::array<fragment, k_fragment_capacity> fragments = {};
    size_t count = 0;
    int32_t err = EMEL_OK;

    auto fn = [&]() {
      (void)collect_emel_fragments(machine, *vocab, entry.text, entry.parse_special,
                                   fragments, count, err);
    };

    results.push_back(measure_case(entry.name, cfg, fn));
  }
}

inline void append_reference_special_preprocessor_cases(
    std::vector<result> & results,
    const config & cfg,
    const char * short_name,
    const char * long_name,
    const emel::model::data::tokenizer_model model_id,
    const bool set_strip_flags,
    const char * label) {
  const auto cases = make_special_cases(short_name, long_name);

  for (const auto & entry : cases) {
    const auto vocab = make_simple_vocab(model_id, set_strip_flags);

    auto fn = [&]() {
      const auto fragments =
          build_reference_special_fragments(*vocab, entry.text, entry.parse_special, label);
      static volatile size_t sink = 0;
      sink += fragments.fragments.size();
      for (const auto & frag : fragments.fragments) {
        sink += frag.text.size();
      }
    };

    results.push_back(measure_case(entry.name, cfg, fn));
  }
}

}  // namespace emel::bench::tokenizer_preprocessor
