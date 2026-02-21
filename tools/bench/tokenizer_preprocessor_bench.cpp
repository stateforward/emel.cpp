#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"
#include "emel/tokenizer/bpe/regex.hpp"
#include "emel/tokenizer/preprocessor/detail.hpp"
#include "emel/tokenizer/preprocessor/bpe/sm.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"
#include "emel/tokenizer/preprocessor/ugm/sm.hpp"

#include "unicode.h"

namespace {

using tokenizer_pre = emel::model::data::tokenizer_pre;
using emel::tokenizer::preprocessor::fragment;
using emel::tokenizer::preprocessor::fragment_kind;

constexpr size_t k_fragment_capacity = emel::tokenizer::preprocessor::k_max_fragments;

struct preprocessor_case {
  const char * name = nullptr;
  tokenizer_pre pre = tokenizer_pre::DEFAULT;
  std::string text;
};

struct ugm_case {
  const char * name = nullptr;
  std::string text;
  bool parse_special = false;
};

std::unique_ptr<emel::model::data::vocab> make_bpe_vocab(const tokenizer_pre pre) {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->n_tokens = 0;
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab->tokenizer_pre_id = pre;
  return vocab;
}

std::unique_ptr<emel::model::data::vocab> make_ugm_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->n_tokens = 2;
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::UGM;
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
  vocab->lstrip_flags[0] = 0x01;
  vocab->rstrip_flags[0] = 0x01;
  return vocab;
}

std::string make_long_text() {
  std::string out;
  out.reserve(2048);
  for (int i = 0; i < 32; ++i) {
    out += " hello";
    out += " world";
    out += " 123";
    out += "!";
  }
  return out;
}

std::string make_ugm_long_text() {
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

struct reference_fragments {
  std::string text;
  std::vector<std::string> storage;
  std::vector<fragment> fragments;
};

reference_fragments build_reference_fragments(const emel::model::data::vocab & vocab,
                                              const std::string & text) {
  emel::model::data::tokenizer_pre pre_id = emel::model::data::tokenizer_pre::UNKNOWN;
  std::vector<std::string> regex_exprs;
  emel::tokenizer::bpe::detail::assign_bpe_regex(pre_id, regex_exprs, vocab);

  const auto words = ::unicode_regex_split(text, regex_exprs);

  reference_fragments out;
  out.storage.reserve(words.size());
  out.fragments.reserve(words.size());

  for (const auto & word : words) {
    if (word.empty()) {
      continue;
    }
    out.storage.push_back(word);
    out.fragments.push_back(
      fragment{fragment_kind::raw_text, std::string_view(out.storage.back()), -1});
  }

  return out;
}

reference_fragments build_reference_ugm_fragments(const emel::model::data::vocab & vocab,
                                                  const std::string & text,
                                                  const bool parse_special) {
  reference_fragments out;
  out.text = text;

  emel::tokenizer::preprocessor::special_token_cache cache = {};
  if (!emel::tokenizer::preprocessor::detail::build_special_tokens(cache, vocab)) {
    std::fprintf(stderr, "error: ugm reference special token build failed\n");
    std::abort();
  }

  std::array<fragment, k_fragment_capacity> fragments = {};
  size_t count = 0;
  if (!emel::tokenizer::preprocessor::detail::partition_with_specials(
          out.text, cache, parse_special, fragments.data(), fragments.size(), &count)) {
    std::fprintf(stderr, "error: ugm reference partition failed\n");
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

void ensure_preprocessor_parity(const emel::model::data::vocab & vocab,
                                const std::string & text) {
  emel::tokenizer::preprocessor::bpe::sm machine{};
  std::array<fragment, k_fragment_capacity> fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  if (!collect_emel_fragments(machine, vocab, text, false, fragments, count, err) ||
      err != EMEL_OK) {
    std::fprintf(stderr, "error: preprocessor failed for parity check: %d\n", err);
    std::abort();
  }

  const reference_fragments reference = build_reference_fragments(vocab, text);
  if (count != reference.fragments.size()) {
    std::fprintf(stderr,
                 "error: preprocessor parity mismatch count %zu vs %zu\n",
                 count,
                 reference.fragments.size());
    std::abort();
  }

  for (size_t i = 0; i < count; ++i) {
    const auto & lhs = fragments[i];
    const auto & rhs = reference.fragments[i];
    if (lhs.kind != rhs.kind || lhs.text != rhs.text) {
      std::fprintf(stderr,
                   "error: preprocessor parity mismatch at %zu\n",
                   i);
      std::abort();
    }
  }
}

void ensure_preprocessor_ugm_parity(const emel::model::data::vocab & vocab,
                                    const std::string & text,
                                    const bool parse_special) {
  emel::tokenizer::preprocessor::ugm::sm machine{};
  std::array<fragment, k_fragment_capacity> fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  if (!collect_emel_fragments(machine, vocab, text, parse_special, fragments, count, err) ||
      err != EMEL_OK) {
    std::fprintf(stderr, "error: ugm preprocessor failed for parity check: %d\n", err);
    std::abort();
  }

  const reference_fragments reference =
      build_reference_ugm_fragments(vocab, text, parse_special);
  if (count != reference.fragments.size()) {
    std::fprintf(stderr,
                 "error: ugm preprocessor parity mismatch count %zu vs %zu\n",
                 count,
                 reference.fragments.size());
    std::abort();
  }

  for (size_t i = 0; i < count; ++i) {
    const auto & lhs = fragments[i];
    const auto & rhs = reference.fragments[i];
    if (lhs.kind != rhs.kind || lhs.text != rhs.text || lhs.token != rhs.token) {
      std::fprintf(stderr,
                   "error: ugm preprocessor parity mismatch at %zu\n",
                   i);
      std::abort();
    }
  }
}

}  // namespace

namespace emel::bench {

void append_emel_tokenizer_preprocessor_cases(std::vector<result> & results,
                                              const config & cfg) {
  const preprocessor_case cases[] = {
    {"tokenizer/preprocessor_bpe_short", tokenizer_pre::GPT2, "hello world"},
    {"tokenizer/preprocessor_bpe_long", tokenizer_pre::GPT2, make_long_text()},
  };

  for (const auto & entry : cases) {
    const auto vocab = make_bpe_vocab(entry.pre);
    ensure_preprocessor_parity(*vocab, entry.text);

    emel::tokenizer::preprocessor::bpe::sm machine{};
    std::array<fragment, k_fragment_capacity> fragments = {};
    size_t count = 0;
    int32_t err = EMEL_OK;

    auto fn = [&]() {
      (void)collect_emel_fragments(machine, *vocab, entry.text, false,
                                   fragments, count, err);
    };

    results.push_back(measure_case(entry.name, cfg, fn));
  }

  const ugm_case ugm_cases[] = {
    {"tokenizer/preprocessor_ugm_short", "hello A  BBB world", true},
    {"tokenizer/preprocessor_ugm_long", make_ugm_long_text(), true},
  };

  for (const auto & entry : ugm_cases) {
    const auto vocab = make_ugm_vocab();
    ensure_preprocessor_ugm_parity(*vocab, entry.text, entry.parse_special);

    emel::tokenizer::preprocessor::ugm::sm machine{};
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

void append_reference_tokenizer_preprocessor_cases(std::vector<result> & results,
                                                   const config & cfg) {
  const preprocessor_case cases[] = {
    {"tokenizer/preprocessor_bpe_short", tokenizer_pre::GPT2, "hello world"},
    {"tokenizer/preprocessor_bpe_long", tokenizer_pre::GPT2, make_long_text()},
  };

  for (const auto & entry : cases) {
    const auto vocab = make_bpe_vocab(entry.pre);

    auto fn = [&]() {
      const auto fragments = build_reference_fragments(*vocab, entry.text);
      static volatile size_t sink = 0;
      sink += fragments.fragments.size();
      for (const auto & frag : fragments.fragments) {
        sink += frag.text.size();
      }
    };

    results.push_back(measure_case(entry.name, cfg, fn));
  }

  const ugm_case ugm_cases[] = {
    {"tokenizer/preprocessor_ugm_short", "hello A  BBB world", true},
    {"tokenizer/preprocessor_ugm_long", make_ugm_long_text(), true},
  };

  for (const auto & entry : ugm_cases) {
    const auto vocab = make_ugm_vocab();

    auto fn = [&]() {
      const auto fragments =
          build_reference_ugm_fragments(*vocab, entry.text, entry.parse_special);
      static volatile size_t sink = 0;
      sink += fragments.fragments.size();
      for (const auto & frag : fragments.fragments) {
        sink += frag.text.size();
      }
    };

    results.push_back(measure_case(entry.name, cfg, fn));
  }
}

}  // namespace emel::bench
