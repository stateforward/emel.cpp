#include "bench_cases.hpp"
#include "tokenizer_preprocessor_bench_common.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "emel/tokenizer/bpe/regex.hpp"
#include "emel/tokenizer/preprocessor/bpe/sm.hpp"

#include "unicode.h"

namespace {

using tokenizer_pre = emel::model::data::tokenizer_pre;
using emel::bench::tokenizer_preprocessor::fragment;
using emel::bench::tokenizer_preprocessor::fragment_kind;
using emel::bench::tokenizer_preprocessor::k_fragment_capacity;
using emel::bench::tokenizer_preprocessor::reference_fragments;

struct preprocessor_case {
  const char * name = nullptr;
  tokenizer_pre pre = tokenizer_pre::DEFAULT;
  std::string text;
};

std::unique_ptr<emel::model::data::vocab> make_bpe_vocab(const tokenizer_pre pre) {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->n_tokens = 0;
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab->tokenizer_pre_id = pre;
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

void ensure_preprocessor_bpe_parity(const emel::model::data::vocab & vocab,
                                    const std::string & text) {
  emel::tokenizer::preprocessor::bpe::sm machine{};
  std::array<fragment, k_fragment_capacity> fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  if (!emel::bench::tokenizer_preprocessor::collect_emel_fragments(
          machine, vocab, text, false, fragments, count, err) ||
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

}  // namespace

namespace emel::bench {

void append_emel_tokenizer_preprocessor_bpe_cases(std::vector<result> & results,
                                                  const config & cfg) {
  const preprocessor_case cases[] = {
    {"tokenizer/preprocessor_bpe_short", tokenizer_pre::GPT2, "hello world"},
    {"tokenizer/preprocessor_bpe_long", tokenizer_pre::GPT2, make_long_text()},
  };

  for (const auto & entry : cases) {
    const auto vocab = make_bpe_vocab(entry.pre);
    ensure_preprocessor_bpe_parity(*vocab, entry.text);

    emel::tokenizer::preprocessor::bpe::sm machine{};
    std::array<fragment, k_fragment_capacity> fragments = {};
    size_t count = 0;
    int32_t err = EMEL_OK;

    auto fn = [&]() {
      (void)emel::bench::tokenizer_preprocessor::collect_emel_fragments(
          machine, *vocab, entry.text, false, fragments, count, err);
    };

    results.push_back(measure_case(entry.name, cfg, fn));
  }
}

void append_reference_tokenizer_preprocessor_bpe_cases(std::vector<result> & results,
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
}

}  // namespace emel::bench
