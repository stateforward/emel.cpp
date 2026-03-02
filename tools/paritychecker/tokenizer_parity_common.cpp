#include "tokenizer_parity.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <string_view>
#include <vector>

#include "emel/emel.h"
#include "emel/text/tokenizer/sm.hpp"

#include "llama-vocab.h"

namespace {

bool run_emel_tokenizer(
    const emel::model::data::vocab & vocab,
    const std::string_view text,
    const bool add_special,
    const bool parse_special,
    const size_t expected_count,
    const emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant,
    const emel::text::encoders::encoder_kind encoder_variant,
    std::vector<int32_t> & tokens_out,
    int32_t & err_out) {
  emel::text::tokenizer::sm machine{};

  int32_t bind_err = EMEL_OK;
  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.preprocessor_variant = preprocessor_variant;
  bind_ev.encoder_variant = encoder_variant;
  bind_ev.error_out = &bind_err;

  const bool bind_ok = machine.process_event(bind_ev);
  if (!bind_ok || bind_err != EMEL_OK) {
    std::fprintf(stderr,
                 "emel tokenizer bind failed: accepted=%s err=%d\n",
                 bind_ok ? "true" : "false",
                 bind_err);
    err_out = bind_err;
    return false;
  }

  const size_t capacity = std::max<size_t>(4096u, expected_count + 32u);
  std::vector<int32_t> token_buffer(capacity, 0);

  int32_t token_count = 0;
  int32_t tokenize_err = EMEL_OK;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = text;
  tok_ev.add_special = add_special;
  tok_ev.parse_special = parse_special;
  tok_ev.token_ids_out = token_buffer.data();
  tok_ev.token_capacity = static_cast<int32_t>(token_buffer.size());
  tok_ev.token_count_out = &token_count;
  tok_ev.error_out = &tokenize_err;

  const bool tokenize_ok = machine.process_event(tok_ev);
  if (!tokenize_ok || tokenize_err != EMEL_OK) {
    std::fprintf(stderr,
                 "emel tokenizer tokenize failed: accepted=%s err=%d\n",
                 tokenize_ok ? "true" : "false",
                 tokenize_err);
    err_out = tokenize_err;
    return false;
  }

  if (token_count < 0 || static_cast<size_t>(token_count) > token_buffer.size()) {
    std::fprintf(stderr,
                 "emel tokenizer returned invalid token count: %d (capacity=%zu)\n",
                 token_count,
                 token_buffer.size());
    err_out = EMEL_ERR_INTERNAL;
    return false;
  }

  tokens_out.assign(token_buffer.begin(), token_buffer.begin() + token_count);
  err_out = EMEL_OK;
  return true;
}

void dump_tokens(const char * label, const std::vector<int32_t> & tokens) {
  std::fprintf(stdout, "%s[%zu]:", label, tokens.size());
  for (const int32_t token : tokens) {
    std::fprintf(stdout, " %d", token);
  }
  std::fprintf(stdout, "\n");
}

void dump_llama_tokens(const char * label, const std::vector<llama_token> & tokens) {
  std::fprintf(stdout, "%s[%zu]:", label, tokens.size());
  for (const llama_token token : tokens) {
    std::fprintf(stdout, " %d", static_cast<int32_t>(token));
  }
  std::fprintf(stdout, "\n");
}

bool compare_token_streams(const std::vector<int32_t> & emel_tokens,
                           const std::vector<llama_token> & llama_tokens,
                           const bool dump) {
  const size_t shared = std::min(emel_tokens.size(), llama_tokens.size());
  for (size_t i = 0; i < shared; ++i) {
    if (emel_tokens[i] == llama_tokens[i]) {
      continue;
    }
    std::fprintf(stderr,
                 "token mismatch at index %zu: emel=%d llama=%d\n",
                 i,
                 emel_tokens[i],
                 static_cast<int32_t>(llama_tokens[i]));
    if (dump) {
      dump_tokens("emel", emel_tokens);
      dump_llama_tokens("llama", llama_tokens);
    }
    return false;
  }

  if (emel_tokens.size() != llama_tokens.size()) {
    std::fprintf(stderr,
                 "token count mismatch: emel=%zu llama=%zu\n",
                 emel_tokens.size(),
                 llama_tokens.size());
    if (dump) {
      dump_tokens("emel", emel_tokens);
      dump_llama_tokens("llama", llama_tokens);
    }
    return false;
  }

  return true;
}

}  // namespace

namespace emel::paritychecker {

int run_tokenizer_variant_parity(
    const parity_options & opts,
    const llama_vocab & llama_vocab_ref,
    const emel::model::data::vocab & emel_vocab,
    const emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant,
    const emel::text::encoders::encoder_kind encoder_variant,
    const char * variant_name) {
  std::vector<llama_token> llama_tokens;
  try {
    llama_tokens = llama_vocab_ref.tokenize(opts.text, opts.add_special, opts.parse_special);
  } catch (const std::exception & ex) {
    std::fprintf(stderr, "llama tokenize threw exception: %s\n", ex.what());
    return 1;
  }

  std::vector<int32_t> emel_tokens;
  int32_t emel_err = EMEL_OK;
  if (!run_emel_tokenizer(emel_vocab,
                          opts.text,
                          opts.add_special,
                          opts.parse_special,
                          llama_tokens.size(),
                          preprocessor_variant,
                          encoder_variant,
                          emel_tokens,
                          emel_err)) {
    std::fprintf(stderr, "emel tokenize failed with err=%d\n", emel_err);
    return 1;
  }

  if (!compare_token_streams(emel_tokens, llama_tokens, opts.dump)) {
    std::fprintf(stderr,
                 "%s tokenizer parity mismatch: model=%s tokenizer=%s pre=%s\n",
                 variant_name,
                 opts.model_path.c_str(),
                 emel_vocab.tokenizer_model_name.data(),
                 emel_vocab.tokenizer_pre_name.data());
    return 1;
  }

  std::fprintf(stdout,
               "%s tokenizer parity ok (%zu tokens, tokenizer=%s pre=%s)\n",
               variant_name,
               llama_tokens.size(),
               emel_vocab.tokenizer_model_name.data(),
               emel_vocab.tokenizer_pre_name.data());
  return 0;
}

}  // namespace emel::paritychecker
