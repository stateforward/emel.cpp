#include "parity_runner.hpp"

#include <climits>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/parser/gguf/actions.hpp"
#include "emel/parser/gguf/context.hpp"
#include "emel/parser/gguf/sm.hpp"
#include "emel/tokenizer/sm.hpp"

#include "llama.h"

namespace {

size_t estimate_token_capacity(std::string_view text) {
  const size_t base = text.size();
  size_t cap = base + 16;
  if (cap < 32) {
    cap = 32;
  }
  return cap;
}

bool load_emel_vocab(const std::string & model_path, emel::model::data & model,
                     emel::parser::gguf::context & gguf_ctx, int32_t & err_out) {
  err_out = EMEL_OK;
  emel::model::loader::event::load map_request{model};
  map_request.model_path = model_path;
  map_request.format_ctx = &gguf_ctx;
  map_request.vocab_only = true;

  if (!emel::parser::gguf::map_parser(map_request, &err_out)) {
    return false;
  }

  emel::parser::gguf::sm parser;
  emel::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  parse_request.model_path = model_path;
  parse_request.format_ctx = &gguf_ctx;
  parse_request.map_tensors = false;

  if (!parser.process_event(parse_request)) {
    err_out = parser.last_error();
    emel::parser::gguf::reset_context(gguf_ctx);
    return false;
  }

  emel::parser::gguf::reset_context(gguf_ctx);
  return true;
}

bool run_emel_tokenize(emel::tokenizer::sm & tokenizer,
                       const emel::model::data::vocab & vocab,
                       std::string_view text,
                       bool add_special,
                       bool parse_special,
                       std::vector<int32_t> & out_tokens,
                       int32_t & err_out) {
  const size_t capacity = estimate_token_capacity(text);
  if (capacity > static_cast<size_t>(INT32_MAX)) {
    err_out = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }

  out_tokens.assign(capacity, 0);
  int32_t token_count = 0;
  emel::tokenizer::event::tokenize tok_ev{};
  tok_ev.vocab = &vocab;
  tok_ev.text = text;
  tok_ev.add_special = add_special;
  tok_ev.parse_special = parse_special;
  tok_ev.token_ids_out = out_tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(capacity);
  tok_ev.token_count_out = &token_count;
  tok_ev.error_out = &err_out;

  if (!tokenizer.process_event(tok_ev)) {
    return false;
  }
  if (err_out != EMEL_OK) {
    return false;
  }
  if (token_count < 0 || token_count > static_cast<int32_t>(capacity)) {
    err_out = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }
  out_tokens.resize(static_cast<size_t>(token_count));
  return true;
}

bool run_llama_tokenize(const llama_vocab * vocab,
                        std::string_view text,
                        bool add_special,
                        bool parse_special,
                        std::vector<llama_token> & out_tokens) {
  if (text.size() > static_cast<size_t>(INT32_MAX)) {
    return false;
  }
  const int32_t text_len = static_cast<int32_t>(text.size());
  int32_t capacity = static_cast<int32_t>(estimate_token_capacity(text));
  out_tokens.assign(static_cast<size_t>(capacity), llama_token{});

  int32_t count = llama_tokenize(
      vocab,
      text.data(),
      text_len,
      out_tokens.data(),
      capacity,
      add_special,
      parse_special);
  if (count == INT32_MIN) {
    return false;
  }
  if (count < 0) {
    capacity = -count;
    if (capacity <= 0) {
      return false;
    }
    out_tokens.assign(static_cast<size_t>(capacity), llama_token{});
    count = llama_tokenize(
        vocab,
        text.data(),
        text_len,
        out_tokens.data(),
        capacity,
        add_special,
        parse_special);
  }
  if (count < 0) {
    return false;
  }
  out_tokens.resize(static_cast<size_t>(count));
  return true;
}

std::string escape_piece(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const unsigned char c : text) {
    if (c >= 0x20 && c <= 0x7e && c != '\\') {
      out.push_back(static_cast<char>(c));
      continue;
    }
    if (c == '\\') {
      out += "\\\\";
      continue;
    }
    char buf[5] = {};
    std::snprintf(buf, sizeof(buf), "\\x%02x", static_cast<unsigned int>(c));
    out += buf;
  }
  return out;
}

std::string emel_token_text(const emel::model::data::vocab & vocab, const int32_t token) {
  if (token < 0 || static_cast<uint32_t>(token) >= vocab.n_tokens) {
    return {};
  }
  const auto & entry = vocab.entries[static_cast<uint32_t>(token)];
  if (entry.text_length == 0) {
    return {};
  }
  return std::string(vocab.token_storage.data() + entry.text_offset,
                     entry.text_length);
}

std::string llama_token_text(const llama_vocab * vocab, const llama_token token) {
  if (vocab == nullptr) {
    return {};
  }
  const char * text = llama_vocab_get_text(vocab, token);
  if (text == nullptr) {
    return {};
  }
  return std::string(text);
}

template <typename T>
void dump_token_list(const char * label,
                     const std::vector<T> & tokens,
                     const emel::model::data::vocab & emel_vocab,
                     const llama_vocab * llama_vocab) {
  std::fprintf(stdout, "%s (%zu):", label, tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    const int32_t id = static_cast<int32_t>(tokens[i]);
    std::string piece;
    if (llama_vocab != nullptr) {
      piece = llama_token_text(llama_vocab, id);
    } else {
      piece = emel_token_text(emel_vocab, id);
    }
    const std::string escaped = escape_piece(piece);
    if (!escaped.empty()) {
      std::fprintf(stdout, "%s%d(\"%s\")", (i == 0 ? " " : ", "), id, escaped.c_str());
    } else {
      std::fprintf(stdout, "%s%d", (i == 0 ? " " : ", "), id);
    }
  }
  std::fprintf(stdout, "\n");
}

bool compare_tokens(const std::vector<int32_t> & emel_tokens,
                    const std::vector<llama_token> & llama_tokens) {
  if (emel_tokens.size() != llama_tokens.size()) {
    std::fprintf(stderr,
                 "token count mismatch: emel=%zu llama=%zu\n",
                 emel_tokens.size(),
                 llama_tokens.size());
    return false;
  }
  for (size_t i = 0; i < emel_tokens.size(); ++i) {
    if (emel_tokens[i] != static_cast<int32_t>(llama_tokens[i])) {
      std::fprintf(stderr,
                   "token mismatch at index %zu: emel=%d llama=%d\n",
                   i,
                   emel_tokens[i],
                   static_cast<int32_t>(llama_tokens[i]));
      return false;
    }
  }
  return true;
}

}  // namespace

namespace emel::paritychecker {

int run_parity(const parity_options & opts) {
  auto model = std::make_unique<emel::model::data>();
  emel::parser::gguf::context gguf_ctx{};
  int32_t err = EMEL_OK;
  if (!load_emel_vocab(opts.model_path, *model, gguf_ctx, err)) {
    std::fprintf(stderr, "emel parser failed: %d\n", err);
    return 1;
  }

  emel::tokenizer::sm tokenizer;
  int32_t bind_err = EMEL_OK;
  emel::tokenizer::event::bind bind_ev{};
  bind_ev.vocab = &model->vocab_data;
  bind_ev.error_out = &bind_err;
  if (!tokenizer.process_event(bind_ev) || bind_err != EMEL_OK) {
    std::fprintf(stderr, "emel tokenizer bind failed: %d\n", bind_err);
    return 1;
  }

  std::vector<int32_t> emel_tokens;
  if (!run_emel_tokenize(tokenizer,
                         model->vocab_data,
                         opts.text,
                         opts.add_special,
                         opts.parse_special,
                         emel_tokens,
                         err)) {
    std::fprintf(stderr, "emel tokenization failed: %d\n", err);
    return 1;
  }

  llama_backend_init();
  llama_model_params params = llama_model_default_params();
  params.vocab_only = true;
  struct llama_model * llama_model = llama_model_load_from_file(opts.model_path.c_str(), params);
  if (llama_model == nullptr) {
    std::fprintf(stderr, "llama model load failed\n");
    llama_backend_free();
    return 1;
  }
  const llama_vocab * llama_vocab = llama_model_get_vocab(llama_model);
  if (llama_vocab == nullptr) {
    std::fprintf(stderr, "llama vocab missing\n");
    llama_model_free(llama_model);
    llama_backend_free();
    return 1;
  }

  std::vector<llama_token> llama_tokens;
  if (!run_llama_tokenize(llama_vocab,
                          opts.text,
                          opts.add_special,
                          opts.parse_special,
                          llama_tokens)) {
    std::fprintf(stderr, "llama tokenization failed\n");
    llama_model_free(llama_model);
    llama_backend_free();
    return 1;
  }

  if (opts.dump_tokens) {
    dump_token_list("emel", emel_tokens, model->vocab_data, llama_vocab);
    dump_token_list("llama", llama_tokens, model->vocab_data, llama_vocab);
  }

  const bool matched = compare_tokens(emel_tokens, llama_tokens);
  if (matched) {
    std::fprintf(stdout, "parity ok (%zu tokens)\n", emel_tokens.size());
  }

  llama_model_free(llama_model);
  llama_backend_free();

  return matched ? 0 : 1;
}

}  // namespace emel::paritychecker
