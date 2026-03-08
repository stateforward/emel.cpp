#include "parity_runner.hpp"
#include "tokenizer_parity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/text/jinja/formatter/sm.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/sm.hpp"

#include "ggml-cpu.h"
#include "ggml.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

namespace {

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_internal = 3;
constexpr const char * k_generation_fixture_name = "Llama-68M-Chat-v1-Q2_K.gguf";

bool file_exists(const std::string & path) {
  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

std::string_view path_basename(const std::string & path) {
  const size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return std::string_view(path.data() + pos + 1u, path.size() - pos - 1u);
}

bool is_expected_generation_fixture(const std::string & model_path) {
  return path_basename(model_path) == k_generation_fixture_name;
}

struct parser_done_capture {
  bool called = false;
  const emel::gbnf::grammar * grammar = nullptr;
};

struct parser_error_capture {
  bool called = false;
  const emel::gbnf::grammar * grammar = nullptr;
  int32_t err = 0;
};

bool on_gbnf_done(void * owner, const emel::gbnf::rule_parser::events::parsing_done & ev) {
  auto * capture = static_cast<parser_done_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  return true;
}

bool on_gbnf_error(void * owner, const emel::gbnf::rule_parser::events::parsing_error & ev) {
  auto * capture = static_cast<parser_error_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  capture->err = ev.err;
  return true;
}

bool run_emel_gbnf_parse(std::string_view grammar_text,
                         emel::gbnf::grammar & grammar_out,
                         int32_t & err_out) {
  parser_done_capture done_capture{};
  parser_error_capture error_capture{};

  emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_done &)> done_cb{
      &done_capture,
      on_gbnf_done};
  emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_error &)> error_cb{
      &error_capture,
      on_gbnf_error};

  emel::gbnf::rule_parser::event::parse parse_ev{
      .grammar_text = grammar_text,
      .grammar_out = &grammar_out,
      .dispatch_done = done_cb,
      .dispatch_error = error_cb,
  };

  emel::gbnf::rule_parser::sm parser{};
  const bool accepted = parser.process_event(parse_ev);
  if (accepted && done_capture.called && !error_capture.called) {
    err_out = k_error_ok;
    return true;
  }
  if (error_capture.called) {
    err_out = error_capture.err;
    return false;
  }
  err_out = k_error_internal;
  return false;
}

bool run_llama_gbnf_parse(std::string_view grammar_text, llama_grammar_rules & rules_out) {
  std::string grammar(grammar_text);
  llama_grammar_parser parser{nullptr};
  if (!parser.parse(grammar.c_str())) {
    return false;
  }
  rules_out = std::move(parser.rules);
  return true;
}

struct jinja_parse_capture {
  bool done_called = false;
  bool error_called = false;
};

struct jinja_render_capture {
  bool done_called = false;
  bool error_called = false;
};

bool on_jinja_parse_done(void * owner,
                         const emel::text::jinja::events::parsing_done &) {
  auto * capture = static_cast<jinja_parse_capture *>(owner);
  capture->done_called = true;
  return true;
}

bool on_jinja_parse_error(void * owner,
                          const emel::text::jinja::events::parsing_error &) {
  auto * capture = static_cast<jinja_parse_capture *>(owner);
  capture->error_called = true;
  return true;
}

bool on_jinja_render_done(void * owner,
                          const emel::text::jinja::events::rendering_done &) {
  auto * capture = static_cast<jinja_render_capture *>(owner);
  capture->done_called = true;
  return true;
}

bool on_jinja_render_error(void * owner,
                           const emel::text::jinja::events::rendering_error &) {
  auto * capture = static_cast<jinja_render_capture *>(owner);
  capture->error_called = true;
  return true;
}

bool run_emel_jinja_parse(std::string_view template_text,
                          emel::text::jinja::program & program_out,
                          int32_t & parse_err_out,
                          size_t & parse_error_pos_out) {
  jinja_parse_capture capture{};
  emel::text::jinja::parser::action::context parse_ctx{};
  emel::text::jinja::parser::sm parser{parse_ctx};
  const emel::text::jinja::event::parse::done_callback done_cb{
      &capture,
      on_jinja_parse_done};
  const emel::text::jinja::event::parse::error_callback error_cb{
      &capture,
      on_jinja_parse_error};
  const emel::text::jinja::event::parse parse_ev{
      template_text,
      program_out,
      done_cb,
      error_cb,
      parse_err_out,
      parse_error_pos_out,
  };
  const bool accepted = parser.process_event(parse_ev);
  return accepted && capture.done_called && !capture.error_called &&
         parse_err_out == static_cast<int32_t>(emel::text::jinja::parser::error::none);
}

bool run_reference_jinja_parse(std::string_view template_text,
                               ::jinja::program & program_out) {
  try {
    ::jinja::lexer lex;
    ::jinja::lexer_result lex_res = lex.tokenize(std::string(template_text));
    program_out = ::jinja::parse_from_tokens(lex_res);
    return true;
  } catch (...) {
    return false;
  }
}

bool run_emel_jinja_render(const emel::text::jinja::program & program,
                           std::string_view template_text,
                           std::string & rendered_out) {
  jinja_render_capture capture{};
  emel::text::jinja::formatter::action::context formatter_ctx{};
  emel::text::jinja::formatter::sm formatter{formatter_ctx};
  std::vector<char> output_buffer(
      std::max<size_t>(1, static_cast<size_t>(template_text.size() + 1)));
  size_t output_len = 0;
  int32_t render_err = static_cast<int32_t>(emel::text::jinja::formatter::error::none);
  size_t render_error_pos = 0;
  const emel::text::jinja::event::render::done_callback done_cb{
      &capture,
      on_jinja_render_done};
  const emel::text::jinja::event::render::error_callback error_cb{
      &capture,
      on_jinja_render_error};
  const emel::text::jinja::event::render render_ev{
      program,
      template_text,
      output_buffer[0],
      output_buffer.size(),
      done_cb,
      error_cb,
      nullptr,
      &output_len,
      nullptr,
      &render_err,
      &render_error_pos,
  };

  const bool accepted = formatter.process_event(render_ev);
  if (!accepted || capture.error_called ||
      render_err != static_cast<int32_t>(emel::text::jinja::formatter::error::none)) {
    return false;
  }
  rendered_out.assign(output_buffer.data(), output_len);
  return true;
}

bool run_reference_jinja_render(const ::jinja::program & program,
                                std::string & rendered_out) {
  try {
    ::jinja::context ctx;
    ctx.set_val("name", ::jinja::mk_val<::jinja::value_string>("world"));
    ctx.set_val("cond", ::jinja::mk_val<::jinja::value_bool>(true));
    auto items = ::jinja::mk_val<::jinja::value_array>();
    items->push_back(::jinja::mk_val<::jinja::value_int>(1));
    items->push_back(::jinja::mk_val<::jinja::value_int>(2));
    items->push_back(::jinja::mk_val<::jinja::value_int>(3));
    ctx.set_val("items", items);
    ::jinja::runtime runtime{ctx};
    auto result = runtime.execute(program);
    auto parts = ::jinja::runtime::gather_string_parts(result);
    rendered_out = ::jinja::render_string_parts(parts);
    return true;
  } catch (...) {
    return false;
  }
}

std::string_view strip_trailing_newline(const std::string & value) {
  size_t len = value.size();
  if (len > 0 && value[len - 1] == '\n') {
    --len;
  }
  return std::string_view(value.data(), len);
}

bool compare_grammars(const emel::gbnf::grammar & emel_grammar,
                      const llama_grammar_rules & llama_rules) {
  if (emel_grammar.rule_count != llama_rules.size()) {
    std::fprintf(stderr,
                 "rule count mismatch: emel=%u llama=%zu\n",
                 emel_grammar.rule_count,
                 llama_rules.size());
    return false;
  }

  for (uint32_t rule_id = 0; rule_id < emel_grammar.rule_count; ++rule_id) {
    const emel::gbnf::rule_view emel_rule = emel_grammar.rule(rule_id);
    const llama_grammar_rule & llama_rule = llama_rules[rule_id];
    const uint32_t llama_len = static_cast<uint32_t>(llama_rule.size());
    if (emel_rule.length != llama_len) {
      std::fprintf(stderr,
                   "rule length mismatch at rule %u: emel=%u llama=%u\n",
                   rule_id,
                   emel_rule.length,
                   llama_len);
      return false;
    }
    for (uint32_t i = 0; i < emel_rule.length; ++i) {
      const emel::gbnf::element & emel_elem = emel_rule.elements[i];
      const llama_grammar_element & llama_elem = llama_rule[i];
      const uint32_t emel_type = static_cast<uint32_t>(emel_elem.type);
      const uint32_t llama_type = static_cast<uint32_t>(llama_elem.type);
      if (emel_type != llama_type || emel_elem.value != llama_elem.value) {
        std::fprintf(stderr,
                     "element mismatch at rule %u index %u: "
                     "emel(type=%u,value=%u) llama(type=%u,value=%u)\n",
                     rule_id,
                     i,
                     emel_type,
                     emel_elem.value,
                     llama_type,
                     llama_elem.value);
        return false;
      }
    }
  }
  return true;
}

void dump_emel_grammar(const emel::gbnf::grammar & grammar) {
  std::fprintf(stdout,
               "emel grammar: rules=%u elements=%u\n",
               grammar.rule_count,
               grammar.element_count);
  for (uint32_t rule_id = 0; rule_id < grammar.rule_count; ++rule_id) {
    const emel::gbnf::rule_view rule = grammar.rule(rule_id);
    std::fprintf(stdout, "  rule[%u] len=%u:", rule_id, rule.length);
    for (uint32_t i = 0; i < rule.length; ++i) {
      const emel::gbnf::element & elem = rule.elements[i];
      std::fprintf(stdout,
                   " (%u,%u)",
                   static_cast<uint32_t>(elem.type),
                   elem.value);
    }
    std::fprintf(stdout, "\n");
  }
}

void dump_llama_grammar(const llama_grammar_rules & rules) {
  std::fprintf(stdout, "llama grammar: rules=%zu\n", rules.size());
  for (size_t rule_id = 0; rule_id < rules.size(); ++rule_id) {
    const llama_grammar_rule & rule = rules[rule_id];
    std::fprintf(stdout, "  rule[%zu] len=%zu:", rule_id, rule.size());
    for (const auto & elem : rule) {
      std::fprintf(stdout,
                   " (%u,%u)",
                   static_cast<unsigned int>(elem.type),
                   static_cast<unsigned int>(elem.value));
    }
    std::fprintf(stdout, "\n");
  }
}

struct llama_backend_guard {
  llama_backend_guard() {
    llama_backend_init();
  }

  ~llama_backend_guard() {
    llama_backend_free();
  }
};

template <size_t k_array_size>
void copy_name(std::array<char, k_array_size> & dst, const std::string & value) {
  static_assert(k_array_size > 0, "copy_name requires non-empty destination");
  dst.fill('\0');
  const size_t copy_len = std::min(value.size(), k_array_size - 1);
  if (copy_len > 0) {
    std::memcpy(dst.data(), value.data(), copy_len);
  }
}

template <size_t k_array_size>
void set_token_flag(std::array<uint8_t, k_array_size> & flags, const uint32_t token_id) {
  const uint32_t byte_index = token_id >> 3u;
  if (byte_index >= k_array_size) {
    return;
  }
  const uint8_t bit = static_cast<uint8_t>(1u << (token_id & 7u));
  flags[byte_index] = static_cast<uint8_t>(flags[byte_index] | bit);
}

bool attr_has(const llama_token_attr attr, const llama_token_attr flag) {
  const uint32_t attr_bits = static_cast<uint32_t>(attr);
  const uint32_t flag_bits = static_cast<uint32_t>(flag);
  return (attr_bits & flag_bits) != 0u;
}

int32_t token_type_from_attr(const llama_token_attr attr) {
  if (attr_has(attr, LLAMA_TOKEN_ATTR_UNKNOWN)) {
    return static_cast<int32_t>(LLAMA_TOKEN_TYPE_UNKNOWN);
  }
  if (attr_has(attr, LLAMA_TOKEN_ATTR_CONTROL)) {
    return static_cast<int32_t>(LLAMA_TOKEN_TYPE_CONTROL);
  }
  if (attr_has(attr, LLAMA_TOKEN_ATTR_USER_DEFINED)) {
    return static_cast<int32_t>(LLAMA_TOKEN_TYPE_USER_DEFINED);
  }
  if (attr_has(attr, LLAMA_TOKEN_ATTR_UNUSED)) {
    return static_cast<int32_t>(LLAMA_TOKEN_TYPE_UNUSED);
  }
  if (attr_has(attr, LLAMA_TOKEN_ATTR_BYTE)) {
    return static_cast<int32_t>(LLAMA_TOKEN_TYPE_BYTE);
  }
  if (attr_has(attr, LLAMA_TOKEN_ATTR_NORMAL)) {
    return static_cast<int32_t>(LLAMA_TOKEN_TYPE_NORMAL);
  }
  return static_cast<int32_t>(LLAMA_TOKEN_TYPE_UNDEFINED);
}

emel::model::data::tokenizer_model to_emel_tokenizer_model(
    const enum llama_vocab_type type) {
  using tokenizer_model = emel::model::data::tokenizer_model;

  switch (type) {
    case LLAMA_VOCAB_TYPE_NONE:
      return tokenizer_model::NONE;
    case LLAMA_VOCAB_TYPE_SPM:
      return tokenizer_model::SPM;
    case LLAMA_VOCAB_TYPE_BPE:
      return tokenizer_model::BPE;
    case LLAMA_VOCAB_TYPE_WPM:
      return tokenizer_model::WPM;
    case LLAMA_VOCAB_TYPE_UGM:
      return tokenizer_model::UGM;
    case LLAMA_VOCAB_TYPE_RWKV:
      return tokenizer_model::RWKV;
    case LLAMA_VOCAB_TYPE_PLAMO2:
      return tokenizer_model::PLAMO2;
    default:
      return tokenizer_model::UNKNOWN;
  }
}

emel::model::data::tokenizer_pre to_emel_tokenizer_pre(
    const llama_vocab_pre_type type) {
  using tokenizer_pre = emel::model::data::tokenizer_pre;

  switch (type) {
    case LLAMA_VOCAB_PRE_TYPE_DEFAULT:
      return tokenizer_pre::DEFAULT;
    case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
      return tokenizer_pre::LLAMA3;
    case LLAMA_VOCAB_PRE_TYPE_JAIS2:
      return tokenizer_pre::JAIS2;
    case LLAMA_VOCAB_PRE_TYPE_DBRX:
      return tokenizer_pre::DBRX;
    case LLAMA_VOCAB_PRE_TYPE_SMAUG:
      return tokenizer_pre::SMAUG;
    case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM:
      return tokenizer_pre::DEEPSEEK_LLM;
    case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER:
      return tokenizer_pre::DEEPSEEK_CODER;
    case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM:
      return tokenizer_pre::DEEPSEEK3_LLM;
    case LLAMA_VOCAB_PRE_TYPE_YOUTU:
      return tokenizer_pre::YOUTU;
    case LLAMA_VOCAB_PRE_TYPE_FALCON:
      return tokenizer_pre::FALCON;
    case LLAMA_VOCAB_PRE_TYPE_MPT:
      return tokenizer_pre::MPT;
    case LLAMA_VOCAB_PRE_TYPE_STARCODER:
      return tokenizer_pre::STARCODER;
    case LLAMA_VOCAB_PRE_TYPE_GPT2:
      return tokenizer_pre::GPT2;
    case LLAMA_VOCAB_PRE_TYPE_JAIS:
      return tokenizer_pre::JAIS;
    case LLAMA_VOCAB_PRE_TYPE_REFACT:
      return tokenizer_pre::REFACT;
    case LLAMA_VOCAB_PRE_TYPE_COMMAND_R:
      return tokenizer_pre::COMMAND_R;
    case LLAMA_VOCAB_PRE_TYPE_QWEN2:
      return tokenizer_pre::QWEN2;
    case LLAMA_VOCAB_PRE_TYPE_QWEN35:
      return tokenizer_pre::QWEN35;
    case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
      return tokenizer_pre::STABLELM2;
    case LLAMA_VOCAB_PRE_TYPE_OLMO:
      return tokenizer_pre::OLMO;
    case LLAMA_VOCAB_PRE_TYPE_PORO:
      return tokenizer_pre::PORO;
    case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
      return tokenizer_pre::CHATGLM4;
    case LLAMA_VOCAB_PRE_TYPE_VIKING:
      return tokenizer_pre::VIKING;
    case LLAMA_VOCAB_PRE_TYPE_TEKKEN:
      return tokenizer_pre::TEKKEN;
    case LLAMA_VOCAB_PRE_TYPE_SMOLLM:
      return tokenizer_pre::SMOLLM;
    case LLAMA_VOCAB_PRE_TYPE_CODESHELL:
      return tokenizer_pre::CODESHELL;
    case LLAMA_VOCAB_PRE_TYPE_BLOOM:
      return tokenizer_pre::BLOOM;
    case LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH:
      return tokenizer_pre::GPT3_FINNISH;
    case LLAMA_VOCAB_PRE_TYPE_EXAONE:
      return tokenizer_pre::EXAONE;
    case LLAMA_VOCAB_PRE_TYPE_EXAONE_MOE:
      return tokenizer_pre::EXAONE_MOE;
    case LLAMA_VOCAB_PRE_TYPE_CHAMELEON:
      return tokenizer_pre::CHAMELEON;
    case LLAMA_VOCAB_PRE_TYPE_MINERVA:
      return tokenizer_pre::MINERVA;
    case LLAMA_VOCAB_PRE_TYPE_GPT4O:
      return tokenizer_pre::GPT4O;
    case LLAMA_VOCAB_PRE_TYPE_TINY_AYA:
      return tokenizer_pre::TINY_AYA;
    case LLAMA_VOCAB_PRE_TYPE_SUPERBPE:
      return tokenizer_pre::SUPERBPE;
    case LLAMA_VOCAB_PRE_TYPE_TRILLION:
      return tokenizer_pre::TRILLION;
    case LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING:
      return tokenizer_pre::GRANITE_DOCLING;
    case LLAMA_VOCAB_PRE_TYPE_BAILINGMOE:
      return tokenizer_pre::BAILINGMOE;
    case LLAMA_VOCAB_PRE_TYPE_SEED_CODER:
      return tokenizer_pre::SEED_CODER;
    case LLAMA_VOCAB_PRE_TYPE_HUNYUAN:
      return tokenizer_pre::HUNYUAN;
    case LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE:
      return tokenizer_pre::HUNYUAN_DENSE;
    case LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM:
      return tokenizer_pre::JOYAI_LLM;
    case LLAMA_VOCAB_PRE_TYPE_KIMI_K2:
      return tokenizer_pre::KIMI_K2;
    case LLAMA_VOCAB_PRE_TYPE_GROK_2:
      return tokenizer_pre::GROK_2;
    case LLAMA_VOCAB_PRE_TYPE_AFMOE:
      return tokenizer_pre::AFMOE;
    case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:
      return tokenizer_pre::MINIMAX_M2;
    case LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN:
      return tokenizer_pre::SOLAR_OPEN;
    case LLAMA_VOCAB_PRE_TYPE_CHATGLM3:
    case LLAMA_VOCAB_PRE_TYPE_LLAMA4:
    case LLAMA_VOCAB_PRE_TYPE_PIXTRAL:
      return tokenizer_pre::UNKNOWN;
    default:
      return tokenizer_pre::UNKNOWN;
  }
}

bool load_emel_vocab_from_llama(const llama_vocab & src, emel::model::data::vocab & dst) {
  dst = {};
  dst.tokenizer_model_id = to_emel_tokenizer_model(src.get_type());
  dst.tokenizer_pre_id = to_emel_tokenizer_pre(src.get_pre_type());
  copy_name(dst.tokenizer_model_name, src.get_tokenizer_model());
  copy_name(dst.tokenizer_pre_name, src.get_tokenizer_pre());

  const uint32_t token_count = src.n_tokens();
  if (token_count > emel::model::data::k_max_vocab_tokens) {
    std::fprintf(stderr,
                 "vocab token count exceeds emel capacity: %u > %d\n",
                 token_count,
                 emel::model::data::k_max_vocab_tokens);
    return false;
  }
  dst.n_tokens = token_count;
  dst.n_token_types = src.n_token_types();

  uint32_t token_bytes_used = 0;
  for (uint32_t token_id = 0; token_id < token_count; ++token_id) {
    const llama_token llama_id = static_cast<llama_token>(token_id);
    const auto & token = src.get_token_data(llama_id);
    const uint32_t token_len = static_cast<uint32_t>(token.text.size());
    if (token_bytes_used + token_len > emel::model::data::k_max_vocab_bytes) {
      std::fprintf(stderr,
                   "token storage exceeds emel capacity at token %u (%u + %u > %d)\n",
                   token_id,
                   token_bytes_used,
                   token_len,
                   emel::model::data::k_max_vocab_bytes);
      return false;
    }

    if (token_len > 0) {
      std::memcpy(dst.token_storage.data() + token_bytes_used,
                  token.text.data(),
                  token_len);
    }

    emel::model::data::vocab_entry & entry = dst.entries[token_id];
    entry.text_offset = token_bytes_used;
    entry.text_length = token_len;
    entry.score = token.score;
    entry.type = token_type_from_attr(token.attr);
    token_bytes_used += token_len;

    if (attr_has(token.attr, LLAMA_TOKEN_ATTR_LSTRIP)) {
      set_token_flag(dst.lstrip_flags, token_id);
    }
    if (attr_has(token.attr, LLAMA_TOKEN_ATTR_RSTRIP)) {
      set_token_flag(dst.rstrip_flags, token_id);
    }
  }
  dst.token_bytes_used = token_bytes_used;

  const std::vector<std::string> merges = src.get_bpe_merges();
  if (merges.size() > emel::model::data::k_max_merges) {
    std::fprintf(stderr,
                 "merge count exceeds emel capacity: %zu > %d\n",
                 merges.size(),
                 emel::model::data::k_max_merges);
    return false;
  }

  uint32_t merge_bytes_used = 0;
  for (size_t i = 0; i < merges.size(); ++i) {
    const std::string & merge = merges[i];
    const uint32_t merge_len = static_cast<uint32_t>(merge.size());
    if (merge_bytes_used + merge_len > emel::model::data::k_max_merge_bytes) {
      std::fprintf(stderr,
                   "merge storage exceeds emel capacity at merge %zu (%u + %u > %d)\n",
                   i,
                   merge_bytes_used,
                   merge_len,
                   emel::model::data::k_max_merge_bytes);
      return false;
    }
    if (merge_len > 0) {
      std::memcpy(dst.merge_storage.data() + merge_bytes_used,
                  merge.data(),
                  merge_len);
    }
    dst.merge_offsets[i] = merge_bytes_used;
    dst.merge_lengths[i] = merge_len;
    merge_bytes_used += merge_len;
  }
  dst.n_merges = static_cast<uint32_t>(merges.size());
  dst.merge_bytes_used = merge_bytes_used;

  const std::vector<char> precompiled_charsmap = src.get_precompiled_charsmap();
  if (precompiled_charsmap.size() > emel::model::data::k_max_precompiled_charsmap_bytes) {
    std::fprintf(stderr,
                 "precompiled charsmap exceeds emel capacity: %zu > %d\n",
                 precompiled_charsmap.size(),
                 emel::model::data::k_max_precompiled_charsmap_bytes);
    return false;
  }
  if (!precompiled_charsmap.empty()) {
    std::memcpy(dst.precompiled_charsmap.data(),
                precompiled_charsmap.data(),
                precompiled_charsmap.size());
  }
  dst.precompiled_charsmap_size = static_cast<uint32_t>(precompiled_charsmap.size());

  dst.bos_id = src.token_bos();
  dst.eos_id = src.token_eos();
  dst.eot_id = src.token_eot();
  dst.eom_id = src.token_eom();
  dst.unk_id = src.token_unk();
  dst.sep_id = src.token_sep();
  dst.pad_id = src.token_pad();
  dst.mask_id = src.token_mask();
  dst.prefix_id = src.token_prefix();
  dst.suffix_id = src.token_suffix();
  dst.middle_id = src.token_middle();
  dst.fim_pre_id = src.token_fim_pre();
  dst.fim_suf_id = src.token_fim_suf();
  dst.fim_mid_id = src.token_fim_mid();
  dst.fim_pad_id = src.token_fim_pad();
  dst.fim_rep_id = src.token_fim_rep();
  dst.fim_sep_id = src.token_fim_sep();

  dst.add_bos = src.get_add_bos();
  dst.add_eos = src.get_add_eos();
  dst.add_sep = src.get_add_sep();
  dst.add_space_prefix = src.get_add_space_prefix();
  dst.ignore_merges = src.get_ignore_merges();
  dst.remove_extra_whitespaces = src.get_remove_extra_whitespaces();
  dst.escape_whitespaces = src.get_escape_whitespaces();
  dst.treat_whitespace_as_suffix = src.get_treat_whitespace_as_suffix();

  return true;
}

constexpr double k_f32_rtol = 1e-5;
constexpr double k_f32_atol = 1e-6;

constexpr int64_t k_vec_len = 512;
constexpr int64_t k_softmax_width = 64;
constexpr int64_t k_softmax_rows = 8;
constexpr int64_t k_mm_k = 48;
constexpr int64_t k_mm_m = 24;
constexpr int64_t k_mm_n = 32;

std::vector<float> make_signed_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const float wave = std::sin(static_cast<float>(i) * 0.013f) * scale;
    const float bucket = static_cast<float>((i % 29) - 14) * 0.03125f;
    out[static_cast<size_t>(i)] = wave + bucket + bias;
  }
  return out;
}

std::vector<float> make_positive_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out = make_signed_data(count, scale, bias);
  for (float & value : out) {
    value = std::fabs(value) + 0.5f;
  }
  return out;
}

template <class tensor_type>
void fill_default_nb(tensor_type & tensor) {
  constexpr uint64_t elem_size = sizeof(float);
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

emel::kernel::event::tensor_view make_src_view(const float * data,
                                               const uint64_t ne0,
                                               const uint64_t ne1 = 1,
                                               const uint64_t ne2 = 1,
                                               const uint64_t ne3 = 1) {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(tensor);
  return tensor;
}

emel::kernel::event::tensor_view_mut make_dst_view(float * data,
                                                   const uint64_t ne0,
                                                   const uint64_t ne1 = 1,
                                                   const uint64_t ne2 = 1,
                                                   const uint64_t ne3 = 1) {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(tensor);
  return tensor;
}

bool almost_equal_f32(const float actual, const float expected) {
  const double diff = std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
  const double tol = k_f32_atol + k_f32_rtol * std::fabs(static_cast<double>(expected));
  return diff <= tol;
}

bool compare_f32_vectors(const char * backend,
                         const char * case_name,
                         const std::vector<float> & actual,
                         const std::vector<float> & expected) {
  if (actual.size() != expected.size()) {
    std::fprintf(stderr,
                 "[%s] %s size mismatch: emel=%zu ggml=%zu\n",
                 backend,
                 case_name,
                 actual.size(),
                 expected.size());
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!almost_equal_f32(actual[i], expected[i])) {
      std::fprintf(stderr,
                   "[%s] %s mismatch at %zu: emel=%0.8f ggml=%0.8f\n",
                   backend,
                   case_name,
                   i,
                   actual[i],
                   expected[i]);
      return false;
    }
  }
  return true;
}

struct ggml_case_context {
  std::vector<uint8_t> arena;
  ggml_context * ctx = nullptr;

  explicit ggml_case_context(const size_t arena_bytes = 64u * 1024u * 1024u)
      : arena(arena_bytes) {
    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = false;
    ctx = ggml_init(params);
  }

  ~ggml_case_context() {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

void set_tensor_f32(ggml_tensor * tensor, const std::vector<float> & values) {
  std::memcpy(ggml_get_data_f32(tensor), values.data(), values.size() * sizeof(float));
}

bool compute_graph(ggml_case_context & c, ggml_tensor * out) {
  ggml_cgraph * graph = ggml_new_graph(c.ctx);
  if (graph == nullptr || out == nullptr) {
    return false;
  }
  ggml_build_forward_expand(graph, out);
  return ggml_graph_compute_with_ctx(c.ctx, graph, 1) == GGML_STATUS_SUCCESS;
}

template <class build_fn>
bool run_ggml_unary(const std::vector<float> & src,
                    std::vector<float> & out,
                    build_fn build) {
  ggml_case_context c{};
  ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, static_cast<int64_t>(src.size()));
  set_tensor_f32(a, src);
  ggml_tensor * out_tensor = build(c.ctx, a);
  if (!compute_graph(c, out_tensor)) {
    return false;
  }
  const float * out_data = ggml_get_data_f32(out_tensor);
  out.assign(out_data, out_data + src.size());
  return true;
}

template <class build_fn>
bool run_ggml_binary(const std::vector<float> & lhs,
                     const std::vector<float> & rhs,
                     std::vector<float> & out,
                     build_fn build) {
  ggml_case_context c{};
  ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, static_cast<int64_t>(lhs.size()));
  ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, static_cast<int64_t>(rhs.size()));
  set_tensor_f32(a, lhs);
  set_tensor_f32(b, rhs);
  ggml_tensor * out_tensor = build(c.ctx, a, b);
  if (!compute_graph(c, out_tensor)) {
    return false;
  }
  const float * out_data = ggml_get_data_f32(out_tensor);
  out.assign(out_data, out_data + lhs.size());
  return true;
}

bool run_ggml_softmax(const std::vector<float> & src, std::vector<float> & out) {
  ggml_case_context c{};
  ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_softmax_width, k_softmax_rows);
  set_tensor_f32(a, src);
  ggml_tensor * out_tensor = ggml_soft_max(c.ctx, a);
  if (!compute_graph(c, out_tensor)) {
    return false;
  }
  const float * out_data = ggml_get_data_f32(out_tensor);
  out.assign(out_data, out_data + src.size());
  return true;
}

bool run_ggml_mul_mat(const std::vector<float> & matrix_a,
                      const std::vector<float> & matrix_b,
                      std::vector<float> & out) {
  ggml_case_context c{};
  ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_mm_k, k_mm_n); // [n, k]
  ggml_tensor * b = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_mm_k, k_mm_m); // [m, k]
  set_tensor_f32(a, matrix_a);
  set_tensor_f32(b, matrix_b);

  ggml_tensor * out_tensor = ggml_mul_mat(c.ctx, a, b);
  if (!compute_graph(c, out_tensor)) {
    return false;
  }
  const float * out_data = ggml_get_data_f32(out_tensor);
  out.assign(out_data, out_data + static_cast<size_t>(k_mm_n * k_mm_m));
  return true;
}

template <class exec_fn>
bool run_backend_kernel_parity(const char * backend, exec_fn exec) {
  bool ok = true;

  auto fail = [&](const char * case_name, const char * reason) {
    std::fprintf(stderr, "[%s] %s failed: %s\n", backend, case_name, reason);
    ok = false;
  };

  {
    auto src = make_signed_data(k_vec_len, 1.25f, 0.1f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_dup ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_dup", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_dup(ctx, a);
               })) {
      fail("op_dup", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_dup", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_add ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_add", "emel rejected request");
    } else if (!run_ggml_binary(lhs, rhs, ggml_out, [](ggml_context * ctx, ggml_tensor * a,
                                                        ggml_tensor * b) {
                 return ggml_add(ctx, a, b);
               })) {
      fail("op_add", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_add", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sub ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_sub", "emel rejected request");
    } else if (!run_ggml_binary(lhs, rhs, ggml_out, [](ggml_context * ctx, ggml_tensor * a,
                                                        ggml_tensor * b) {
                 return ggml_sub(ctx, a, b);
               })) {
      fail("op_sub", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_sub", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.25f, 0.75f);
    auto rhs = make_signed_data(k_vec_len, 0.45f, 0.5f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_mul ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_mul", "emel rejected request");
    } else if (!run_ggml_binary(lhs, rhs, ggml_out, [](ggml_context * ctx, ggml_tensor * a,
                                                        ggml_tensor * b) {
                 return ggml_mul(ctx, a, b);
               })) {
      fail("op_mul", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_mul", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto lhs = make_positive_data(k_vec_len, 0.3f, 0.25f);
    auto rhs = make_positive_data(k_vec_len, 0.2f, 0.75f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_div ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_div", "emel rejected request");
    } else if (!run_ggml_binary(lhs, rhs, ggml_out, [](ggml_context * ctx, ggml_tensor * a,
                                                        ggml_tensor * b) {
                 return ggml_div(ctx, a, b);
               })) {
      fail("op_div", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_div", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.5f, 0.125f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sqr ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_sqr", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_sqr(ctx, a);
               })) {
      fail("op_sqr", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_sqr", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_positive_data(k_vec_len, 0.35f, 0.2f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sqrt ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_sqrt", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_sqrt(ctx, a);
               })) {
      fail("op_sqrt", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_sqrt", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_positive_data(k_vec_len, 0.4f, 0.125f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_log ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_log", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_log(ctx, a);
               })) {
      fail("op_log", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_log", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, 0.1f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sin ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_sin", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_sin(ctx, a);
               })) {
      fail("op_sin", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_sin", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, -0.2f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_cos ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_cos", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_cos(ctx, a);
               })) {
      fail("op_cos", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_cos", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_softmax_width * k_softmax_rows, 0.1f, 0.05f);
    std::vector<float> emel_out(static_cast<size_t>(k_softmax_width * k_softmax_rows));
    emel::kernel::event::op_soft_max ev{
      .src0 = make_src_view(src.data(),
                            static_cast<uint64_t>(k_softmax_width),
                            static_cast<uint64_t>(k_softmax_rows)),
      .dst = make_dst_view(emel_out.data(),
                           static_cast<uint64_t>(k_softmax_width),
                           static_cast<uint64_t>(k_softmax_rows)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_soft_max", "emel rejected request");
    } else if (!run_ggml_softmax(src, ggml_out)) {
      fail("op_soft_max", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_soft_max", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto matrix_b = make_signed_data(k_mm_k * k_mm_m, 0.12f, 0.25f); // [m, k]
    auto matrix_a = make_signed_data(k_mm_k * k_mm_n, 0.08f, -0.1f); // [n, k]
    std::vector<float> src1(static_cast<size_t>(k_mm_k * k_mm_n));
    for (int64_t p = 0; p < k_mm_k; ++p) {
      for (int64_t j = 0; j < k_mm_n; ++j) {
        src1[static_cast<size_t>(p * k_mm_n + j)] = matrix_a[static_cast<size_t>(j * k_mm_k + p)];
      }
    }
    std::vector<float> emel_out(static_cast<size_t>(k_mm_n * k_mm_m));
    emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix_b.data(), static_cast<uint64_t>(k_mm_k), static_cast<uint64_t>(k_mm_m)),
      .src1 = make_src_view(src1.data(), static_cast<uint64_t>(k_mm_n), static_cast<uint64_t>(k_mm_k)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_mm_n), static_cast<uint64_t>(k_mm_m)),
      .nth = 1,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_mul_mat", "emel rejected request");
    } else if (!run_ggml_mul_mat(matrix_a, matrix_b, ggml_out)) {
      fail("op_mul_mat", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_mul_mat", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::neg,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_unary_neg", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_neg(ctx, a);
               })) {
      fail("op_unary_neg", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_unary_neg", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::relu,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_unary_relu", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_relu(ctx, a);
               })) {
      fail("op_unary_relu", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_unary_relu", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.35f, 0.1f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::exp,
    };
    std::vector<float> ggml_out;
    if (!exec(ev)) {
      fail("op_unary_exp", "emel rejected request");
    } else if (!run_ggml_unary(src, ggml_out, [](ggml_context * ctx, ggml_tensor * a) {
                 return ggml_exp(ctx, a);
               })) {
      fail("op_unary_exp", "ggml execution failed");
    } else if (!compare_f32_vectors(backend, "op_unary_exp", emel_out, ggml_out)) {
      ok = false;
    }
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, 0.0f);
    std::vector<float> emel_out(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sum ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(emel_out.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (exec(ev)) {
      fail("op_sum", "expected unsupported op to be rejected");
    }
  }

  return ok;
}

int run_kernel_parity(const emel::paritychecker::parity_options &) {
  emel::kernel::x86_64::sm x86_machine{};
  emel::kernel::aarch64::sm aarch_machine{};

  auto x86_exec = [&](const auto & ev) {
    return x86_machine.process_event(ev);
  };
  auto aarch_exec = [&](const auto & ev) {
    return aarch_machine.process_event(ev);
  };

  const bool x86_ok = run_backend_kernel_parity("x86_64", x86_exec);
  const bool aarch_ok = run_backend_kernel_parity("aarch64", aarch_exec);

  if (x86_ok && aarch_ok) {
    std::fprintf(stdout, "kernel parity ok\n");
    return 0;
  }
  return 1;
}

int run_tokenizer_parity(const emel::paritychecker::parity_options & opts) {
  llama_backend_guard backend_guard{};

  llama_model_params model_params = llama_model_default_params();
  model_params.vocab_only = true;
  model_params.check_tensors = false;

  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      llama_model_load_from_file(opts.model_path.c_str(), model_params),
      llama_model_free);
  if (model == nullptr) {
    std::fprintf(stderr, "failed to load model: %s\n", opts.model_path.c_str());
    return 1;
  }

  const llama_vocab * llama_vocab_ptr = llama_model_get_vocab(model.get());
  if (llama_vocab_ptr == nullptr) {
    std::fprintf(stderr, "model has no vocabulary: %s\n", opts.model_path.c_str());
    return 1;
  }

  auto emel_vocab = std::make_unique<emel::model::data::vocab>();
  if (!load_emel_vocab_from_llama(*llama_vocab_ptr, *emel_vocab)) {
    std::fprintf(stderr, "failed to map llama vocab into emel layout\n");
    return 1;
  }

  using tokenizer_model = emel::model::data::tokenizer_model;
  switch (emel_vocab->tokenizer_model_id) {
    case tokenizer_model::SPM:
      return emel::paritychecker::run_tokenizer_spm_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
    case tokenizer_model::BPE:
      return emel::paritychecker::run_tokenizer_bpe_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
    case tokenizer_model::WPM:
      return emel::paritychecker::run_tokenizer_wpm_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
    case tokenizer_model::UGM:
      return emel::paritychecker::run_tokenizer_ugm_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
    case tokenizer_model::RWKV:
      return emel::paritychecker::run_tokenizer_rwkv_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
    case tokenizer_model::PLAMO2:
      return emel::paritychecker::run_tokenizer_plamo2_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
    case tokenizer_model::NONE:
    case tokenizer_model::UNKNOWN:
    default:
      return emel::paritychecker::run_tokenizer_fallback_parity(
          opts, *llama_vocab_ptr, *emel_vocab);
  }
}

int run_gbnf_parser_parity(const emel::paritychecker::parity_options & opts) {
  emel::gbnf::grammar emel_grammar{};
  int32_t emel_err = k_error_ok;
  const bool emel_ok = run_emel_gbnf_parse(opts.text, emel_grammar, emel_err);

  llama_grammar_rules llama_rules;
  const bool llama_ok = run_llama_gbnf_parse(opts.text, llama_rules);

  if (emel_ok != llama_ok) {
    std::fprintf(stderr,
                 "parse outcome mismatch: emel=%s llama=%s (emel_err=%d)\n",
                 emel_ok ? "ok" : "error",
                 llama_ok ? "ok" : "error",
                 emel_err);
    if (opts.dump) {
      if (emel_ok) {
        dump_emel_grammar(emel_grammar);
      }
      if (llama_ok) {
        dump_llama_grammar(llama_rules);
      }
    }
    return 1;
  }

  if (!emel_ok) {
    std::fprintf(stdout, "parity ok (both parsers rejected grammar)\n");
    return 0;
  }

  const bool matched = compare_grammars(emel_grammar, llama_rules);
  if (!matched) {
    if (opts.dump) {
      dump_emel_grammar(emel_grammar);
      dump_llama_grammar(llama_rules);
    }
    return 1;
  }

  if (opts.dump) {
    dump_emel_grammar(emel_grammar);
  }
  std::fprintf(stdout,
               "parity ok (%u rules, %u elements)\n",
               emel_grammar.rule_count,
               emel_grammar.element_count);
  return 0;
}

int run_jinja_parity(const emel::paritychecker::parity_options & opts) {
  emel::text::jinja::program emel_program{};
  int32_t emel_parse_err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t emel_parse_error_pos = 0;
  const bool emel_parse_ok = run_emel_jinja_parse(
      opts.text, emel_program, emel_parse_err, emel_parse_error_pos);

  ::jinja::program reference_program;
  const bool reference_parse_ok = run_reference_jinja_parse(opts.text, reference_program);

  if (emel_parse_ok != reference_parse_ok) {
    std::fprintf(stderr,
                 "jinja parse outcome mismatch: emel=%s reference=%s (emel_err=%d at %zu)\n",
                 emel_parse_ok ? "ok" : "error",
                 reference_parse_ok ? "ok" : "error",
                 emel_parse_err,
                 emel_parse_error_pos);
    return 1;
  }

  if (!emel_parse_ok) {
    std::fprintf(stdout, "jinja parity ok (both parsers rejected template)\n");
    return 0;
  }

  std::string emel_rendered;
  if (!run_emel_jinja_render(emel_program, opts.text, emel_rendered)) {
    std::fprintf(stderr, "jinja render failed in emel formatter\n");
    return 1;
  }

  std::string reference_rendered;
  if (!run_reference_jinja_render(reference_program, reference_rendered)) {
    std::fprintf(stderr, "jinja render failed in reference runtime\n");
    return 1;
  }

  const std::string_view emel_cmp = strip_trailing_newline(emel_rendered);
  const std::string_view reference_cmp = strip_trailing_newline(reference_rendered);
  if (emel_cmp != reference_cmp) {
    std::fprintf(stderr,
                 "jinja render mismatch: emel_len=%zu reference_len=%zu\n",
                 emel_rendered.size(),
                 reference_rendered.size());
    if (opts.dump) {
      std::fprintf(stdout, "emel:\n%s\n", emel_rendered.c_str());
      std::fprintf(stdout, "reference:\n%s\n", reference_rendered.c_str());
    }
    return 1;
  }

  std::fprintf(stdout, "jinja parity ok (output bytes=%zu)\n", emel_rendered.size());
  return 0;
}

}  // namespace

namespace emel::paritychecker {

int run_parity(const parity_options & opts) {
  switch (opts.mode) {
    case parity_mode::gbnf_parser:
      return run_gbnf_parser_parity(opts);
    case parity_mode::kernel:
      return run_kernel_parity(opts);
    case parity_mode::jinja:
      return run_jinja_parity(opts);
    case parity_mode::tokenizer:
    default:
      return run_tokenizer_parity(opts);
  }
}

}  // namespace emel::paritychecker
