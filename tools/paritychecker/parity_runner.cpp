#include "parity_runner.hpp"
#include "tokenizer_parity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/sm.hpp"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/jinja/formatter/sm.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

#include "ggml-cpu.h"
#include "ggml.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "llama.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

namespace {

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_internal = 3;
constexpr const char * k_generation_fixture_name = "Llama-68M-Chat-v1-Q2_K.gguf";
constexpr size_t k_generation_output_capacity = 256u;

using llama_model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

bool file_exists(const std::string & path) {
  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

std::filesystem::path expected_generation_fixture_path() {
#ifdef PARITYCHECKER_REPO_ROOT
  const std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "models" / k_generation_fixture_name;
#else
  return std::filesystem::path("tests") / "models" / k_generation_fixture_name;
#endif
}

std::filesystem::path normalize_fixture_path(const std::filesystem::path & path) {
  std::error_code ec;
  const std::filesystem::path absolute_path = std::filesystem::absolute(path, ec);
  if (ec) {
    return {};
  }
  return absolute_path.lexically_normal();
}

bool is_expected_generation_fixture(const std::string & model_path) {
  const std::filesystem::path expected_path =
      normalize_fixture_path(expected_generation_fixture_path());
  const std::filesystem::path provided_path =
      normalize_fixture_path(std::filesystem::path(model_path));
  return !expected_path.empty() && !provided_path.empty() && expected_path == provided_path;
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

void silence_llama_log(ggml_log_level, const char *, void *) {}

struct llama_log_silencer {
  ggml_log_callback callback = nullptr;
  void * user_data = nullptr;

  llama_log_silencer() {
    llama_log_get(&callback, &user_data);
    llama_log_set(silence_llama_log, nullptr);
  }

  ~llama_log_silencer() {
    llama_log_set(callback, user_data);
  }
};

template <size_t k_array_size>
void copy_name(std::array<char, k_array_size> & dst, const std::string_view value) {
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

struct gguf_capture {
  bool probe_done = false;
  bool probe_error = false;
  bool bind_done = false;
  bool bind_error = false;
  bool parse_done = false;
  bool parse_error = false;
  emel::gguf::loader::requirements requirements = {};
  emel::error::type err = emel::error::cast(emel::gguf::loader::error::none);
};

struct weight_capture {
  bool bind_done = false;
  bool bind_error = false;
  bool plan_done = false;
  bool plan_error = false;
  bool apply_done = false;
  bool apply_error = false;
  uint32_t effect_count = 0u;
  emel::error::type err = emel::error::cast(emel::model::weight_loader::error::none);
};

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0u;
  uint64_t bytes_done = 0u;
  bool used_mmap = false;
};

struct initialize_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
};

struct generation_capture {
  bool done = false;
  bool error = false;
  const emel::generator::event::generate * request = nullptr;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct generation_result {
  std::array<char, k_generation_output_capacity> output = {};
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct initialize_backend {
  llama_model_ptr model = {nullptr, llama_model_free};
  llama_context_ptr decode_ctx = {nullptr, llama_free};
  const llama_vocab * vocab = nullptr;
  int32_t vocab_size = 0;
  int32_t fallback_token_id = 0;
};

emel::error::type sampler_select_argmax(int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out) {
  int32_t best_index = 0;
  float best_score = (&candidate_scores)[0];
  for (int32_t idx = 1; idx < candidate_count; ++idx) {
    if ((&candidate_scores)[idx] > best_score) {
      best_score = (&candidate_scores)[idx];
      best_index = idx;
    }
  }

  selected_token_out = (&candidate_ids)[best_index];
  return emel::error::cast(emel::logits::sampler::error::none);
}

struct generation_load_state {
  std::unique_ptr<emel::model::data> model_data = std::make_unique<emel::model::data>();
  std::vector<uint8_t> file_bytes = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::model::weight_loader::sm weight_loader = {};
  emel::model::loader::sm model_loader = {};
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  std::unique_ptr<emel::generator::sm> generator = {};
  initialize_backend backend = {};
  std::array<emel::logits::sampler::fn, 1> samplers = {
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  int model_topology = 1;
  int prefill_plan = 2;
  int decode_plan = 3;
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::vector<emel::model::weight_loader::effect_request> effect_requests = {};
  std::vector<emel::model::weight_loader::effect_result> effect_results = {};
  gguf_capture gguf = {};
  weight_capture weight = {};
  load_capture load = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
};

uint32_t read_u32_le(const std::span<const uint8_t> bytes) {
  uint32_t value = 0u;
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    value |= static_cast<uint32_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

uint64_t read_u64_le(const std::span<const uint8_t> bytes) {
  uint64_t value = 0u;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out) {
  out.clear();

  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }

  const bool seek_end_ok = std::fseek(file, 0, SEEK_END) == 0;
  const long file_size = seek_end_ok ? std::ftell(file) : -1L;
  const bool seek_start_ok = file_size >= 0 && std::fseek(file, 0, SEEK_SET) == 0;
  if (!seek_end_ok || file_size < 0 || !seek_start_ok) {
    std::fclose(file);
    return false;
  }

  out.resize(static_cast<size_t>(file_size));
  const size_t read_size = out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), file);
  const bool read_ok = read_size == out.size();
  std::fclose(file);
  return read_ok;
}

const char * model_loader_error_name(const emel::error::type err) {
  using error = emel::model::loader::error;

  switch (err) {
    case emel::error::cast(error::none):
      return "none";
    case emel::error::cast(error::invalid_request):
      return "invalid_request";
    case emel::error::cast(error::parse_failed):
      return "parse_failed";
    case emel::error::cast(error::backend_error):
      return "backend_error";
    case emel::error::cast(error::model_invalid):
      return "model_invalid";
    case emel::error::cast(error::internal_error):
      return "internal_error";
    case emel::error::cast(error::untracked):
      return "untracked";
    default:
      return "unknown";
  }
}

const char * generator_error_name(const emel::error::type err) {
  using error = emel::generator::error;

  switch (err) {
    case emel::error::cast(error::none):
      return "none";
    case emel::error::cast(error::invalid_request):
      return "invalid_request";
    case emel::error::cast(error::backend):
      return "backend";
    default:
      return "unknown";
  }
}

emel::error::type map_gguf_error(const emel::error::type err) {
  using gguf_error = emel::gguf::loader::error;
  using model_error = emel::model::loader::error;

  switch (err) {
    case emel::error::cast(gguf_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(gguf_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(gguf_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(gguf_error::capacity):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(gguf_error::parse_failed):
      return emel::error::cast(model_error::parse_failed);
    case emel::error::cast(gguf_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(gguf_error::untracked):
      return emel::error::cast(model_error::untracked);
    default:
      return emel::error::cast(model_error::untracked);
  }
}

emel::error::type map_weight_loader_error(const emel::error::type err) {
  using weight_error = emel::model::weight_loader::error;
  using model_error = emel::model::loader::error;

  switch (err) {
    case emel::error::cast(weight_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(weight_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(weight_error::capacity):
    case emel::error::cast(weight_error::backend_error):
    case emel::error::cast(weight_error::out_of_memory):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(weight_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(weight_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(weight_error::untracked):
      return emel::error::cast(model_error::untracked);
    default:
      return emel::error::cast(model_error::untracked);
  }
}

void reset_gguf_capture(generation_load_state & state) {
  state.gguf = {};
}

void reset_weight_capture(generation_load_state & state) {
  state.weight = {};
}

void reset_load_capture(generation_load_state & state) {
  state.load = {};
}

void reset_initialize_capture(generation_load_state & state) {
  state.initialize = {};
}

void reset_generation_capture(generation_load_state & state) {
  state.generation = {};
}

void on_probe_done(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.gguf.probe_done = true;
  state.gguf.probe_error = false;
  state.gguf.requirements = ev.requirements_out;
}

void on_probe_error(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.gguf.probe_done = false;
  state.gguf.probe_error = true;
  state.gguf.err = ev.err;
}

void on_bind_done(void * owner, const emel::gguf::loader::events::bind_done &) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.gguf.bind_done = true;
  state.gguf.bind_error = false;
}

void on_bind_error(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.gguf.bind_done = false;
  state.gguf.bind_error = true;
  state.gguf.err = ev.err;
}

void on_parse_done(void * owner, const emel::gguf::loader::events::parse_done &) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.gguf.parse_done = true;
  state.gguf.parse_error = false;
}

void on_parse_error(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.gguf.parse_done = false;
  state.gguf.parse_error = true;
  state.gguf.err = ev.err;
}

void on_weight_bind_done(void * owner, const emel::model::weight_loader::events::bind_done &) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.weight.bind_done = true;
  state.weight.bind_error = false;
}

void on_weight_bind_error(void * owner,
                          const emel::model::weight_loader::events::bind_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.weight.bind_done = false;
  state.weight.bind_error = true;
  state.weight.err = ev.err;
}

void on_weight_plan_done(void * owner,
                         const emel::model::weight_loader::events::plan_done & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.weight.plan_done = true;
  state.weight.plan_error = false;
  state.weight.effect_count = ev.effect_count;
}

void on_weight_plan_error(void * owner,
                          const emel::model::weight_loader::events::plan_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.weight.plan_done = false;
  state.weight.plan_error = true;
  state.weight.err = ev.err;
}

void on_weight_apply_done(void * owner, const emel::model::weight_loader::events::apply_done &) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.weight.apply_done = true;
  state.weight.apply_error = false;
}

void on_weight_apply_error(void * owner,
                           const emel::model::weight_loader::events::apply_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.weight.apply_done = false;
  state.weight.apply_error = true;
  state.weight.err = ev.err;
}

void on_load_done(void * owner, const emel::model::loader::events::load_done & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.load.done = true;
  state.load.error = false;
  state.load.err = emel::error::cast(emel::model::loader::error::none);
  state.load.bytes_total = ev.bytes_total;
  state.load.bytes_done = ev.bytes_done;
  state.load.used_mmap = ev.used_mmap;
}

void on_load_error(void * owner, const emel::model::loader::events::load_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.load.done = false;
  state.load.error = true;
  state.load.err = ev.err;
}

void on_initialize_done(void * owner, const emel::generator::events::initialize_done &) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.initialize.done = true;
  state.initialize.error = false;
  state.initialize.err = emel::error::cast(emel::generator::error::none);
}

void on_initialize_error(void * owner, const emel::generator::events::initialize_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.initialize.done = false;
  state.initialize.error = true;
  state.initialize.err = ev.err;
}

void on_generation_done(void * owner, const emel::generator::events::generation_done & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.generation.done = true;
  state.generation.error = false;
  state.generation.request = ev.request;
  state.generation.err = emel::error::cast(emel::generator::error::none);
  state.generation.tokens_generated = ev.tokens_generated;
  state.generation.output_length = ev.output_length;
}

void on_generation_error(void * owner, const emel::generator::events::generation_error & ev) {
  auto & state = *static_cast<generation_load_state *>(owner);
  state.generation.done = false;
  state.generation.error = true;
  state.generation.request = ev.request;
  state.generation.err = ev.err;
  state.generation.tokens_generated = ev.tokens_generated;
  state.generation.output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void * tokenizer_sm,
                             const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(
    void * tokenizer_sm,
    const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool backend_validate(const emel::graph::processor::event::execute & request, int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<initialize_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return request.compute_ctx != nullptr && backend != nullptr && backend->decode_ctx != nullptr &&
         backend->vocab != nullptr && backend->vocab_size > 0;
}

bool backend_prepare_graph(const emel::graph::processor::event::execute &,
                           bool * reused_out,
                           int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_alloc_graph(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_bind_inputs(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_run_kernel(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_extract_outputs(const emel::graph::processor::event::execute & request,
                             int32_t * outputs_out,
                             int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<initialize_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  if (err_out != nullptr) {
    *err_out = 0;
  }
  if (io == nullptr || backend == nullptr || io->logits == nullptr || io->logits_capacity <= 0) {
    if (err_out != nullptr) {
      *err_out = 1;
    }
    return false;
  }
  if (backend->decode_ctx == nullptr || backend->vocab == nullptr || io->token_ids == nullptr ||
      io->token_count <= 0 || backend->vocab_size <= 0 ||
      backend->vocab_size > io->logits_capacity) {
    if (err_out != nullptr) {
      *err_out = 1;
    }
    return false;
  }

  llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(io->token_ids), io->token_count);
  const int32_t decode_status = llama_decode(backend->decode_ctx.get(), batch);
  if (decode_status != 0) {
    if (err_out != nullptr) {
      *err_out = decode_status;
    }
    return false;
  }

  float * logits = llama_get_logits_ith(backend->decode_ctx.get(), -1);
  if (logits == nullptr) {
    if (err_out != nullptr) {
      *err_out = 1;
    }
    return false;
  }

  for (int32_t idx = 0; idx < backend->vocab_size; ++idx) {
    io->logits[idx] = logits[idx];
  }
  for (int32_t idx = backend->vocab_size; idx < io->logits_capacity; ++idx) {
    io->logits[idx] = -1.0f;
  }
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  return true;
}

emel::text::tokenizer::preprocessor::preprocessor_kind generation_preprocessor_variant(
    const emel::model::data & model_data) {
  using tokenizer_model = emel::model::data::tokenizer_model;
  using preprocessor_kind = emel::text::tokenizer::preprocessor::preprocessor_kind;

  switch (model_data.vocab_data.tokenizer_model_id) {
    case tokenizer_model::SPM:
      return preprocessor_kind::spm;
    case tokenizer_model::BPE:
      return preprocessor_kind::bpe;
    case tokenizer_model::WPM:
      return preprocessor_kind::wpm;
    case tokenizer_model::UGM:
      return preprocessor_kind::ugm;
    case tokenizer_model::RWKV:
      return preprocessor_kind::rwkv;
    case tokenizer_model::PLAMO2:
      return preprocessor_kind::plamo2;
    case tokenizer_model::NONE:
    case tokenizer_model::UNKNOWN:
    default:
      return preprocessor_kind::fallback;
  }
}

emel::text::encoders::encoder_kind generation_encoder_variant(
    const emel::model::data & model_data) {
  using tokenizer_model = emel::model::data::tokenizer_model;
  using encoder_kind = emel::text::encoders::encoder_kind;

  switch (model_data.vocab_data.tokenizer_model_id) {
    case tokenizer_model::SPM:
      return encoder_kind::spm;
    case tokenizer_model::BPE:
      return encoder_kind::bpe;
    case tokenizer_model::WPM:
      return encoder_kind::wpm;
    case tokenizer_model::UGM:
      return encoder_kind::ugm;
    case tokenizer_model::RWKV:
      return encoder_kind::rwkv;
    case tokenizer_model::PLAMO2:
      return encoder_kind::plamo2;
    case tokenizer_model::NONE:
    case tokenizer_model::UNKNOWN:
    default:
      return encoder_kind::fallback;
  }
}

std::string_view vocab_token_view(const emel::model::data::vocab & vocab,
                                  const int32_t token_id) {
  if (token_id < 0 || static_cast<uint32_t>(token_id) >= vocab.n_tokens) {
    return {};
  }

  const auto & entry = vocab.entries[static_cast<size_t>(token_id)];
  const size_t begin = static_cast<size_t>(entry.text_offset);
  const size_t length = static_cast<size_t>(entry.text_length);
  if (begin + length > static_cast<size_t>(vocab.token_bytes_used)) {
    return {};
  }

  return std::string_view{vocab.token_storage.data() + begin, length};
}

bool is_printable_ascii_token(const std::string_view piece) {
  if (piece.empty()) {
    return false;
  }

  for (const char ch : piece) {
    const uint8_t byte = static_cast<uint8_t>(ch);
    if (byte < 0x20u || byte > 0x7eu) {
      return false;
    }
  }

  return true;
}

int32_t find_generation_fallback_token_id(const emel::model::data::vocab & vocab) {
  constexpr std::array<std::string_view, 4> k_preferred_tokens = {
      "hello",
      "world",
      "Hello",
      "!",
  };

  for (const std::string_view preferred : k_preferred_tokens) {
    for (uint32_t token_id = 0; token_id < vocab.n_tokens; ++token_id) {
      if (vocab_token_view(vocab, static_cast<int32_t>(token_id)) == preferred) {
        return static_cast<int32_t>(token_id);
      }
    }
  }

  const int32_t normal_type = static_cast<int32_t>(LLAMA_TOKEN_TYPE_NORMAL);
  for (uint32_t token_id = 0; token_id < vocab.n_tokens; ++token_id) {
    const auto & entry = vocab.entries[token_id];
    const std::string_view piece = vocab_token_view(vocab, static_cast<int32_t>(token_id));
    if (entry.type == normal_type && is_printable_ascii_token(piece)) {
      return static_cast<int32_t>(token_id);
    }
  }

  return vocab.eos_id >= 0 ? vocab.eos_id : 0;
}

bool load_generation_vocab_from_llama(const std::string & model_path,
                                      generation_load_state & state) {
  llama_model_params model_params = llama_model_default_params();
  model_params.check_tensors = false;

  llama_model_ptr model(
      llama_model_load_from_file(model_path.c_str(), model_params),
      llama_model_free);
  if (model == nullptr) {
    return false;
  }

  const llama_vocab * vocab_ptr = llama_model_get_vocab(model.get());
  if (vocab_ptr == nullptr) {
    return false;
  }

  if (!load_emel_vocab_from_llama(*vocab_ptr, state.model_data->vocab_data)) {
    return false;
  }

  state.model_data->params.n_vocab =
      static_cast<int32_t>(state.model_data->vocab_data.n_tokens);
  state.backend.fallback_token_id =
      find_generation_fallback_token_id(state.model_data->vocab_data);
  state.backend.vocab = vocab_ptr;
  state.backend.vocab_size = llama_vocab_n_tokens(vocab_ptr);

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 0;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.n_threads = 1;
  context_params.n_threads_batch = 1;
  context_params.embeddings = false;
  state.backend.decode_ctx = llama_context_ptr{
      llama_init_from_model(model.get(), context_params),
      llama_free,
  };
  if (state.backend.decode_ctx == nullptr) {
    state.backend.vocab = nullptr;
    state.backend.vocab_size = 0;
    return false;
  }

  state.backend.model = std::move(model);
  return true;
}

emel::error::type run_emel_initialize_generator(
    generation_load_state & state,
    const emel::paritychecker::parity_options & opts) {
  if (state.model_data == nullptr) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  const int32_t prompt_capacity =
      std::max<int32_t>(32, static_cast<int32_t>(opts.text.size()) + 8);
  const int32_t decode_capacity = std::max<int32_t>(4, opts.max_tokens);
  const int32_t block_capacity = std::max<int32_t>(8, prompt_capacity + decode_capacity);

  state.generator = std::make_unique<emel::generator::sm>(
      *state.model_data,
      state.conditioner,
      nullptr,
      emel::text::formatter::format_raw);

  reset_initialize_capture(state);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::event::initialize request{
    &state.model_topology,
    &state.prefill_plan,
    &state.decode_plan,
    &state.tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
    &state.backend,
    backend_validate,
    backend_prepare_graph,
    backend_alloc_graph,
    backend_bind_inputs,
    backend_run_kernel,
    backend_extract_outputs,
    std::span<emel::logits::sampler::fn>{state.samplers},
  };
  request.preprocessor_variant = generation_preprocessor_variant(*state.model_data);
  request.encoder_variant = generation_encoder_variant(*state.model_data);
  request.add_special = false;
  request.parse_special = false;
  request.max_node_count = 8;
  request.max_tensor_count = 8;
  request.bytes_per_tensor = 4;
  request.workspace_capacity_bytes = 4096;
  request.max_prompt_tokens = prompt_capacity;
  request.max_generated_tokens = decode_capacity;
  request.max_blocks = block_capacity;
  request.block_tokens = 16;
  request.strip_leading_space = false;
  request.error_out = &error_out;
  request.on_done = {&state, on_initialize_done};
  request.on_error = {&state, on_initialize_error};

  const bool accepted = state.generator->process_event(request);
  if (accepted && state.initialize.done && !state.initialize.error) {
    return emel::error::cast(emel::generator::error::none);
  }
  if (state.initialize.error) {
    return state.initialize.err;
  }
  if (error_out != emel::error::cast(emel::generator::error::none)) {
    return error_out;
  }
  return emel::error::cast(emel::generator::error::invalid_request);
}

emel::error::type run_emel_generate(generation_load_state & state,
                                    const emel::paritychecker::parity_options & opts,
                                    std::span<char> output,
                                    size_t & output_length_out) {
  if (state.generator == nullptr) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  reset_generation_capture(state);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::event::generate request{
    opts.text,
    opts.max_tokens,
    output,
    output_length_out,
  };
  request.error_out = &error_out;
  request.on_done = {&state, on_generation_done};
  request.on_error = {&state, on_generation_error};

  const bool accepted = state.generator->process_event(request);
  if (accepted && state.generation.done && !state.generation.error) {
    return emel::error::cast(emel::generator::error::none);
  }
  if (state.generation.error) {
    return state.generation.err;
  }
  if (error_out != emel::error::cast(emel::generator::error::none)) {
    return error_out;
  }
  return emel::error::cast(emel::generator::error::invalid_request);
}

llama_token select_argmax_token_from_logits(const float * logits, const int32_t vocab_size) {
  int32_t best_index = 0;
  float best_score = logits[0];
  for (int32_t idx = 1; idx < vocab_size; ++idx) {
    if (logits[idx] > best_score) {
      best_score = logits[idx];
      best_index = idx;
    }
  }
  return static_cast<llama_token>(best_index);
}

bool tokenize_reference_prompt(const initialize_backend & backend,
                               const emel::paritychecker::parity_options & opts,
                               std::vector<llama_token> & tokens_out) {
  if (backend.vocab == nullptr) {
    return false;
  }

  int32_t token_capacity = static_cast<int32_t>(opts.text.size()) + 8;
  token_capacity = std::max(token_capacity, 8);
  tokens_out.resize(static_cast<size_t>(token_capacity));
  int32_t token_count = llama_tokenize(
      backend.vocab,
      opts.text.c_str(),
      static_cast<int32_t>(opts.text.size()),
      tokens_out.data(),
      token_capacity,
      false,
      false);
  if (token_count < 0) {
    token_capacity = -token_count;
    tokens_out.resize(static_cast<size_t>(token_capacity));
    token_count = llama_tokenize(
        backend.vocab,
        opts.text.c_str(),
        static_cast<int32_t>(opts.text.size()),
        tokens_out.data(),
        token_capacity,
        false,
        false);
  }
  if (token_count <= 0) {
    return false;
  }

  tokens_out.resize(static_cast<size_t>(token_count));
  return true;
}

bool append_reference_piece(const initialize_backend & backend,
                            const llama_token token,
                            generation_result & result_out) {
  if (backend.vocab == nullptr || result_out.output_length >= result_out.output.size()) {
    return false;
  }

  if (llama_vocab_is_control(backend.vocab, token) || llama_vocab_is_eog(backend.vocab, token)) {
    return true;
  }

  const char * piece = llama_vocab_get_text(backend.vocab, token);
  if (piece == nullptr) {
    return false;
  }

  const size_t piece_len = std::strlen(piece);
  if (result_out.output_length + piece_len > result_out.output.size()) {
    return false;
  }

  if (piece_len > 0u) {
    std::memcpy(result_out.output.data() + result_out.output_length, piece, piece_len);
  }
  result_out.output_length += piece_len;
  return true;
}

llama_context_ptr make_reference_context(initialize_backend & backend) {
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 0;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.n_threads = 1;
  context_params.n_threads_batch = 1;
  context_params.embeddings = false;
  return llama_context_ptr{
      backend.model != nullptr ? llama_init_from_model(backend.model.get(), context_params) : nullptr,
      llama_free,
  };
}

emel::error::type run_reference_generate(initialize_backend & backend,
                                         const emel::paritychecker::parity_options & opts,
                                         generation_result & result_out) {
  if (backend.model == nullptr || backend.vocab == nullptr || backend.vocab_size <= 0) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  llama_context_ptr ctx = make_reference_context(backend);
  if (ctx == nullptr) {
    return emel::error::cast(emel::generator::error::backend);
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(backend, opts, prompt_tokens)) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  llama_batch prompt_batch =
      llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
  if (llama_decode(ctx.get(), prompt_batch) != 0) {
    return emel::error::cast(emel::generator::error::backend);
  }

  for (int32_t step = 0; step < opts.max_tokens; ++step) {
    float * logits = llama_get_logits_ith(ctx.get(), -1);
    if (logits == nullptr) {
      return emel::error::cast(emel::generator::error::backend);
    }

    const llama_token selected = select_argmax_token_from_logits(logits, backend.vocab_size);
    result_out.tokens_generated += 1;
    if (!append_reference_piece(backend, selected, result_out)) {
      return emel::error::cast(emel::generator::error::backend);
    }
    if (llama_vocab_is_eog(backend.vocab, selected)) {
      break;
    }

    llama_token next_token = selected;
    llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
    if (llama_decode(ctx.get(), decode_batch) != 0) {
      return emel::error::cast(emel::generator::error::backend);
    }
  }

  return emel::error::cast(emel::generator::error::none);
}

size_t first_mismatch_offset(const generation_result & lhs, const generation_result & rhs) {
  const size_t shared = std::min(lhs.output_length, rhs.output_length);
  for (size_t idx = 0; idx < shared; ++idx) {
    if (lhs.output[idx] != rhs.output[idx]) {
      return idx;
    }
  }
  return shared;
}

bool generation_results_match(const generation_result & emel_result,
                              const generation_result & reference_result) {
  return emel_result.tokens_generated == reference_result.tokens_generated &&
         emel_result.output_length == reference_result.output_length &&
         std::string_view{emel_result.output.data(), emel_result.output_length} ==
             std::string_view{reference_result.output.data(), reference_result.output_length};
}

void dump_generation_result(const char * label, const generation_result & result) {
  std::fprintf(stdout,
               "%s: generated_tokens=%d output_bytes=%zu text=%.*s\n",
               label,
               result.tokens_generated,
               result.output_length,
               static_cast<int>(result.output_length),
               result.output.data());
}

std::string_view kv_key_view(const generation_load_state & state,
                             const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.key_offset) + static_cast<size_t>(entry.key_length) >
      state.kv_arena.size()) {
    return {};
  }

  return std::string_view{
    reinterpret_cast<const char *>(state.kv_arena.data() + entry.key_offset),
    entry.key_length,
  };
}

std::span<const uint8_t> kv_value_view(const generation_load_state & state,
                                       const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.value_offset) + static_cast<size_t>(entry.value_length) >
      state.kv_arena.size()) {
    return {};
  }

  return std::span<const uint8_t>{
    state.kv_arena.data() + entry.value_offset,
    entry.value_length,
  };
}

const emel::gguf::loader::kv_entry * find_kv_entry(const generation_load_state & state,
                                                   const std::string_view key) {
  for (const auto & entry : state.kv_entries) {
    if (kv_key_view(state, entry) == key) {
      return &entry;
    }
  }
  return nullptr;
}

bool decode_integer_value(const generation_load_state & state,
                          const emel::gguf::loader::kv_entry & entry,
                          uint64_t & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(state, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  switch (entry.value_type) {
    case constants::gguf_type_uint8:
      if (bytes.size() != 1u) {
        return false;
      }
      value_out = bytes[0];
      return true;
    case constants::gguf_type_int8:
      if (bytes.size() != 1u) {
        return false;
      }
      value_out = static_cast<uint64_t>(static_cast<int8_t>(bytes[0]));
      return true;
    case constants::gguf_type_uint16:
    case constants::gguf_type_int16:
      if (bytes.size() != 2u) {
        return false;
      }
      value_out = static_cast<uint64_t>(bytes[0]) |
                  (static_cast<uint64_t>(bytes[1]) << 8u);
      return true;
    case constants::gguf_type_uint32:
    case constants::gguf_type_int32:
      if (bytes.size() != sizeof(uint32_t)) {
        return false;
      }
      value_out = read_u32_le(bytes);
      return true;
    case constants::gguf_type_uint64:
    case constants::gguf_type_int64:
      if (bytes.size() != sizeof(uint64_t)) {
        return false;
      }
      value_out = read_u64_le(bytes);
      return true;
    default:
      return false;
  }
}

bool decode_string_value(const generation_load_state & state,
                         const emel::gguf::loader::kv_entry & entry,
                         std::string_view & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(state, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_string ||
      bytes.size() < sizeof(uint64_t)) {
    return false;
  }

  const uint64_t length = read_u64_le(bytes.first(sizeof(uint64_t)));
  if (length > bytes.size() - sizeof(uint64_t)) {
    return false;
  }

  value_out = std::string_view{
    reinterpret_cast<const char *>(bytes.data() + sizeof(uint64_t)),
    static_cast<size_t>(length),
  };
  return true;
}

bool decode_string_array_count(const generation_load_state & state,
                               const emel::gguf::loader::kv_entry & entry,
                               uint32_t & count_out) {
  const std::span<const uint8_t> bytes = kv_value_view(state, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  const uint32_t element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  if (element_type != constants::gguf_type_string) {
    return false;
  }

  const uint64_t count = read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
  if (count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  count_out = static_cast<uint32_t>(count);
  return true;
}

bool copy_tensor_names(const std::span<const uint8_t> file_image, emel::model::data & model_data) {
  model_data.name_bytes_used = 0u;

  for (uint32_t i = 0u; i < model_data.n_tensors; ++i) {
    auto & tensor = model_data.tensors[i];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length > model_data.name_storage.size()) {
      return false;
    }

    const uint32_t copied_offset = model_data.name_bytes_used;
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + copied_offset,
                  file_image.data() + name_offset,
                  name_length);
    }

    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
    tensor.name_offset = copied_offset;
  }

  return true;
}

std::string_view tensor_name_view(const emel::model::data & model_data,
                                  const emel::model::data::tensor_record & tensor) {
  if (static_cast<size_t>(tensor.name_offset) + static_cast<size_t>(tensor.name_length) >
      model_data.name_storage.size()) {
    return {};
  }

  return std::string_view{
    model_data.name_storage.data() + tensor.name_offset,
    tensor.name_length,
  };
}

bool try_parse_block_index(const std::string_view name, int32_t & block_index_out) {
  constexpr std::string_view k_prefix = "blk.";
  if (!name.starts_with(k_prefix)) {
    return false;
  }

  size_t cursor = k_prefix.size();
  if (cursor >= name.size()) {
    return false;
  }

  int32_t value = 0;
  bool saw_digit = false;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    saw_digit = true;
    value = value * 10 + static_cast<int32_t>(name[cursor] - '0');
    ++cursor;
  }

  if (!saw_digit || cursor >= name.size() || name[cursor] != '.') {
    return false;
  }

  block_index_out = value;
  return true;
}

emel::error::type populate_model_metadata(const generation_load_state & state,
                                          emel::model::data & model_data) {
  const auto * architecture_entry = find_kv_entry(state, "general.architecture");
  if (architecture_entry == nullptr) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  std::string_view architecture = {};
  if (!decode_string_value(state, *architecture_entry, architecture) ||
      architecture.size() >= model_data.architecture_name.size()) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  copy_name(model_data.architecture_name, architecture);

  const auto assign_i32 = [&](const std::string_view key, int32_t & field) {
    const auto * entry = find_kv_entry(state, key);
    if (entry == nullptr) {
      return true;
    }

    uint64_t value = 0u;
    if (!decode_integer_value(state, *entry, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  };

  if (!assign_i32("llama.context_length", model_data.params.n_ctx) ||
      !assign_i32("llama.embedding_length", model_data.params.n_embd) ||
      !assign_i32("llama.feed_forward_length", model_data.params.n_ff) ||
      !assign_i32("llama.attention.head_count", model_data.params.n_head) ||
      !assign_i32("llama.attention.head_count_kv", model_data.params.n_head_kv) ||
      !assign_i32("llama.rope.dimension_count", model_data.params.n_rot) ||
      !assign_i32("llama.block_count", model_data.params.n_layer) ||
      !assign_i32("llama.vocab_size", model_data.params.n_vocab)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const auto * tokens_entry = find_kv_entry(state, "tokenizer.tokens");
  if (tokens_entry != nullptr) {
    uint32_t token_count = 0u;
    if (!decode_string_array_count(state, *tokens_entry, token_count) ||
        token_count > static_cast<uint32_t>(emel::model::data::k_max_vocab_tokens)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    model_data.vocab_data.n_tokens = token_count;
    if (model_data.params.n_vocab == 0) {
      model_data.params.n_vocab = static_cast<int32_t>(token_count);
    }
  }

  return emel::error::cast(emel::model::loader::error::none);
}

std::string_view architecture_name_view(const emel::model::data & model_data) {
  size_t length = 0u;
  while (length < model_data.architecture_name.size() &&
         model_data.architecture_name[length] != '\0') {
    ++length;
  }

  return std::string_view{model_data.architecture_name.data(), length};
}

emel::error::type run_emel_parse_model(void * owner,
                                       const emel::model::loader::event::load & req) {
  auto & state = *static_cast<generation_load_state *>(owner);

  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  const std::span<const uint8_t> file_image{
    static_cast<const uint8_t *>(req.file_image),
    static_cast<size_t>(req.file_size),
  };

  reset_gguf_capture(state);
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&state, on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{&state, on_probe_error};
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe probe_ev{
    file_image,
    requirements,
    probe_done_cb,
    probe_error_cb,
  };

  if (!state.gguf_loader.process_event(probe_ev) ||
      !state.gguf.probe_done ||
      state.gguf.probe_error) {
    return map_gguf_error(state.gguf.err);
  }

  if (requirements.tensor_count > static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  state.kv_arena.resize(static_cast<size_t>(arena_bytes));
  state.kv_entries.resize(requirements.kv_count);

  reset_gguf_capture(state);
  const emel::gguf::loader::event::bind_done_fn bind_done_cb{&state, on_bind_done};
  const emel::gguf::loader::event::bind_error_fn bind_error_cb{&state, on_bind_error};
  const emel::gguf::loader::event::bind_storage bind_ev{
    std::span<uint8_t>{state.kv_arena},
    std::span<emel::gguf::loader::kv_entry>{state.kv_entries},
    std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                requirements.tensor_count},
    bind_done_cb,
    bind_error_cb,
  };

  if (!state.gguf_loader.process_event(bind_ev) ||
      !state.gguf.bind_done ||
      state.gguf.bind_error) {
    return map_gguf_error(state.gguf.err);
  }

  reset_gguf_capture(state);
  const emel::gguf::loader::event::parse_done_fn parse_done_cb{&state, on_parse_done};
  const emel::gguf::loader::event::parse_error_fn parse_error_cb{&state, on_parse_error};
  const emel::gguf::loader::event::parse parse_ev{
    file_image,
    parse_done_cb,
    parse_error_cb,
  };

  if (!state.gguf_loader.process_event(parse_ev) ||
      !state.gguf.parse_done ||
      state.gguf.parse_error) {
    return map_gguf_error(state.gguf.err);
  }

  req.model_data.n_tensors = requirements.tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  return populate_model_metadata(state, req.model_data);
}

emel::error::type run_emel_load_weights(void * owner,
                                        const emel::model::loader::event::load & req,
                                        uint64_t & bytes_total,
                                        uint64_t & bytes_done,
                                        bool & used_mmap) {
  auto & state = *static_cast<generation_load_state *>(owner);
  if (req.model_data.n_tensors == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  state.effect_requests.resize(req.model_data.n_tensors);
  state.effect_results.resize(req.model_data.n_tensors);

  reset_weight_capture(state);
  emel::model::weight_loader::event::bind_storage bind_ev{
    std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                req.model_data.n_tensors},
  };
  bind_ev.on_done = {&state, on_weight_bind_done};
  bind_ev.on_error = {&state, on_weight_bind_error};
  if (!state.weight_loader.process_event(bind_ev) ||
      !state.weight.bind_done ||
      state.weight.bind_error) {
    return map_weight_loader_error(state.weight.err);
  }

  reset_weight_capture(state);
  emel::model::weight_loader::event::plan_load plan_ev{
    std::span<emel::model::weight_loader::effect_request>{state.effect_requests},
  };
  plan_ev.on_done = {&state, on_weight_plan_done};
  plan_ev.on_error = {&state, on_weight_plan_error};
  if (!state.weight_loader.process_event(plan_ev) ||
      !state.weight.plan_done ||
      state.weight.plan_error) {
    return map_weight_loader_error(state.weight.err);
  }

  const uint32_t effect_count = state.weight.effect_count;
  for (uint32_t i = 0u; i < effect_count; ++i) {
    state.effect_results[i] = emel::model::weight_loader::effect_result{
      .kind = state.effect_requests[i].kind,
      .handle = state.effect_requests[i].target,
      .err = emel::error::cast(emel::model::weight_loader::error::none),
    };
  }

  reset_weight_capture(state);
  emel::model::weight_loader::event::apply_effect_results apply_ev{
    std::span<const emel::model::weight_loader::effect_result>{state.effect_results.data(),
                                                               effect_count},
  };
  apply_ev.on_done = {&state, on_weight_apply_done};
  apply_ev.on_error = {&state, on_weight_apply_error};
  if (!state.weight_loader.process_event(apply_ev) ||
      !state.weight.apply_done ||
      state.weight.apply_error) {
    return map_weight_loader_error(state.weight.err);
  }

  req.model_data.weights_data = req.file_image;
  req.model_data.weights_size = req.file_size;
  req.model_data.weights_mapped = false;
  req.model_data.weights_split_count = 1u;
  req.model_data.weights_split_offsets[0] = 0u;
  req.model_data.weights_split_sizes[0] = req.file_size;
  bytes_total = req.file_size;
  bytes_done = req.file_size;
  used_mmap = false;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type run_emel_map_layers(void *, const emel::model::loader::event::load & req) {
  int32_t max_block_index = -1;
  for (uint32_t i = 0u; i < req.model_data.n_tensors; ++i) {
    int32_t block_index = -1;
    if (try_parse_block_index(tensor_name_view(req.model_data, req.model_data.tensors[i]),
                              block_index) &&
        block_index > max_block_index) {
      max_block_index = block_index;
    }
  }

  if (max_block_index >= 0) {
    req.model_data.n_layers = max_block_index + 1;
    return emel::error::cast(emel::model::loader::error::none);
  }

  if (req.model_data.params.n_layer > 0) {
    req.model_data.n_layers = req.model_data.params.n_layer;
    return emel::error::cast(emel::model::loader::error::none);
  }

  return emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type run_emel_validate_structure(void *,
                                              const emel::model::loader::event::load & req) {
  if (req.model_data.n_tensors == 0u ||
      req.model_data.n_layers <= 0 ||
      req.model_data.weights_data == nullptr ||
      req.model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type run_emel_validate_architecture(
    void *,
    const emel::model::loader::event::load & req) {
  return architecture_name_view(req.model_data) == "llama"
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
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

int run_generation_harness_contract(const emel::paritychecker::parity_options & opts) {
  llama_backend_guard backend_guard{};
  llama_log_silencer log_silencer{};

  if (!file_exists(opts.model_path)) {
    std::fprintf(stderr, "generation load failed: missing model file %s\n", opts.model_path.c_str());
    return 1;
  }
  if (!is_expected_generation_fixture(opts.model_path)) {
    const std::filesystem::path expected_path = expected_generation_fixture_path();
    std::fprintf(stderr,
                 "generation requires canonical fixture %s, got %s\n",
                 expected_path.string().c_str(),
                 opts.model_path.c_str());
    return 1;
  }

  generation_load_state state{};
  if (!read_file_bytes(opts.model_path, state.file_bytes) || state.file_bytes.empty()) {
    std::fprintf(stderr, "generation load failed: unable to read model file %s\n", opts.model_path.c_str());
    return 1;
  }

  reset_load_capture(state);
  emel::model::loader::event::parse_model_fn parse_model{&state, run_emel_parse_model};
  emel::model::loader::event::load request{*state.model_data, parse_model};
  request.model_path = opts.model_path;
  request.file_image = state.file_bytes.data();
  request.file_size = state.file_bytes.size();
  request.load_weights = {&state, run_emel_load_weights};
  request.map_layers = {nullptr, run_emel_map_layers};
  request.validate_structure = {nullptr, run_emel_validate_structure};
  request.validate_architecture_impl = {nullptr, run_emel_validate_architecture};
  request.on_done = {&state, on_load_done};
  request.on_error = {&state, on_load_error};

  if (!state.model_loader.process_event(request) || !state.load.done || state.load.error) {
    const emel::error::type err = state.load.error
                                      ? state.load.err
                                      : emel::error::cast(emel::model::loader::error::internal_error);
    std::fprintf(stderr,
                 "generation load failed (fixture=%s err=%s)\n",
                 k_generation_fixture_name,
                 model_loader_error_name(err));
    return 1;
  }

  if (!load_generation_vocab_from_llama(opts.model_path, state)) {
    std::fprintf(stderr,
                 "generation vocab load failed (fixture=%s)\n",
                 k_generation_fixture_name);
    return 1;
  }

  const emel::error::type initialize_err = run_emel_initialize_generator(state, opts);
  if (initialize_err != emel::error::cast(emel::generator::error::none)) {
    std::fprintf(stderr,
                 "generation initialize failed (fixture=%s err=%s)\n",
                 k_generation_fixture_name,
                 generator_error_name(initialize_err));
    return 1;
  }

  generation_result emel_result{};
  const emel::error::type generation_err =
      run_emel_generate(state, opts, std::span<char>{emel_result.output}, emel_result.output_length);
  if (generation_err != emel::error::cast(emel::generator::error::none)) {
    std::fprintf(stderr,
                 "generation error (fixture=%s err=%s generated_tokens=%d output_bytes=%zu)\n",
                 k_generation_fixture_name,
                 generator_error_name(generation_err),
                 state.generation.tokens_generated,
                 state.generation.output_length);
    return 1;
  }
  emel_result.tokens_generated = state.generation.tokens_generated;
  emel_result.output_length = state.generation.output_length;

  generation_result reference_result{};
  const emel::error::type reference_err =
      run_reference_generate(state.backend, opts, reference_result);
  if (reference_err != emel::error::cast(emel::generator::error::none)) {
    std::fprintf(stderr,
                 "generation reference failed (fixture=%s err=%s)\n",
                 k_generation_fixture_name,
                 generator_error_name(reference_err));
    return 1;
  }

  if (!generation_results_match(emel_result, reference_result)) {
    const size_t mismatch_offset = first_mismatch_offset(emel_result, reference_result);
    std::fprintf(stderr,
                 "generation parity mismatch (fixture=%s emel_tokens=%d reference_tokens=%d "
                 "emel_bytes=%zu reference_bytes=%zu first_mismatch=%zu)\n",
                 k_generation_fixture_name,
                 emel_result.tokens_generated,
                 reference_result.tokens_generated,
                 emel_result.output_length,
                 reference_result.output_length,
                 mismatch_offset);
    if (opts.dump) {
      dump_generation_result("emel", emel_result);
      dump_generation_result("reference", reference_result);
    }
    return 1;
  }

  std::fprintf(stdout,
               "generation parity ok (fixture=%s prompt_bytes=%zu max_tokens=%d generated_tokens=%d "
               "output_bytes=%zu text=%.*s)\n",
               k_generation_fixture_name,
               opts.text.size(),
               opts.max_tokens,
               emel_result.tokens_generated,
               emel_result.output_length,
               static_cast<int>(emel_result.output_length),
               emel_result.output.data());
  if (opts.dump) {
    dump_generation_result("emel", emel_result);
    dump_generation_result("reference", reference_result);
  }
  return 0;
}

}  // namespace

namespace emel::paritychecker {

int run_parity(const parity_options & opts) {
  switch (opts.mode) {
    case parity_mode::generation:
      return run_generation_harness_contract(opts);
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
