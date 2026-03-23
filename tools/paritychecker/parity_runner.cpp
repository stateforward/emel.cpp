#include "parity_runner.hpp"
#include "tokenizer_parity.hpp"

#include <algorithm>
#include <array>
#include <cinttypes>
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
#include "emel/generator/detail.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/sm.hpp"
#include "emel/kernel/aarch64/actions.hpp"
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
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

#undef QK_K

extern "C" {
void ggml_vec_dot_f16(
    int n,
    float * s,
    size_t bs,
    ggml_fp16_t * x,
    size_t bx,
    ggml_fp16_t * y,
    size_t by,
    int nrc);
void ggml_vec_dot_q2_K_q8_K(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
struct reference_block_q8_k {
  float d = 0.0f;
  std::array<int8_t, ::emel::kernel::detail::quant::QK_K> qs = {};
  std::array<int16_t, ::emel::kernel::detail::quant::QK_K / 16> bsums = {};
};
void quantize_row_q8_K_ref(const float * x, reference_block_q8_k * y, int64_t k);
void dequantize_row_q2_K(const void * x, float * y, int64_t k);
void dequantize_row_q3_K(const void * x, float * y, int64_t k);
void dequantize_row_q6_K(const void * x, float * y, int64_t k);
}

namespace {

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_internal = 3;
constexpr const char * k_generation_fixture_name = "Llama-68M-Chat-v1-Q2_K.gguf";
constexpr size_t k_generation_output_capacity = 65536u;
constexpr size_t k_generation_token_trace_capacity = 4096u;
constexpr std::string_view k_reference_impl_source =
#ifdef PARITYCHECKER_REFERENCE_SOURCE
    PARITYCHECKER_REFERENCE_SOURCE;
#else
    "unknown";
#endif
constexpr std::string_view k_reference_impl_ref =
#ifdef PARITYCHECKER_REFERENCE_REF
    PARITYCHECKER_REFERENCE_REF;
#else
    "unknown";
#endif

using llama_model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
namespace kernel_quant = emel::kernel::detail::quant;

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

struct generation_trace {
  std::array<int32_t, k_generation_token_trace_capacity> token_ids = {};
  std::array<float, k_generation_token_trace_capacity> top_score_gaps = {};
  int32_t token_count = 0;
};

struct generation_result {
  std::array<char, k_generation_output_capacity> output = {};
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
  generation_trace trace = {};
};

struct initialize_backend {
  llama_model_ptr model = {nullptr, llama_model_free};
  const llama_vocab * vocab = nullptr;
  int32_t vocab_size = 0;
  int32_t fallback_token_id = 0;
  int32_t emel_reference_decode_calls = 0;
  int32_t emel_reference_logits_calls = 0;
  int32_t direct_reference_decode_calls = 0;
  int32_t direct_reference_logits_calls = 0;
};

struct argmax_summary {
  int32_t selected_token = -1;
  float best_score = -std::numeric_limits<float>::infinity();
  float second_best_score = -std::numeric_limits<float>::infinity();
};

argmax_summary summarize_argmax_scores(const int32_t & candidate_ids,
                                       const float & candidate_scores,
                                       const int32_t & candidate_count) {
  argmax_summary summary{};
  if (candidate_count <= 0) {
    return summary;
  }

  int32_t best_index = 0;
  float best_score = (&candidate_scores)[0];
  float second_best_score = -std::numeric_limits<float>::infinity();
  for (int32_t idx = 1; idx < candidate_count; ++idx) {
    const float score = (&candidate_scores)[idx];
    if (score > best_score) {
      second_best_score = best_score;
      best_score = score;
      best_index = idx;
    } else if (score > second_best_score) {
      second_best_score = score;
    }
  }

  summary.selected_token = (&candidate_ids)[best_index];
  summary.best_score = best_score;
  summary.second_best_score = second_best_score;
  return summary;
}

void append_trace_token(generation_trace & trace,
                        const int32_t token_id,
                        const float best_score,
                        const float second_best_score) {
  if (trace.token_count < 0 ||
      static_cast<size_t>(trace.token_count) >= trace.token_ids.size()) {
    return;
  }

  const size_t index = static_cast<size_t>(trace.token_count);
  trace.token_ids[index] = token_id;
  trace.top_score_gaps[index] = best_score - second_best_score;
  trace.token_count += 1;
}

struct generation_load_state;

emel::error::type sampler_select_argmax(generation_load_state & state,
                                        int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out);

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
  generation_trace * emel_trace = nullptr;
  std::array<emel::logits::sampler::fn, 1> samplers = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::vector<emel::model::weight_loader::effect_request> effect_requests = {};
  std::vector<emel::model::weight_loader::effect_result> effect_results = {};
  gguf_capture gguf = {};
  weight_capture weight = {};
  load_capture load = {};
  initialize_capture initialize = {};
  generation_capture generation = {};

  generation_load_state()
      : samplers{emel::logits::sampler::fn::from<generation_load_state, sampler_select_argmax>(
            this)} {}
};

emel::error::type sampler_select_argmax(generation_load_state & state,
                                        int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out) {
  const argmax_summary summary =
      summarize_argmax_scores(candidate_ids, candidate_scores, candidate_count);
  selected_token_out = summary.selected_token;
  if (state.emel_trace != nullptr) {
    append_trace_token(
        *state.emel_trace, summary.selected_token, summary.best_score, summary.second_best_score);
  }
  return emel::error::cast(emel::logits::sampler::error::none);
}

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

void reset_reference_decode_seam(initialize_backend & backend) {
  backend.emel_reference_decode_calls = 0;
  backend.emel_reference_logits_calls = 0;
  backend.direct_reference_decode_calls = 0;
  backend.direct_reference_logits_calls = 0;
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
  model_params.n_gpu_layers = 0;

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
    &state.tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
    std::span<emel::logits::sampler::fn>{state.samplers},
  };
  request.preprocessor_variant = generation_preprocessor_variant(*state.model_data);
  request.encoder_variant = generation_encoder_variant(*state.model_data);
  request.add_special = false;
  request.parse_special = false;
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
                                    size_t & output_length_out,
                                    generation_trace & trace_out) {
  if (state.generator == nullptr) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  reset_generation_capture(state);
  trace_out = {};
  state.emel_trace = &trace_out;
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
  state.emel_trace = nullptr;
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

argmax_summary select_argmax_from_logits(const float * logits, const int32_t vocab_size) {
  argmax_summary summary{};
  if (logits == nullptr || vocab_size <= 0) {
    return summary;
  }

  int32_t best_index = 0;
  float best_score = logits[0];
  float second_best_score = -std::numeric_limits<float>::infinity();
  for (int32_t idx = 1; idx < vocab_size; ++idx) {
    const float score = logits[idx];
    if (score > best_score) {
      second_best_score = best_score;
      best_score = score;
      best_index = idx;
    } else if (score > second_best_score) {
      second_best_score = score;
    }
  }

  summary.selected_token = best_index;
  summary.best_score = best_score;
  summary.second_best_score = second_best_score;
  return summary;
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

int32_t run_direct_reference_decode(initialize_backend & backend,
                                    llama_context * ctx,
                                    const llama_batch batch) {
  backend.direct_reference_decode_calls += 1;
  return llama_decode(ctx, batch);
}

float * read_direct_reference_logits(initialize_backend & backend, llama_context * ctx) {
  backend.direct_reference_logits_calls += 1;
  return llama_get_logits_ith(ctx, -1);
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
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
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
  if (run_direct_reference_decode(backend, ctx.get(), prompt_batch) != 0) {
    return emel::error::cast(emel::generator::error::backend);
  }

  for (int32_t step = 0; step < opts.max_tokens; ++step) {
    float * logits = read_direct_reference_logits(backend, ctx.get());
    if (logits == nullptr) {
      return emel::error::cast(emel::generator::error::backend);
    }

    const argmax_summary summary = select_argmax_from_logits(logits, backend.vocab_size);
    const llama_token selected = static_cast<llama_token>(summary.selected_token);
    append_trace_token(
        result_out.trace, summary.selected_token, summary.best_score, summary.second_best_score);
    result_out.tokens_generated += 1;
    if (!append_reference_piece(backend, selected, result_out)) {
      return emel::error::cast(emel::generator::error::backend);
    }
    if (llama_vocab_is_eog(backend.vocab, selected)) {
      break;
    }

    llama_token next_token = selected;
    llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
    if (run_direct_reference_decode(backend, ctx.get(), decode_batch) != 0) {
      return emel::error::cast(emel::generator::error::backend);
    }
  }

  return emel::error::cast(emel::generator::error::none);
}

int32_t first_token_mismatch_index(const generation_result & lhs, const generation_result & rhs) {
  const int32_t shared = std::min(lhs.trace.token_count, rhs.trace.token_count);
  for (int32_t idx = 0; idx < shared; ++idx) {
    if (lhs.trace.token_ids[static_cast<size_t>(idx)] !=
        rhs.trace.token_ids[static_cast<size_t>(idx)]) {
      return idx;
    }
  }
  return shared;
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

bool run_prefill_from_token_prefix(emel::generator::detail::native_backend & backend,
                                   std::span<const int32_t> prefix_tokens) {
  if (prefix_tokens.empty() ||
      prefix_tokens.size() > backend.bound_tokens.size() ||
      prefix_tokens.size() > backend.bound_positions.size()) {
    return false;
  }

  for (size_t idx = 0; idx < prefix_tokens.size(); ++idx) {
    backend.bound_tokens[idx] = prefix_tokens[idx];
    backend.bound_positions[idx] = static_cast<int32_t>(idx);
  }
  backend.bound_token_count = static_cast<int32_t>(prefix_tokens.size());
  backend.bound_position_count = static_cast<int32_t>(prefix_tokens.size());
  backend.bound_ready = true;
  return emel::generator::detail::run_prefill(backend);
}

bool matmul_vector_dequantized(const emel::generator::detail::tensor_matrix & matrix,
                               std::span<const float> input,
                               std::span<float> output) {
  if (matrix.tensor == nullptr ||
      matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  std::vector<float> row(static_cast<size_t>(matrix.cols));
  for (int32_t row_index = 0; row_index < matrix.rows; ++row_index) {
    if (!emel::generator::detail::copy_tensor_row(*matrix.tensor, row_index, row)) {
      return false;
    }
    double sum = 0.0;
    for (int32_t col = 0; col < matrix.cols; ++col) {
      sum += static_cast<double>(row[static_cast<size_t>(col)]) *
             static_cast<double>(input[static_cast<size_t>(col)]);
    }
    output[static_cast<size_t>(row_index)] = static_cast<float>(sum);
  }

  return true;
}

struct exact_matmul_mode {
  bool attention = false;
  bool ffn = false;
  bool output = false;
  uint8_t only_dtype = 0u;
  bool use_reference_q8 = false;
  bool use_scalar_quantized = false;
};

bool quantize_input_blocks_reference(std::span<const float> input,
                                     std::array<reference_block_q8_k,
                                                kernel_quant::MAX_Q8_K_BLOCKS> & blocks,
                                     uint64_t & block_count_out);

bool quantize_input_blocks(std::span<const float> input,
                           std::array<kernel_quant::block_q8_k,
                                      kernel_quant::MAX_Q8_K_BLOCKS> & blocks,
                           uint64_t & block_count_out);

float ggml_row_dot_reference_q8(const emel::generator::detail::tensor_matrix & matrix,
                                uint32_t row,
                                const reference_block_q8_k * q8_blocks,
                                uint64_t block_count);

bool matmul_vector_reference_q8(const emel::generator::detail::tensor_matrix & matrix,
                                std::span<const float> input,
                                std::span<float> output) {
  if (matrix.tensor == nullptr ||
      matrix.rows <= 0 ||
      matrix.cols <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  std::array<reference_block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
  uint64_t block_count = 0;
  if (!quantize_input_blocks_reference(input, q8_blocks, block_count)) {
    return false;
  }

  for (uint32_t row = 0; row < static_cast<uint32_t>(matrix.rows); ++row) {
    output[static_cast<size_t>(row)] =
        ggml_row_dot_reference_q8(matrix, row, q8_blocks.data(), block_count);
  }
  return true;
}

bool matmul_vector_scalar_quantized(const emel::generator::detail::tensor_matrix & matrix,
                                    std::span<const float> input,
                                    std::span<float> output) {
  if (matrix.tensor == nullptr ||
      matrix.rows <= 0 ||
      matrix.cols <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  const auto dtype = static_cast<emel::kernel::event::dtype>(matrix.tensor->type);
  if (dtype != emel::kernel::event::dtype::q2_k &&
      dtype != emel::kernel::event::dtype::q3_k &&
      dtype != emel::kernel::event::dtype::q6_k) {
    return false;
  }

  std::array<kernel_quant::block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
  uint64_t block_count = 0;
  if (!quantize_input_blocks(input, q8_blocks, block_count)) {
    return false;
  }

  const auto * matrix_data = static_cast<const uint8_t *>(matrix.tensor->data);
  const size_t row_bytes =
      emel::generator::detail::row_storage_bytes(*matrix.tensor, matrix.cols);
  for (uint32_t row = 0; row < static_cast<uint32_t>(matrix.rows); ++row) {
    const uint8_t * row_ptr = matrix_data + static_cast<size_t>(row) * row_bytes;
    switch (dtype) {
      case emel::kernel::event::dtype::q2_k:
        output[static_cast<size_t>(row)] = emel::kernel::detail::dot_q2_k_q8_k_row_scalar(
            reinterpret_cast<const kernel_quant::block_q2_k *>(row_ptr),
            q8_blocks.data(),
            block_count);
        break;
      case emel::kernel::event::dtype::q3_k:
        output[static_cast<size_t>(row)] = emel::kernel::detail::dot_q3_k_q8_k_row_scalar(
            reinterpret_cast<const kernel_quant::block_q3_k *>(row_ptr),
            q8_blocks.data(),
            block_count);
        break;
      case emel::kernel::event::dtype::q6_k:
        output[static_cast<size_t>(row)] = emel::kernel::detail::dot_q6_k_q8_k_row_scalar(
            reinterpret_cast<const kernel_quant::block_q6_k *>(row_ptr),
            q8_blocks.data(),
            block_count);
        break;
      default:
        return false;
    }
  }

  return true;
}

bool matmul_vector_mode(emel::generator::detail::native_backend & backend,
                        const emel::generator::detail::tensor_matrix & matrix,
                        std::span<const float> input,
                        std::span<float> output,
                        const bool exact,
                        const uint8_t only_dtype = 0u,
                        const bool use_reference_q8 = false,
                        const bool use_scalar_quantized = false) {
  const bool dtype_match =
      only_dtype == 0u ||
      (matrix.tensor != nullptr && static_cast<uint8_t>(matrix.tensor->type) == only_dtype);
  const bool exact_enabled = exact && dtype_match;
  if (exact_enabled) {
    return matmul_vector_dequantized(matrix, input, output);
  }
  if (use_reference_q8 && dtype_match) {
    return matmul_vector_reference_q8(matrix, input, output);
  }
  if (use_scalar_quantized && dtype_match) {
    return matmul_vector_scalar_quantized(matrix, input, output);
  }
  return emel::generator::detail::matmul_vector(backend, matrix, input, output);
}

bool compute_logits_with_matmul_mode(emel::generator::detail::native_backend & backend,
                                     const exact_matmul_mode mode) {
  return emel::generator::detail::rms_norm(
             backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm) &&
         matmul_vector_mode(backend,
                            backend.output,
                            backend.norm,
                            backend.bound_logits,
                            mode.output,
                            mode.only_dtype,
                            mode.use_reference_q8,
                            mode.use_scalar_quantized);
}

bool run_layer_with_matmul_mode_scalar_attention(emel::generator::detail::native_backend & backend,
                                                 const int32_t layer_index,
                                                 const int32_t position,
                                                 const exact_matmul_mode mode) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector_mode(
          backend,
          block.attention_q,
          backend.norm,
          backend.q,
          mode.attention,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.attention_k,
          backend.norm,
          backend.k,
          mode.attention,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.attention_v,
          backend.norm,
          backend.v,
          mode.attention,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized)) {
    return false;
  }

  emel::generator::detail::apply_rope(
      backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
  emel::generator::detail::apply_rope(backend.k,
                                      backend.n_head_kv,
                                      backend.head_dim_kv,
                                      backend.n_rot,
                                      position,
                                      backend.rope_freq_base);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
      backend.q_attn.data());

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
      backend.key_cache.data() + cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
      backend.value_cache.data() + cache_offset);

  if (!emel::generator::detail::compute_attention(
          backend, layer_index, position + 1, backend.q_attn) ||
      !matmul_vector_mode(
          backend,
          block.attention_output,
          backend.attn_ctx,
          backend.projected,
          mode.attention,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector_mode(
          backend,
          block.feed_forward_gate,
          backend.norm,
          backend.gate,
          mode.ffn,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.feed_forward_up,
          backend.norm,
          backend.up,
          mode.ffn,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized)) {
    return false;
  }

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
  }

  if (!matmul_vector_mode(
          backend,
          block.feed_forward_down,
          backend.ffn_hidden,
          backend.projected,
          mode.ffn,
          mode.only_dtype,
          mode.use_reference_q8,
          mode.use_scalar_quantized)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

bool run_prefill_with_scalar_attention_matmul_mode(
    emel::generator::detail::native_backend & backend,
    std::span<const int32_t> prefix_tokens,
    const exact_matmul_mode mode) {
  if (prefix_tokens.empty()) {
    return false;
  }

  backend.kv_cache_tokens = 0;
  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (token_id < 0 ||
        token_id >= backend.token_embedding.rows ||
        position < 0 ||
        position >= backend.n_ctx) {
      return false;
    }

    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      return false;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_matmul_mode_scalar_attention(backend, layer, position, mode)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return compute_logits_with_matmul_mode(backend, mode);
}

bool run_layer_with_scalar_attention(emel::generator::detail::native_backend & backend,
                                     const int32_t layer_index,
                                     const int32_t position) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
    return false;
  }

  emel::generator::detail::apply_rope(
      backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
  emel::generator::detail::apply_rope(backend.k,
                                      backend.n_head_kv,
                                      backend.head_dim_kv,
                                      backend.n_rot,
                                      position,
                                      backend.rope_freq_base);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
      backend.q_attn.data());

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
      backend.key_cache.data() + cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
      backend.value_cache.data() + cache_offset);

  if (!emel::generator::detail::compute_attention(
          backend, layer_index, position + 1, backend.q_attn) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
      !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
    return false;
  }

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
  }

  if (!emel::generator::detail::matmul_vector(
          backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

bool run_prefill_with_scalar_attention(emel::generator::detail::native_backend & backend,
                                       std::span<const int32_t> prefix_tokens) {
  if (prefix_tokens.empty()) {
    return false;
  }

  backend.kv_cache_tokens = 0;
  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (token_id < 0 ||
        token_id >= backend.token_embedding.rows ||
        position < 0 ||
        position >= backend.n_ctx) {
      return false;
    }

    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      return false;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool compute_attention_with_ggml_f16_value_contraction(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t head_count = backend.n_head;
  const int32_t kv_head_count = backend.n_head_kv;
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const int32_t kv_dim = kv_head_count * kv_head_dim;
  const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::fill(backend.attn_ctx.begin(), backend.attn_ctx.end(), 0.0f);

  std::vector<ggml_fp16_t> weight_f16(static_cast<size_t>(position_limit));
  std::vector<ggml_fp16_t> value_f16(static_cast<size_t>(position_limit));
  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      float score = 0.0f;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        score += q_vector[q_offset + static_cast<size_t>(dim)] *
                 backend.key_cache[cache_offset + static_cast<size_t>(dim)];
      }
      score *= inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    float score_sum = 0.0f;
    for (int32_t position = 0; position < position_limit; ++position) {
      const float prob = std::exp(backend.attn_scores[static_cast<size_t>(position)] - max_score);
      backend.attn_probs[static_cast<size_t>(position)] = prob;
      score_sum += prob;
    }

    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight = backend.attn_probs[static_cast<size_t>(position)] / score_sum;
      weight_f16[static_cast<size_t>(position)] = kernel_quant::fp32_to_fp16(weight);
    }

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      for (int32_t position = 0; position < position_limit; ++position) {
        const size_t cache_offset =
            emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
            kv_offset;
        value_f16[static_cast<size_t>(position)] = kernel_quant::fp32_to_fp16(
            backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }

      float dot = 0.0f;
      ggml_vec_dot_f16(position_limit,
                       &dot,
                       0u,
                       value_f16.data(),
                       0u,
                       weight_f16.data(),
                       0u,
                       1);
      backend.attn_ctx[q_offset + static_cast<size_t>(dim)] = dot;
    }
  }

  return true;
}

bool run_layer_with_scalar_attention_ggml_f16_value_contraction(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
    return false;
  }

  emel::generator::detail::apply_rope(
      backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
  emel::generator::detail::apply_rope(backend.k,
                                      backend.n_head_kv,
                                      backend.head_dim_kv,
                                      backend.n_rot,
                                      position,
                                      backend.rope_freq_base);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
      backend.q_attn.data());

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
      backend.key_cache.data() + cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
      backend.value_cache.data() + cache_offset);

  if (!compute_attention_with_ggml_f16_value_contraction(
          backend, layer_index, position + 1, backend.q_attn) ||
      !emel::generator::detail::matmul_vector(
          backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
      !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
    return false;
  }

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
  }

  if (!emel::generator::detail::matmul_vector(
          backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

bool run_prefill_with_scalar_attention_ggml_f16_value_contraction(
    emel::generator::detail::native_backend & backend,
    std::span<const int32_t> prefix_tokens) {
  if (prefix_tokens.empty()) {
    return false;
  }

  backend.kv_cache_tokens = 0;
  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (token_id < 0 ||
        token_id >= backend.token_embedding.rows ||
        position < 0 ||
        position >= backend.n_ctx) {
      return false;
    }

    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      return false;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention_ggml_f16_value_contraction(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_reference_prefix_decode(llama_context * ctx,
                                 std::span<const llama_token> prompt_tokens,
                                 std::span<const int32_t> generated_tokens);

void dump_candidate_logits_compare(const char * label,
                                   const emel::generator::detail::native_backend & backend,
                                   const int32_t token_a,
                                   const int32_t token_b);

void dump_scalar_attention_debug(const generation_load_state & state,
                                 const emel::paritychecker::parity_options & opts,
                                 const generation_result & emel_result,
                                 const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    return;
  }

  std::vector<int32_t> prefix_tokens;
  prefix_tokens.reserve(prompt_tokens.size() + static_cast<size_t>(token_mismatch_index));
  for (const llama_token token : prompt_tokens) {
    prefix_tokens.push_back(static_cast<int32_t>(token));
  }
  for (int32_t idx = 0; idx < token_mismatch_index; ++idx) {
    prefix_tokens.push_back(reference_result.trace.token_ids[static_cast<size_t>(idx)]);
  }

  emel::generator::detail::native_backend dispatch_backend = {};
  emel::generator::detail::native_backend scalar_backend = {};
  emel::generator::detail::native_backend shared_backend = {};
  emel::generator::detail::native_backend ggml_f16_attention_backend = {};
  emel::generator::detail::native_backend exact_backend = {};
  emel::generator::detail::native_backend attention_exact_backend = {};
  emel::generator::detail::native_backend ffn_exact_backend = {};
  emel::generator::detail::native_backend output_exact_backend = {};
  emel::generator::detail::native_backend q2_exact_backend = {};
  emel::generator::detail::native_backend q3_exact_backend = {};
  emel::generator::detail::native_backend q6_exact_backend = {};
  emel::generator::detail::native_backend q2_scalar_quant_backend = {};
  emel::generator::detail::native_backend q3_scalar_quant_backend = {};
  emel::generator::detail::native_backend q6_scalar_quant_backend = {};
  emel::generator::detail::native_backend q2_reference_backend = {};
  emel::generator::detail::native_backend q3_reference_backend = {};
  emel::generator::detail::native_backend q6_reference_backend = {};
  const exact_matmul_mode exact_all{.attention = true, .ffn = true, .output = true};
  const exact_matmul_mode exact_attention_only{.attention = true, .ffn = false, .output = false};
  const exact_matmul_mode exact_ffn_only{.attention = false, .ffn = true, .output = false};
  const exact_matmul_mode exact_output_only{.attention = false, .ffn = false, .output = true};
  const exact_matmul_mode exact_q2_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
  };
  const exact_matmul_mode exact_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
  };
  const exact_matmul_mode exact_q6_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q6_k),
  };
  const exact_matmul_mode scalar_quant_q2_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
      .use_scalar_quantized = true,
  };
  const exact_matmul_mode scalar_quant_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_scalar_quantized = true,
  };
  const exact_matmul_mode scalar_quant_q6_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q6_k),
      .use_scalar_quantized = true,
  };
  const exact_matmul_mode reference_q2_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
      .use_reference_q8 = true,
  };
  const exact_matmul_mode reference_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_reference_q8 = true,
  };
  const exact_matmul_mode reference_q6_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q6_k),
      .use_reference_q8 = true,
  };
  if (emel::generator::detail::prepare(dispatch_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(scalar_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(shared_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ggml_f16_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(attention_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ffn_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(output_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q6_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q6_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q6_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      !run_prefill_from_token_prefix(dispatch_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention(scalar_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_f16_value_contraction(
          ggml_f16_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_matmul_mode(exact_backend, prefix_tokens, exact_all) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          attention_exact_backend, prefix_tokens, exact_attention_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          ffn_exact_backend, prefix_tokens, exact_ffn_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          output_exact_backend, prefix_tokens, exact_output_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q2_exact_backend, prefix_tokens, exact_q2_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q3_exact_backend, prefix_tokens, exact_q3_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q6_exact_backend, prefix_tokens, exact_q6_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q2_scalar_quant_backend, prefix_tokens, scalar_quant_q2_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q3_scalar_quant_backend, prefix_tokens, scalar_quant_q3_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q6_scalar_quant_backend, prefix_tokens, scalar_quant_q6_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q2_reference_backend, prefix_tokens, reference_q2_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q3_reference_backend, prefix_tokens, reference_q3_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q6_reference_backend, prefix_tokens, reference_q6_only)) {
    std::fprintf(stdout, "generation_debug.flash: unable to replay mismatch prefix\n");
    return;
  }
  shared_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  shared_backend.kernel.set_kind(shared_backend.kernel_kind);
  if (!run_prefill_from_token_prefix(shared_backend, prefix_tokens)) {
    std::fprintf(stdout, "generation_debug.flash: unable to replay shared backend prefix\n");
    return;
  }

  const int32_t emel_token = emel_result.trace.token_ids[static_cast<size_t>(token_mismatch_index)];
  const int32_t reference_token =
      reference_result.trace.token_ids[static_cast<size_t>(token_mismatch_index)];
  if (state.backend.model != nullptr && state.backend.model->layers.size() >= 2) {
    for (int32_t layer = 0; layer < 2; ++layer) {
      const auto & ref_layer = state.backend.model->layers[static_cast<size_t>(layer)];
      std::fprintf(stdout,
                   "generation_debug.reference.layer%d.transforms: bo=%d wo_s=%d "
                   "ffn_gate_b=%d ffn_up_b=%d ffn_down_b=%d "
                   "ffn_gate_s=%d ffn_up_s=%d ffn_down_s=%d\n",
                   layer,
                   ref_layer.bo != nullptr ? 1 : 0,
                   ref_layer.wo_s != nullptr ? 1 : 0,
                   ref_layer.ffn_gate_b != nullptr ? 1 : 0,
                   ref_layer.ffn_up_b != nullptr ? 1 : 0,
                   ref_layer.ffn_down_b != nullptr ? 1 : 0,
                   ref_layer.ffn_gate_s != nullptr ? 1 : 0,
                   ref_layer.ffn_up_s != nullptr ? 1 : 0,
                   ref_layer.ffn_down_s != nullptr ? 1 : 0);
    }
  }
  const argmax_summary dispatch_summary =
      select_argmax_from_logits(dispatch_backend.bound_logits.data(), dispatch_backend.n_vocab);
  const argmax_summary scalar_summary =
      select_argmax_from_logits(scalar_backend.bound_logits.data(), scalar_backend.n_vocab);
  const argmax_summary shared_summary =
      select_argmax_from_logits(shared_backend.bound_logits.data(), shared_backend.n_vocab);
  const argmax_summary ggml_f16_attention_summary = select_argmax_from_logits(
      ggml_f16_attention_backend.bound_logits.data(), ggml_f16_attention_backend.n_vocab);
  const argmax_summary exact_summary =
      select_argmax_from_logits(exact_backend.bound_logits.data(), exact_backend.n_vocab);
  const argmax_summary attention_exact_summary = select_argmax_from_logits(
      attention_exact_backend.bound_logits.data(), attention_exact_backend.n_vocab);
  const argmax_summary ffn_exact_summary =
      select_argmax_from_logits(ffn_exact_backend.bound_logits.data(), ffn_exact_backend.n_vocab);
  const argmax_summary output_exact_summary = select_argmax_from_logits(
      output_exact_backend.bound_logits.data(), output_exact_backend.n_vocab);
  const argmax_summary q2_exact_summary =
      select_argmax_from_logits(q2_exact_backend.bound_logits.data(), q2_exact_backend.n_vocab);
  const argmax_summary q3_exact_summary =
      select_argmax_from_logits(q3_exact_backend.bound_logits.data(), q3_exact_backend.n_vocab);
  const argmax_summary q6_exact_summary =
      select_argmax_from_logits(q6_exact_backend.bound_logits.data(), q6_exact_backend.n_vocab);
  const argmax_summary q2_scalar_quant_summary = select_argmax_from_logits(
      q2_scalar_quant_backend.bound_logits.data(), q2_scalar_quant_backend.n_vocab);
  const argmax_summary q3_scalar_quant_summary = select_argmax_from_logits(
      q3_scalar_quant_backend.bound_logits.data(), q3_scalar_quant_backend.n_vocab);
  const argmax_summary q6_scalar_quant_summary = select_argmax_from_logits(
      q6_scalar_quant_backend.bound_logits.data(), q6_scalar_quant_backend.n_vocab);
  const argmax_summary q2_reference_summary = select_argmax_from_logits(
      q2_reference_backend.bound_logits.data(), q2_reference_backend.n_vocab);
  const argmax_summary q3_reference_summary = select_argmax_from_logits(
      q3_reference_backend.bound_logits.data(), q3_reference_backend.n_vocab);
  const argmax_summary q6_reference_summary = select_argmax_from_logits(
      q6_reference_backend.bound_logits.data(), q6_reference_backend.n_vocab);
  const auto dump_flash_kernel_compare =
      [&](const char * label, emel::generator::detail::native_backend & backend) {
        const int32_t last_layer = backend.n_layer - 1;
        const int32_t last_position = static_cast<int32_t>(prefix_tokens.size()) - 1;
        auto neon_request =
            emel::generator::detail::make_flash_attn_request(backend, last_layer, last_position);
        std::vector<float> neon_dst(backend.attn_ctx.size(), -1.0f);
        std::vector<float> shared_dst(backend.attn_ctx.size(), -1.0f);
        neon_request.dst = emel::generator::detail::make_dst_view_3d(
            neon_dst.data(), neon_request.dst.ne[0], neon_request.dst.ne[1], neon_request.dst.ne[2]);
        auto shared_request = neon_request;
        shared_request.dst = emel::generator::detail::make_dst_view_3d(
            shared_dst.data(),
            shared_request.dst.ne[0],
            shared_request.dst.ne[1],
            shared_request.dst.ne[2]);
        emel::kernel::detail::flash_attn_workspace neon_workspace{};
        emel::kernel::detail::flash_attn_workspace shared_workspace{};
        if (!emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
                neon_request, true, neon_workspace) ||
            !emel::kernel::detail::run_flash_attn_ext_with_workspace(
                shared_request, shared_workspace)) {
          std::fprintf(stdout, "generation_debug.flash.kernel.%s: unable to compare\n", label);
          return;
        }
        float max_abs = 0.0f;
        size_t max_idx = 0u;
        for (size_t idx = 0; idx < neon_dst.size(); ++idx) {
          const float diff = std::fabs(neon_dst[idx] - shared_dst[idx]);
          if (diff > max_abs) {
            max_abs = diff;
            max_idx = idx;
          }
        }
        std::fprintf(stdout,
                     "generation_debug.flash.kernel.%s: max_abs=%g idx=%zu neon=%g shared=%g\n",
                     label,
                     max_abs,
                     max_idx,
                     neon_dst[max_idx],
                     shared_dst[max_idx]);
      };
  dump_flash_kernel_compare("dispatch_state", dispatch_backend);
  dump_flash_kernel_compare("shared_state", shared_backend);
  std::fprintf(stdout,
               "generation_debug.flash: prefix_tokens=%zu dispatch_argmax=%d scalar_argmax=%d "
               "shared_argmax=%d ggml_f16_attn_argmax=%d exact_argmax=%d attention_exact_argmax=%d "
               "ffn_exact_argmax=%d output_exact_argmax=%d q2_exact_argmax=%d "
               "q3_exact_argmax=%d q6_exact_argmax=%d q2_scalar_argmax=%d q3_scalar_argmax=%d "
               "q6_scalar_argmax=%d q2_reference_argmax=%d q3_reference_argmax=%d "
               "q6_reference_argmax=%d reference_token=%d\n",
               prefix_tokens.size(),
               dispatch_summary.selected_token,
               scalar_summary.selected_token,
               shared_summary.selected_token,
               ggml_f16_attention_summary.selected_token,
               exact_summary.selected_token,
               attention_exact_summary.selected_token,
               ffn_exact_summary.selected_token,
               output_exact_summary.selected_token,
               q2_exact_summary.selected_token,
               q3_exact_summary.selected_token,
               q6_exact_summary.selected_token,
               q2_scalar_quant_summary.selected_token,
               q3_scalar_quant_summary.selected_token,
               q6_scalar_quant_summary.selected_token,
               q2_reference_summary.selected_token,
               q3_reference_summary.selected_token,
               q6_reference_summary.selected_token,
               reference_token);
  std::fprintf(stdout,
               "generation_debug.flash.scores: dispatch_emel=%g dispatch_reference=%g "
               "scalar_emel=%g scalar_reference=%g shared_emel=%g shared_reference=%g "
               "exact_emel=%g exact_reference=%g attention_exact_emel=%g "
               "attention_exact_reference=%g ffn_exact_emel=%g ffn_exact_reference=%g "
               "output_exact_emel=%g output_exact_reference=%g\n",
               dispatch_backend.bound_logits[static_cast<size_t>(emel_token)],
               dispatch_backend.bound_logits[static_cast<size_t>(reference_token)],
               scalar_backend.bound_logits[static_cast<size_t>(emel_token)],
               scalar_backend.bound_logits[static_cast<size_t>(reference_token)],
               shared_backend.bound_logits[static_cast<size_t>(emel_token)],
               shared_backend.bound_logits[static_cast<size_t>(reference_token)],
               exact_backend.bound_logits[static_cast<size_t>(emel_token)],
               exact_backend.bound_logits[static_cast<size_t>(reference_token)],
               attention_exact_backend.bound_logits[static_cast<size_t>(emel_token)],
               attention_exact_backend.bound_logits[static_cast<size_t>(reference_token)],
               ffn_exact_backend.bound_logits[static_cast<size_t>(emel_token)],
               ffn_exact_backend.bound_logits[static_cast<size_t>(reference_token)],
               output_exact_backend.bound_logits[static_cast<size_t>(emel_token)],
               output_exact_backend.bound_logits[static_cast<size_t>(reference_token)]);
  dump_candidate_logits_compare(
      "generation_debug.flash.dispatch_output", dispatch_backend, emel_token, reference_token);
  dump_candidate_logits_compare(
      "generation_debug.flash.output_exact", output_exact_backend, emel_token, reference_token);

  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx == nullptr) {
    std::fprintf(stdout, "generation_debug.reference: unable to create reference context\n");
    return;
  }

  const std::span<const int32_t> generated_prefix{
      prefix_tokens.data() + prompt_tokens.size(),
      prefix_tokens.size() - prompt_tokens.size(),
  };
  if (!run_reference_prefix_decode(reference_ctx.get(), prompt_tokens, generated_prefix)) {
    std::fprintf(stdout, "generation_debug.reference: unable to decode mismatch prefix\n");
    return;
  }

  const float * reference_logits = llama_get_logits_ith(reference_ctx.get(), -1);
  if (reference_logits == nullptr) {
    std::fprintf(stdout, "generation_debug.reference: missing logits at mismatch prefix\n");
    return;
  }

  const argmax_summary reference_summary =
      select_argmax_from_logits(reference_logits, dispatch_backend.n_vocab);
  float max_abs = 0.0f;
  int32_t max_idx = 0;
  for (int32_t idx = 0; idx < dispatch_backend.n_vocab; ++idx) {
    const float diff =
        std::fabs(dispatch_backend.bound_logits[static_cast<size_t>(idx)] - reference_logits[idx]);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
    }
  }
  std::fprintf(stdout,
               "generation_debug.reference: argmax=%d max_abs=%g idx=%d dispatch=%g "
               "reference=%g\n",
               reference_summary.selected_token,
               max_abs,
               max_idx,
               dispatch_backend.bound_logits[static_cast<size_t>(max_idx)],
               reference_logits[max_idx]);
  std::fprintf(stdout,
               "generation_debug.reference.scores: dispatch_emel=%g dispatch_reference=%g "
               "reference_emel=%g reference_reference=%g\n",
               dispatch_backend.bound_logits[static_cast<size_t>(emel_token)],
               dispatch_backend.bound_logits[static_cast<size_t>(reference_token)],
               reference_logits[emel_token],
               reference_logits[reference_token]);
}

std::string_view token_piece_view(const initialize_backend & backend, const int32_t token_id) {
  if (backend.vocab == nullptr || token_id < 0) {
    return {};
  }

  const char * piece = llama_vocab_get_text(backend.vocab, token_id);
  if (piece == nullptr) {
    return {};
  }
  return piece;
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

void dump_generation_trace_window(const char * label,
                                  const initialize_backend & backend,
                                  const generation_result & result,
                                  const int32_t center_index) {
  const int32_t start = std::max(0, center_index - 2);
  const int32_t stop = std::min(result.trace.token_count, center_index + 3);
  for (int32_t idx = start; idx < stop; ++idx) {
    const int32_t token_id = result.trace.token_ids[static_cast<size_t>(idx)];
    const std::string_view piece = token_piece_view(backend, token_id);
    std::fprintf(stdout,
                 "%s.trace[%d]: token=%d gap=%g piece=%.*s\n",
                 label,
                 idx,
                 token_id,
                 result.trace.top_score_gaps[static_cast<size_t>(idx)],
                 static_cast<int>(piece.size()),
                 piece.data() != nullptr ? piece.data() : "");
  }
}

bool quantize_input_blocks(std::span<const float> input,
                           std::array<kernel_quant::block_q8_k,
                                      kernel_quant::MAX_Q8_K_BLOCKS> & blocks,
                           uint64_t & block_count_out) {
  if ((input.size() % kernel_quant::QK_K) != 0u) {
    return false;
  }
  block_count_out = input.size() / kernel_quant::QK_K;
  if (block_count_out > blocks.size()) {
    return false;
  }
  for (uint64_t block = 0; block < block_count_out; ++block) {
    kernel_quant::quantize_row_q8_k_strided(
        input.data() + block * kernel_quant::QK_K, 1, &blocks[block], kernel_quant::QK_K);
  }
  return true;
}

bool quantize_input_blocks_reference(std::span<const float> input,
                                     std::array<reference_block_q8_k,
                                                kernel_quant::MAX_Q8_K_BLOCKS> & blocks,
                                     uint64_t & block_count_out) {
  static_assert(sizeof(reference_block_q8_k) == sizeof(kernel_quant::block_q8_k));
  if ((input.size() % kernel_quant::QK_K) != 0u) {
    return false;
  }
  block_count_out = input.size() / kernel_quant::QK_K;
  if (block_count_out > blocks.size()) {
    return false;
  }
  for (uint64_t block = 0; block < block_count_out; ++block) {
    quantize_row_q8_K_ref(
        input.data() + block * kernel_quant::QK_K, &blocks[block], kernel_quant::QK_K);
  }
  return true;
}

float ggml_row_dot(const emel::generator::detail::tensor_matrix & matrix,
                   const uint32_t row,
                   const kernel_quant::block_q8_k * q8_blocks,
                   const uint64_t block_count) {
  const auto * row_ptr =
      static_cast<const uint8_t *>(matrix.tensor->data) +
      static_cast<size_t>(row) *
          emel::generator::detail::row_storage_bytes(*matrix.tensor, matrix.cols);
  float out = 0.0f;
  switch (static_cast<emel::kernel::event::dtype>(matrix.tensor->type)) {
    case emel::kernel::event::dtype::q2_k:
      ggml_vec_dot_q2_K_q8_K(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    case emel::kernel::event::dtype::q3_k:
      ggml_vec_dot_q3_K_q8_K(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    case emel::kernel::event::dtype::q6_k:
      ggml_vec_dot_q6_K_q8_K(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}

float ggml_row_dot_reference_q8(const emel::generator::detail::tensor_matrix & matrix,
                                const uint32_t row,
                                const reference_block_q8_k * q8_blocks,
                                const uint64_t block_count) {
  const auto * row_ptr =
      static_cast<const uint8_t *>(matrix.tensor->data) +
      static_cast<size_t>(row) *
          emel::generator::detail::row_storage_bytes(*matrix.tensor, matrix.cols);
  float out = 0.0f;
  switch (static_cast<emel::kernel::event::dtype>(matrix.tensor->type)) {
    case emel::kernel::event::dtype::q2_k:
      ggml_vec_dot_q2_K_q8_K(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    case emel::kernel::event::dtype::q3_k:
      ggml_vec_dot_q3_K_q8_K(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    case emel::kernel::event::dtype::q6_k:
      ggml_vec_dot_q6_K_q8_K(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}

float dequantized_row_dot(const emel::generator::detail::tensor_matrix & matrix,
                          const uint32_t row,
                          std::span<const float> input) {
  if (matrix.tensor == nullptr ||
      row >= static_cast<uint32_t>(matrix.rows) ||
      static_cast<size_t>(matrix.cols) != input.size()) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  std::vector<float> weights(static_cast<size_t>(matrix.cols));
  if (!emel::generator::detail::copy_tensor_row(*matrix.tensor, static_cast<int32_t>(row), weights)) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  double sum = 0.0;
  for (int32_t idx = 0; idx < matrix.cols; ++idx) {
    sum += static_cast<double>(weights[static_cast<size_t>(idx)]) *
           static_cast<double>(input[static_cast<size_t>(idx)]);
  }
  return static_cast<float>(sum);
}

void dump_candidate_logits_compare(const char * label,
                                   const emel::generator::detail::native_backend & backend,
                                   const int32_t token_a,
                                   const int32_t token_b) {
  if (backend.output.tensor == nullptr ||
      backend.output.rows <= 0 ||
      backend.output.cols <= 0 ||
      static_cast<size_t>(backend.output.cols) != backend.norm.size() ||
      token_a < 0 ||
      token_b < 0 ||
      token_a >= backend.output.rows ||
      token_b >= backend.output.rows) {
    std::fprintf(stdout, "%s: unavailable\n", label);
    return;
  }

  std::array<kernel_quant::block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> emel_q8 = {};
  std::array<reference_block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> reference_q8 = {};
  uint64_t emel_q8_blocks = 0;
  uint64_t reference_q8_blocks = 0;
  if (!quantize_input_blocks(backend.norm, emel_q8, emel_q8_blocks) ||
      !quantize_input_blocks_reference(backend.norm, reference_q8, reference_q8_blocks) ||
      emel_q8_blocks != reference_q8_blocks) {
    std::fprintf(stdout, "%s: q8 quantize failed\n", label);
    return;
  }

  const auto dump_token = [&](const char * token_label, const int32_t token_id) {
    const float emel_logit = backend.bound_logits[static_cast<size_t>(token_id)];
    const float ggml_emel_q8 = ggml_row_dot(
        backend.output, static_cast<uint32_t>(token_id), emel_q8.data(), emel_q8_blocks);
    const float ggml_reference_q8 = ggml_row_dot_reference_q8(
        backend.output, static_cast<uint32_t>(token_id), reference_q8.data(), reference_q8_blocks);
    const float exact = dequantized_row_dot(
        backend.output, static_cast<uint32_t>(token_id), backend.norm);
    std::fprintf(stdout,
                 "%s.%s: token=%d dtype=%u emel=%g ggml_emel_q8=%g ggml_reference_q8=%g "
                 "exact=%g emel_delta=%g exact_delta=%g\n",
                 label,
                 token_label,
                 token_id,
                 static_cast<unsigned>(backend.output.tensor->type),
                 emel_logit,
                 ggml_emel_q8,
                 ggml_reference_q8,
                 exact,
                 emel_logit - ggml_emel_q8,
                 ggml_emel_q8 - exact);
  };

  dump_token("candidate_a", token_a);
  dump_token("candidate_b", token_b);
}

void dump_q8_quantize_compare(const char * label, std::span<const float> input) {
  std::array<kernel_quant::block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> emel_blocks = {};
  std::array<reference_block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> reference_blocks = {};
  uint64_t emel_block_count = 0;
  uint64_t reference_block_count = 0;
  if (!quantize_input_blocks(input, emel_blocks, emel_block_count) ||
      !quantize_input_blocks_reference(input, reference_blocks, reference_block_count) ||
      emel_block_count != reference_block_count) {
    std::fprintf(stdout, "%s: q8 quantize failed\n", label);
    return;
  }

  float max_d_abs = 0.0f;
  uint64_t max_d_block = 0;
  float emel_d_at_max = 0.0f;
  float reference_d_at_max = 0.0f;
  int max_q_diff = 0;
  uint64_t max_q_block = 0;
  int32_t max_q_idx = 0;
  int32_t emel_q_at_max = 0;
  int32_t reference_q_at_max = 0;
  int max_bsum_diff = 0;
  uint64_t max_bsum_block = 0;
  int32_t max_bsum_idx = 0;
  int32_t emel_bsum_at_max = 0;
  int32_t reference_bsum_at_max = 0;

  for (uint64_t block = 0; block < emel_block_count; ++block) {
    const float d_diff = std::fabs(emel_blocks[block].d - reference_blocks[block].d);
    if (d_diff > max_d_abs) {
      max_d_abs = d_diff;
      max_d_block = block;
      emel_d_at_max = emel_blocks[block].d;
      reference_d_at_max = reference_blocks[block].d;
    }
    for (int32_t idx = 0; idx < kernel_quant::QK_K; ++idx) {
      const int diff = std::abs(static_cast<int>(emel_blocks[block].qs[static_cast<size_t>(idx)]) -
                                static_cast<int>(reference_blocks[block].qs[static_cast<size_t>(idx)]));
      if (diff > max_q_diff) {
        max_q_diff = diff;
        max_q_block = block;
        max_q_idx = idx;
        emel_q_at_max = emel_blocks[block].qs[static_cast<size_t>(idx)];
        reference_q_at_max = reference_blocks[block].qs[static_cast<size_t>(idx)];
      }
    }
    for (int32_t idx = 0; idx < kernel_quant::QK_K / 16; ++idx) {
      const int diff =
          std::abs(static_cast<int>(emel_blocks[block].bsums[static_cast<size_t>(idx)]) -
                   static_cast<int>(reference_blocks[block].bsums[static_cast<size_t>(idx)]));
      if (diff > max_bsum_diff) {
        max_bsum_diff = diff;
        max_bsum_block = block;
        max_bsum_idx = idx;
        emel_bsum_at_max = emel_blocks[block].bsums[static_cast<size_t>(idx)];
        reference_bsum_at_max = reference_blocks[block].bsums[static_cast<size_t>(idx)];
      }
    }
  }

  std::fprintf(stdout,
               "%s: blocks=%" PRIu64 " max_d_abs=%g d_block=%" PRIu64 " emel_d=%g reference_d=%g "
               "max_q_diff=%d q_block=%" PRIu64 " q_idx=%d emel_q=%d reference_q=%d "
               "max_bsum_diff=%d bsum_block=%" PRIu64 " bsum_idx=%d emel_bsum=%d reference_bsum=%d\n",
               label,
               emel_block_count,
               max_d_abs,
               max_d_block,
               emel_d_at_max,
               reference_d_at_max,
               max_q_diff,
               max_q_block,
               max_q_idx,
               emel_q_at_max,
               reference_q_at_max,
               max_bsum_diff,
               max_bsum_block,
               max_bsum_idx,
               emel_bsum_at_max,
               reference_bsum_at_max);
}

void dump_matrix_compare(const char * label,
                         emel::generator::detail::native_backend & backend,
                         const emel::generator::detail::tensor_matrix & matrix,
                         std::span<const float> input) {
  std::vector<float> emel_out(static_cast<size_t>(matrix.rows));
  if (!emel::generator::detail::matmul_vector(backend, matrix, input, emel_out)) {
    std::fprintf(stdout, "%s: emel matmul failed\n", label);
    return;
  }

  std::array<kernel_quant::block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
  uint64_t block_count = 0;
  if (!quantize_input_blocks(input, q8_blocks, block_count)) {
    std::fprintf(stdout, "%s: q8 quantize failed\n", label);
    return;
  }

  float max_abs = 0.0f;
  uint32_t max_row = 0;
  float emel_at_max = 0.0f;
  float ggml_at_max = 0.0f;
  for (uint32_t row = 0; row < static_cast<uint32_t>(matrix.rows); ++row) {
    const float ref = ggml_row_dot(matrix, row, q8_blocks.data(), block_count);
    const float diff = std::fabs(emel_out[row] - ref);
    if (diff > max_abs) {
      max_abs = diff;
      max_row = row;
      emel_at_max = emel_out[row];
      ggml_at_max = ref;
    }
  }

  std::fprintf(stdout,
               "%s: dtype=%u rows=%d cols=%d max_abs=%g row=%u emel=%g ggml=%g\n",
               label,
               static_cast<unsigned>(matrix.tensor->type),
               matrix.rows,
               matrix.cols,
               max_abs,
               max_row,
               emel_at_max,
               ggml_at_max);
}

void dump_matrix_compare_reference_q8(const char * label,
                                      emel::generator::detail::native_backend & backend,
                                      const emel::generator::detail::tensor_matrix & matrix,
                                      std::span<const float> input) {
  std::vector<float> emel_out(static_cast<size_t>(matrix.rows));
  if (!emel::generator::detail::matmul_vector(backend, matrix, input, emel_out)) {
    std::fprintf(stdout, "%s: emel matmul failed\n", label);
    return;
  }

  std::array<reference_block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
  uint64_t block_count = 0;
  if (!quantize_input_blocks_reference(input, q8_blocks, block_count)) {
    std::fprintf(stdout, "%s: reference q8 quantize failed\n", label);
    return;
  }

  float max_abs = 0.0f;
  uint32_t max_row = 0;
  float emel_at_max = 0.0f;
  float ggml_at_max = 0.0f;
  for (uint32_t row = 0; row < static_cast<uint32_t>(matrix.rows); ++row) {
    const float ref = ggml_row_dot_reference_q8(matrix, row, q8_blocks.data(), block_count);
    const float diff = std::fabs(emel_out[row] - ref);
    if (diff > max_abs) {
      max_abs = diff;
      max_row = row;
      emel_at_max = emel_out[row];
      ggml_at_max = ref;
    }
  }

  std::fprintf(stdout,
               "%s: dtype=%u rows=%d cols=%d max_abs=%g row=%u emel=%g ggml=%g\n",
               label,
               static_cast<unsigned>(matrix.tensor->type),
               matrix.rows,
               matrix.cols,
               max_abs,
               max_row,
               emel_at_max,
               ggml_at_max);
}

void dump_row_compare(const char * label,
                      const emel::model::data::tensor_record & tensor,
                      const int32_t row) {
  const int32_t cols = static_cast<int32_t>(tensor.dims[0]);
  const size_t row_bytes = emel::generator::detail::row_storage_bytes(tensor, cols);
  const auto * row_ptr = static_cast<const uint8_t *>(tensor.data) + static_cast<size_t>(row) * row_bytes;
  std::vector<float> emel(cols);
  std::vector<float> ggml(cols);
  if (!emel::generator::detail::copy_tensor_row(tensor, row, emel)) {
    std::fprintf(stdout, "%s: emel row copy failed\n", label);
    return;
  }

  switch (static_cast<emel::kernel::event::dtype>(tensor.type)) {
    case emel::kernel::event::dtype::q2_k:
      dequantize_row_q2_K(row_ptr, ggml.data(), cols);
      break;
    case emel::kernel::event::dtype::q3_k:
      dequantize_row_q3_K(row_ptr, ggml.data(), cols);
      break;
    case emel::kernel::event::dtype::q6_k:
      dequantize_row_q6_K(row_ptr, ggml.data(), cols);
      break;
    case emel::kernel::event::dtype::f32:
      std::memcpy(ggml.data(), row_ptr, static_cast<size_t>(cols) * sizeof(float));
      break;
    default:
      std::fprintf(stdout, "%s: unsupported dtype=%u\n", label, static_cast<unsigned>(tensor.type));
      return;
  }

  float max_abs = 0.0f;
  int32_t max_idx = 0;
  for (int32_t idx = 0; idx < cols; ++idx) {
    const float diff = std::fabs(emel[static_cast<size_t>(idx)] - ggml[static_cast<size_t>(idx)]);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
    }
  }

  std::fprintf(stdout,
               "%s: dtype=%u cols=%d max_abs=%g idx=%d emel=%g ggml=%g\n",
               label,
               static_cast<unsigned>(tensor.type),
               cols,
               max_abs,
               max_idx,
               emel[static_cast<size_t>(max_idx)],
               ggml[static_cast<size_t>(max_idx)]);
}

void dump_vector_compare(const char * label,
                         const emel::model::data::tensor_record & tensor,
                         std::span<const float> emel_values) {
  std::vector<float> ggml(emel_values.size());
  if (!emel::generator::detail::copy_tensor_row(tensor, 0, ggml)) {
    std::fprintf(stdout, "%s: vector copy failed\n", label);
    return;
  }

  float max_abs = 0.0f;
  int32_t max_idx = 0;
  for (int32_t idx = 0; idx < static_cast<int32_t>(emel_values.size()); ++idx) {
    const float diff = std::fabs(emel_values[static_cast<size_t>(idx)] - ggml[static_cast<size_t>(idx)]);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
    }
  }

  std::fprintf(stdout,
               "%s: dtype=%u len=%zu max_abs=%g idx=%d emel=%g ggml=%g\n",
               label,
               static_cast<unsigned>(tensor.type),
               emel_values.size(),
               max_abs,
               max_idx,
               emel_values[static_cast<size_t>(max_idx)],
               ggml[static_cast<size_t>(max_idx)]);
}

struct reference_tensor_capture {
  const char * name = nullptr;
  std::vector<float> values = {};
  std::array<int64_t, 4> shape = {1, 1, 1, 1};
};

struct reference_graph_capture {
  std::array<reference_tensor_capture, 31> entries = {{
      {"attn_norm-0", {}},
      {"Qcur-0", {}},
      {"Kcur-0", {}},
      {"Vcur-0", {}},
      {"kq_soft_max-0", {}},
      {"kqv-0", {}},
      {"kqv_out-0", {}},
      {"attn_out-0", {}},
      {"ffn_inp-0", {}},
      {"ffn_norm-0", {}},
      {"ffn_gate-0", {}},
      {"ffn_up-0", {}},
      {"ffn_swiglu-0", {}},
      {"ffn_out-0", {}},
      {"l_out-0", {}},
      {"attn_norm-1", {}},
      {"Qcur-1", {}},
      {"Kcur-1", {}},
      {"Vcur-1", {}},
      {"kq_soft_max-1", {}},
      {"kqv-1", {}},
      {"kqv_out-1", {}},
      {"attn_out-1", {}},
      {"ffn_inp-1", {}},
      {"ffn_norm-1", {}},
      {"ffn_gate-1", {}},
      {"ffn_up-1", {}},
      {"ffn_swiglu-1", {}},
      {"ffn_out-1", {}},
      {"l_out-1", {}},
      {"result_norm", {}},
  }};
};

bool capture_reference_eval_tensor(ggml_tensor * tensor, const bool ask, void * user_data) {
  auto * capture = static_cast<reference_graph_capture *>(user_data);
  if (tensor == nullptr || capture == nullptr) {
    return true;
  }

  for (auto & entry : capture->entries) {
    if (std::strcmp(tensor->name, entry.name) != 0) {
      continue;
    }
    if (ask) {
      return true;
    }
    const int64_t ne0 = std::max<int64_t>(tensor->ne[0], 1);
    const int64_t ne1 = std::max<int64_t>(tensor->ne[1], 1);
    const int64_t ne2 = std::max<int64_t>(tensor->ne[2], 1);
    const int64_t ne3 = std::max<int64_t>(tensor->ne[3], 1);
    const int64_t count = ne0 * ne1 * ne2 * ne3;
    if (count <= 0 || tensor->data == nullptr || tensor->type != GGML_TYPE_F32) {
      return false;
    }
    entry.shape = {ne0, ne1, ne2, ne3};
    entry.values.resize(static_cast<size_t>(count));
    const auto * base = static_cast<const uint8_t *>(tensor->data);
    size_t out_index = 0;
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
      for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
          for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const size_t offset = static_cast<size_t>(i0) * static_cast<size_t>(tensor->nb[0]) +
                                  static_cast<size_t>(i1) * static_cast<size_t>(tensor->nb[1]) +
                                  static_cast<size_t>(i2) * static_cast<size_t>(tensor->nb[2]) +
                                  static_cast<size_t>(i3) * static_cast<size_t>(tensor->nb[3]);
            std::memcpy(&entry.values[out_index], base + offset, sizeof(float));
            ++out_index;
          }
        }
      }
    }
    return true;
  }

  return false;
}

std::span<const float> find_reference_tensor(const reference_graph_capture & capture,
                                             const char * name) {
  for (const auto & entry : capture.entries) {
    if (std::strcmp(entry.name, name) == 0) {
      return entry.values;
    }
  }
  return {};
}

const reference_tensor_capture * find_reference_capture(const reference_graph_capture & capture,
                                                        const char * name) {
  for (const auto & entry : capture.entries) {
    if (std::strcmp(entry.name, name) == 0) {
      return &entry;
    }
  }
  return nullptr;
}

std::span<const float> reference_last_token_row(std::span<const float> reference_values,
                                                size_t row_width);

bool capture_reference_graph_for_tokens(const generation_load_state & state,
                                        std::span<const llama_token> tokens,
                                        reference_graph_capture & graph_capture) {
  llama_context_params context_params = llama_context_default_params();
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  context_params.n_ctx = 0;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.n_threads = 1;
  context_params.n_threads_batch = 1;
  context_params.embeddings = true;
  context_params.cb_eval = capture_reference_eval_tensor;
  context_params.cb_eval_user_data = &graph_capture;
  llama_context_ptr ctx = llama_context_ptr{
      state.backend.model != nullptr
          ? llama_init_from_model(state.backend.model.get(), context_params)
          : nullptr,
      llama_free,
  };
  if (ctx == nullptr) {
    return false;
  }

  if (tokens.empty()) {
    return false;
  }

  llama_batch prompt_batch =
      llama_batch_get_one(const_cast<llama_token *>(tokens.data()),
                          static_cast<int32_t>(tokens.size()));
  return llama_decode(ctx.get(), prompt_batch) == 0;
}

bool run_reference_prefix_decode(llama_context * ctx,
                                 std::span<const llama_token> prompt_tokens,
                                 std::span<const int32_t> generated_tokens) {
  if (ctx == nullptr || prompt_tokens.empty()) {
    return false;
  }

  llama_batch prompt_batch =
      llama_batch_get_one(const_cast<llama_token *>(prompt_tokens.data()),
                          static_cast<int32_t>(prompt_tokens.size()));
  if (llama_decode(ctx, prompt_batch) != 0) {
    return false;
  }

  for (const int32_t token_id : generated_tokens) {
    llama_token token = static_cast<llama_token>(token_id);
    llama_batch decode_batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, decode_batch) != 0) {
      return false;
    }
  }

  return true;
}

bool capture_reference_graph_for_generation_prefix(const generation_load_state & state,
                                                   std::span<const llama_token> prompt_tokens,
                                                   std::span<const int32_t> generated_tokens,
                                                   reference_graph_capture & graph_capture) {
  llama_context_params context_params = llama_context_default_params();
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  context_params.n_ctx = 0;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.n_threads = 1;
  context_params.n_threads_batch = 1;
  context_params.embeddings = true;
  context_params.cb_eval = capture_reference_eval_tensor;
  context_params.cb_eval_user_data = &graph_capture;
  llama_context_ptr ctx = llama_context_ptr{
      state.backend.model != nullptr
          ? llama_init_from_model(state.backend.model.get(), context_params)
          : nullptr,
      llama_free,
  };
  if (ctx == nullptr) {
    return false;
  }

  return run_reference_prefix_decode(ctx.get(), prompt_tokens, generated_tokens);
}

bool capture_reference_graph(const generation_load_state & state,
                             const emel::paritychecker::parity_options & opts,
                             reference_graph_capture & graph_capture) {
  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    return false;
  }
  return capture_reference_graph_for_tokens(state, prompt_tokens, graph_capture);
}

float read_ggml_tensor_value_f32(const ggml_tensor & tensor,
                                 const int64_t i0,
                                 const int64_t i1 = 0,
                                 const int64_t i2 = 0,
                                 const int64_t i3 = 0) {
  const size_t offset = static_cast<size_t>(i0) * tensor.nb[0] +
                        static_cast<size_t>(i1) * tensor.nb[1] +
                        static_cast<size_t>(i2) * tensor.nb[2] +
                        static_cast<size_t>(i3) * tensor.nb[3];
  const auto * base = static_cast<const uint8_t *>(tensor.data);
  switch (tensor.type) {
    case GGML_TYPE_F32: {
      float value = 0.0f;
      std::memcpy(&value, base + offset, sizeof(value));
      return value;
    }
    case GGML_TYPE_F16: {
      ggml_fp16_t value = 0;
      std::memcpy(&value, base + offset, sizeof(value));
      return ggml_fp16_to_fp32(value);
    }
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}

bool capture_reference_value_cache_rows(llama_context * ctx,
                                        const int32_t layer_index,
                                        std::vector<float> & values_out) {
  values_out.clear();
  if (ctx == nullptr) {
    return false;
  }

  auto * memory = ctx->get_memory();
  auto * kv_cache = dynamic_cast<llama_kv_cache *>(memory);
  if (kv_cache == nullptr) {
    return false;
  }

  const llama_pos pos_max = llama_memory_seq_pos_max(memory, 0);
  if (pos_max < 0) {
    return true;
  }

  const uint32_t n_kv = static_cast<uint32_t>(pos_max + 1);
  llama_kv_cache::slot_info slot{};
  slot.s0 = 0;
  slot.s1 = 0;
  slot.resize(1);
  slot.strm[0] = 0;
  slot.idxs[0].resize(n_kv);
  for (uint32_t idx = 0; idx < n_kv; ++idx) {
    slot.idxs[0][idx] = idx;
  }

  ggml_init_params params = {
      /*.mem_size   =*/ size_t(4 * ggml_tensor_overhead()),
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
  };
  ggml_context * ggctx = ggml_init(params);
  if (ggctx == nullptr) {
    return false;
  }

  ggml_tensor * value_view = kv_cache->get_v(ggctx, layer_index, n_kv, slot);
  if (value_view == nullptr ||
      value_view->ne[0] != static_cast<int64_t>(n_kv) ||
      value_view->type != GGML_TYPE_F16 && value_view->type != GGML_TYPE_F32) {
    ggml_free(ggctx);
    return false;
  }

  const int64_t n_head_kv = std::max<int64_t>(value_view->ne[1], 1);
  const int64_t head_dim = std::max<int64_t>(value_view->ne[2], 1);
  values_out.resize(static_cast<size_t>(n_kv * n_head_kv * head_dim));
  for (uint32_t position = 0; position < n_kv; ++position) {
    for (int64_t head = 0; head < n_head_kv; ++head) {
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        const size_t out_index =
            (static_cast<size_t>(position) * static_cast<size_t>(n_head_kv) +
             static_cast<size_t>(head)) *
                static_cast<size_t>(head_dim) +
            static_cast<size_t>(dim);
        values_out[out_index] =
            read_ggml_tensor_value_f32(*value_view, position, head, dim, 0);
      }
    }
  }

  ggml_free(ggctx);
  return true;
}

bool capture_reference_key_cache_rows(llama_context * ctx,
                                      const int32_t layer_index,
                                      std::vector<float> & values_out) {
  values_out.clear();
  if (ctx == nullptr) {
    return false;
  }

  auto * memory = ctx->get_memory();
  auto * kv_cache = dynamic_cast<llama_kv_cache *>(memory);
  if (kv_cache == nullptr) {
    return false;
  }

  const llama_pos pos_max = llama_memory_seq_pos_max(memory, 0);
  if (pos_max < 0) {
    return true;
  }

  const uint32_t n_kv = static_cast<uint32_t>(pos_max + 1);
  llama_kv_cache::slot_info slot{};
  slot.s0 = 0;
  slot.s1 = 0;
  slot.resize(1);
  slot.strm[0] = 0;
  slot.idxs[0].resize(n_kv);
  for (uint32_t idx = 0; idx < n_kv; ++idx) {
    slot.idxs[0][idx] = idx;
  }

  ggml_init_params params = {
      /*.mem_size   =*/ size_t(4 * ggml_tensor_overhead()),
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
  };
  ggml_context * ggctx = ggml_init(params);
  if (ggctx == nullptr) {
    return false;
  }

  ggml_tensor * key_view = kv_cache->get_k(ggctx, layer_index, n_kv, slot);
  if (key_view == nullptr ||
      key_view->type != GGML_TYPE_F16 && key_view->type != GGML_TYPE_F32) {
    ggml_free(ggctx);
    return false;
  }

  const int64_t head_dim = std::max<int64_t>(key_view->ne[0], 1);
  const int64_t n_head_kv = std::max<int64_t>(key_view->ne[1], 1);
  if (key_view->ne[2] != static_cast<int64_t>(n_kv)) {
    ggml_free(ggctx);
    return false;
  }

  values_out.resize(static_cast<size_t>(n_kv * n_head_kv * head_dim));
  for (uint32_t position = 0; position < n_kv; ++position) {
    for (int64_t head = 0; head < n_head_kv; ++head) {
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        const size_t out_index =
            (static_cast<size_t>(position) * static_cast<size_t>(n_head_kv) +
             static_cast<size_t>(head)) *
                static_cast<size_t>(head_dim) +
            static_cast<size_t>(dim);
        values_out[out_index] =
            read_ggml_tensor_value_f32(*key_view, dim, head, position, 0);
      }
    }
  }

  ggml_free(ggctx);
  return true;
}

void dump_state_compare(const char * label,
                        std::span<const float> emel_values,
                        std::span<const float> reference_values) {
  const std::span<const float> aligned_reference =
      reference_last_token_row(reference_values, emel_values.size());
  if (aligned_reference.size() == emel_values.size()) {
    reference_values = aligned_reference;
  }

  if (reference_values.empty() || emel_values.size() != reference_values.size()) {
    std::fprintf(stdout,
                 "%s: unavailable emel=%zu reference=%zu\n",
                 label,
                 emel_values.size(),
                 reference_values.size());
    return;
  }

  float max_abs = 0.0f;
  int32_t max_idx = 0;
  for (int32_t idx = 0; idx < static_cast<int32_t>(emel_values.size()); ++idx) {
    const float diff =
        std::fabs(emel_values[static_cast<size_t>(idx)] - reference_values[static_cast<size_t>(idx)]);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
    }
  }

  std::fprintf(stdout,
               "%s: max_abs=%g idx=%d emel=%g reference=%g\n",
               label,
               max_abs,
               max_idx,
               emel_values[static_cast<size_t>(max_idx)],
               reference_values[static_cast<size_t>(max_idx)]);
}

std::span<const float> reference_softmax_query_head_slice(const reference_tensor_capture & capture,
                                                          const int32_t head,
                                                          const int32_t query_index) {
  const int64_t ne0 = std::max<int64_t>(capture.shape[0], 1);
  const int64_t ne1 = std::max<int64_t>(capture.shape[1], 1);
  const int64_t ne2 = std::max<int64_t>(capture.shape[2], 1);
  const int64_t ne3 = std::max<int64_t>(capture.shape[3], 1);
  if (head < 0 || query_index < 0 || ne3 < 1 || head >= ne2 || query_index >= ne1) {
    return {};
  }

  const size_t offset =
      ((static_cast<size_t>(head) * static_cast<size_t>(ne1)) + static_cast<size_t>(query_index)) *
      static_cast<size_t>(ne0);
  if (offset + static_cast<size_t>(ne0) > capture.values.size()) {
    return {};
  }
  return std::span<const float>(capture.values).subspan(offset, static_cast<size_t>(ne0));
}

std::span<const float> reference_token_head_slice(const reference_tensor_capture & capture,
                                                  const int32_t token_index,
                                                  const int32_t head_index) {
  const int64_t ne0 = std::max<int64_t>(capture.shape[0], 1);
  const int64_t ne1 = std::max<int64_t>(capture.shape[1], 1);
  const int64_t ne2 = std::max<int64_t>(capture.shape[2], 1);
  if (token_index < 0 || head_index < 0 || token_index >= ne2 || head_index >= ne1) {
    return {};
  }

  const size_t offset =
      ((static_cast<size_t>(token_index) * static_cast<size_t>(ne1)) +
       static_cast<size_t>(head_index)) *
      static_cast<size_t>(ne0);
  if (offset + static_cast<size_t>(ne0) > capture.values.size()) {
    return {};
  }
  return std::span<const float>(capture.values).subspan(offset, static_cast<size_t>(ne0));
}

std::span<const float> reference_token_tensor_slice(const reference_tensor_capture & capture,
                                                    const int32_t token_index);

std::vector<float> reference_attention_context_slice(const reference_tensor_capture & capture,
                                                     const int32_t query_index,
                                                     const int32_t head_count,
                                                     const int32_t head_dim) {
  const int64_t ne0 = std::max<int64_t>(capture.shape[0], 1);
  const int64_t ne1 = std::max<int64_t>(capture.shape[1], 1);
  const int64_t ne2 = std::max<int64_t>(capture.shape[2], 1);
  if (query_index < 0 ||
      head_count < 0 ||
      head_dim < 0 ||
      ne0 != head_dim ||
      ne1 <= query_index ||
      ne2 != head_count) {
    return {};
  }

  std::vector<float> out(static_cast<size_t>(head_count) * static_cast<size_t>(head_dim), 0.0f);
  for (int32_t head = 0; head < head_count; ++head) {
    const size_t capture_offset =
        ((static_cast<size_t>(head) * static_cast<size_t>(ne1)) + static_cast<size_t>(query_index)) *
        static_cast<size_t>(ne0);
    const size_t out_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    if (capture_offset + static_cast<size_t>(head_dim) > capture.values.size()) {
      return {};
    }
    std::copy_n(capture.values.data() + static_cast<std::ptrdiff_t>(capture_offset),
                static_cast<size_t>(head_dim),
                out.data() + static_cast<std::ptrdiff_t>(out_offset));
  }
  return out;
}

bool compute_attention_with_softmax_debug(emel::generator::detail::native_backend & backend,
                                          const reference_graph_capture & graph_capture,
                                          const int32_t layer_index,
                                          const int32_t position_limit,
                                          const std::string & layer_prefix,
                                          std::span<const float> reference_value_cache_rows = {}) {
  const int32_t head_count = backend.n_head;
  const int32_t kv_head_count = backend.n_head_kv;
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const int32_t kv_dim = kv_head_count * kv_head_dim;
  const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::fill(backend.attn_ctx.begin(), backend.attn_ctx.end(), 0.0f);

  const std::string softmax_key = "kq_soft_max-" + std::to_string(layer_index);
  const reference_tensor_capture * softmax_capture =
      find_reference_capture(graph_capture, softmax_key.c_str());
  const std::string value_key = "Vcur-" + std::to_string(layer_index);
  const reference_tensor_capture * value_capture =
      find_reference_capture(graph_capture, value_key.c_str());
  const std::string raw_ctx_key = "kqv-" + std::to_string(layer_index);
  const reference_tensor_capture * raw_ctx_capture =
      find_reference_capture(graph_capture, raw_ctx_key.c_str());
  const std::string ctx_key = "kqv_out-" + std::to_string(layer_index);
  const reference_tensor_capture * ctx_capture =
      find_reference_capture(graph_capture, ctx_key.c_str());
  const std::span<const float> reference_ctx = reference_last_token_row(
      find_reference_tensor(graph_capture, ctx_key.c_str()), backend.attn_ctx.size());

  std::vector<float> ctx_reference_probs(backend.attn_ctx.size(), 0.0f);
  std::vector<float> ctx_reference_values(backend.attn_ctx.size(), 0.0f);
  std::vector<float> ctx_emel_ggml_f16_dot(backend.attn_ctx.size(), 0.0f);
  std::vector<float> ctx_reference_all_ggml_f16_dot(backend.attn_ctx.size(), 0.0f);
  const std::vector<float> raw_reference_ctx =
      raw_ctx_capture != nullptr
          ? reference_attention_context_slice(
                *raw_ctx_capture, 0, head_count, head_dim)
          : std::vector<float>{};

  float max_abs = 0.0f;
  int32_t max_head = 0;
  int32_t max_pos = 0;
  float emel_at_max = 0.0f;
  float reference_at_max = 0.0f;
  float value_cache_max_abs = 0.0f;
  int32_t value_cache_head = 0;
  int32_t value_cache_pos = 0;
  int32_t value_cache_dim = 0;
  float value_cache_emel = 0.0f;
  float value_cache_reference = 0.0f;
  float value_cache_row_max_abs = 0.0f;
  int32_t value_cache_row_pos = 0;
  int32_t value_cache_row_idx = 0;
  float value_cache_row_emel = 0.0f;
  float value_cache_row_reference = 0.0f;
  const bool have_reference_value_cache =
      reference_value_cache_rows.size() == static_cast<size_t>(position_limit * kv_dim);

  if (have_reference_value_cache || value_capture != nullptr) {
    for (int32_t position = 0; position < position_limit; ++position) {
      const std::span<const float> reference_row =
          have_reference_value_cache
              ? reference_value_cache_rows.subspan(
                    static_cast<size_t>(position) * static_cast<size_t>(kv_dim),
                    static_cast<size_t>(kv_dim))
              : reference_token_tensor_slice(*value_capture, position);
      if (reference_row.size() != static_cast<size_t>(kv_dim)) {
        continue;
      }
      const size_t cache_offset = emel::generator::detail::layer_cache_offset(
          backend, layer_index, position, kv_dim);
      for (int32_t idx = 0; idx < kv_dim; ++idx) {
        const float cached_value = backend.value_cache[cache_offset + static_cast<size_t>(idx)];
        const float diff = std::fabs(cached_value - reference_row[static_cast<size_t>(idx)]);
        if (diff > value_cache_row_max_abs) {
          value_cache_row_max_abs = diff;
          value_cache_row_pos = position;
          value_cache_row_idx = idx;
          value_cache_row_emel = cached_value;
          value_cache_row_reference = reference_row[static_cast<size_t>(idx)];
        }
      }
    }
  }

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);
    std::vector<ggml_fp16_t> emel_weight_f16(static_cast<size_t>(position_limit));
    std::vector<ggml_fp16_t> reference_weight_f16(static_cast<size_t>(position_limit));

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      float score = 0.0f;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        score += backend.q_attn[q_offset + static_cast<size_t>(dim)] *
                 backend.key_cache[cache_offset + static_cast<size_t>(dim)];
      }
      score *= inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    float score_sum = 0.0f;
    for (int32_t position = 0; position < position_limit; ++position) {
      const float prob = std::exp(backend.attn_scores[static_cast<size_t>(position)] - max_score);
      backend.attn_probs[static_cast<size_t>(position)] = prob;
      score_sum += prob;
    }

    const std::span<const float> reference_probs =
        softmax_capture != nullptr
            ? reference_softmax_query_head_slice(*softmax_capture, head, position_limit - 1)
            : std::span<const float>{};

    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight = backend.attn_probs[static_cast<size_t>(position)] / score_sum;
      backend.attn_probs[static_cast<size_t>(position)] = weight;
      const float reference_weight =
          static_cast<size_t>(position) < reference_probs.size()
              ? reference_probs[static_cast<size_t>(position)]
              : weight;
      emel_weight_f16[static_cast<size_t>(position)] = kernel_quant::fp32_to_fp16(weight);
      reference_weight_f16[static_cast<size_t>(position)] =
          kernel_quant::fp32_to_fp16(reference_weight);

      if (static_cast<size_t>(position) < reference_probs.size()) {
        const float diff = std::fabs(weight - reference_probs[static_cast<size_t>(position)]);
        if (diff > max_abs) {
          max_abs = diff;
          max_head = head;
          max_pos = position;
          emel_at_max = weight;
          reference_at_max = reference_probs[static_cast<size_t>(position)];
        }
      }

      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      const std::span<const float> reference_value =
          have_reference_value_cache
              ? reference_value_cache_rows.subspan(
                    static_cast<size_t>(position) * static_cast<size_t>(kv_dim) + kv_offset,
                    static_cast<size_t>(head_dim))
              : value_capture != nullptr
                    ? reference_token_head_slice(*value_capture, position, kv_head)
                    : std::span<const float>{};
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const float cached_value = backend.value_cache[cache_offset + static_cast<size_t>(dim)];
        if (static_cast<size_t>(dim) < reference_value.size()) {
          const float diff = std::fabs(cached_value - reference_value[static_cast<size_t>(dim)]);
          if (diff > value_cache_max_abs) {
            value_cache_max_abs = diff;
            value_cache_head = head;
            value_cache_pos = position;
            value_cache_dim = dim;
            value_cache_emel = cached_value;
            value_cache_reference = reference_value[static_cast<size_t>(dim)];
          }
        }
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * cached_value;
        ctx_reference_probs[q_offset + static_cast<size_t>(dim)] +=
            reference_weight * cached_value;
        if (static_cast<size_t>(dim) < reference_value.size()) {
          ctx_reference_values[q_offset + static_cast<size_t>(dim)] +=
              weight * reference_value[static_cast<size_t>(dim)];
        }
      }
    }

    std::vector<ggml_fp16_t> emel_value_f16(static_cast<size_t>(position_limit));
    std::vector<ggml_fp16_t> reference_value_f16(static_cast<size_t>(position_limit));
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      bool have_reference_value_dim = true;
      for (int32_t position = 0; position < position_limit; ++position) {
        const size_t cache_offset =
            emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
            kv_offset;
        emel_value_f16[static_cast<size_t>(position)] = kernel_quant::fp32_to_fp16(
            backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
        const std::span<const float> reference_value =
            have_reference_value_cache
                ? reference_value_cache_rows.subspan(
                      static_cast<size_t>(position) * static_cast<size_t>(kv_dim) + kv_offset,
                      static_cast<size_t>(head_dim))
                : value_capture != nullptr
                      ? reference_token_head_slice(*value_capture, position, kv_head)
                      : std::span<const float>{};
        if (static_cast<size_t>(dim) < reference_value.size()) {
          reference_value_f16[static_cast<size_t>(position)] =
              kernel_quant::fp32_to_fp16(reference_value[static_cast<size_t>(dim)]);
        } else {
          have_reference_value_dim = false;
        }
      }
      float dot = 0.0f;
      ggml_vec_dot_f16(position_limit,
                       &dot,
                       0u,
                       emel_value_f16.data(),
                       0u,
                       emel_weight_f16.data(),
                       0u,
                       1);
      ctx_emel_ggml_f16_dot[q_offset + static_cast<size_t>(dim)] = dot;
      if (have_reference_value_dim) {
        float reference_dot = 0.0f;
        ggml_vec_dot_f16(position_limit,
                         &reference_dot,
                         0u,
                         reference_value_f16.data(),
                         0u,
                         reference_weight_f16.data(),
                         0u,
                         1);
        ctx_reference_all_ggml_f16_dot[q_offset + static_cast<size_t>(dim)] = reference_dot;
      }
    }
  }

  if (!have_reference_value_cache && value_capture == nullptr) {
    std::fprintf(stdout, "%s.value_cache: unavailable\n", layer_prefix.c_str());
  } else {
    std::fprintf(stdout,
                 "%s.value_cache: max_abs=%g head=%d pos=%d dim=%d emel=%g reference=%g\n",
                 layer_prefix.c_str(),
                 value_cache_max_abs,
                 value_cache_head,
                 value_cache_pos,
                 value_cache_dim,
                 value_cache_emel,
                 value_cache_reference);
    std::fprintf(stdout,
                 "%s.value_cache_row: max_abs=%g pos=%d idx=%d emel=%g reference=%g\n",
                 layer_prefix.c_str(),
                 value_cache_row_max_abs,
                 value_cache_row_pos,
                 value_cache_row_idx,
                 value_cache_row_emel,
                 value_cache_row_reference);
  }

  if (softmax_capture == nullptr) {
    std::fprintf(stdout, "%s.kq_soft_max: unavailable\n", layer_prefix.c_str());
  } else {
    std::fprintf(stdout,
                 "%s.kq_soft_max: max_abs=%g head=%d pos=%d emel=%g reference=%g\n",
                 layer_prefix.c_str(),
                 max_abs,
                 max_head,
                 max_pos,
                 emel_at_max,
                 reference_at_max);
    std::fprintf(stdout,
                 "%s.kq_soft_max_shape: ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 layer_prefix.c_str(),
                 softmax_capture->shape[0],
                 softmax_capture->shape[1],
                 softmax_capture->shape[2],
                 softmax_capture->shape[3]);
    if (softmax_capture->shape[0] > position_limit) {
      float tail_max = 0.0f;
      int32_t tail_head = 0;
      int32_t tail_pos = position_limit;
      for (int32_t head = 0; head < head_count; ++head) {
        const std::span<const float> reference_probs =
            reference_softmax_query_head_slice(*softmax_capture, head, position_limit - 1);
        for (int32_t pos = position_limit; pos < static_cast<int32_t>(reference_probs.size()); ++pos) {
          const float value = std::fabs(reference_probs[static_cast<size_t>(pos)]);
          if (value > tail_max) {
            tail_max = value;
            tail_head = head;
            tail_pos = pos;
          }
        }
      }
      std::fprintf(stdout,
                   "%s.kq_soft_max_tail: max_abs=%g head=%d pos=%d\n",
                   layer_prefix.c_str(),
                   tail_max,
                   tail_head,
                   tail_pos);
    }
  }

  if (ctx_capture != nullptr) {
    std::fprintf(stdout,
                 "%s.kqv_out_shape: ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 layer_prefix.c_str(),
                 ctx_capture->shape[0],
                 ctx_capture->shape[1],
                 ctx_capture->shape[2],
                 ctx_capture->shape[3]);
  }
  if (raw_ctx_capture != nullptr) {
    std::fprintf(stdout,
                 "%s.kqv_raw_shape: ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 layer_prefix.c_str(),
                 raw_ctx_capture->shape[0],
                 raw_ctx_capture->shape[1],
                 raw_ctx_capture->shape[2],
                 raw_ctx_capture->shape[3]);
  }
  const std::string attn_out_key = "attn_out-" + std::to_string(layer_index);
  const reference_tensor_capture * attn_out_capture =
      find_reference_capture(graph_capture, attn_out_key.c_str());
  if (attn_out_capture != nullptr) {
    std::fprintf(stdout,
                 "%s.attn_out_shape: ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 layer_prefix.c_str(),
                 attn_out_capture->shape[0],
                 attn_out_capture->shape[1],
                 attn_out_capture->shape[2],
                 attn_out_capture->shape[3]);
  }
  const std::string ffn_inp_key = "ffn_inp-" + std::to_string(layer_index);
  const reference_tensor_capture * ffn_inp_capture =
      find_reference_capture(graph_capture, ffn_inp_key.c_str());
  if (ffn_inp_capture != nullptr) {
    std::fprintf(stdout,
                 "%s.ffn_inp_shape: ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 layer_prefix.c_str(),
                 ffn_inp_capture->shape[0],
                 ffn_inp_capture->shape[1],
                 ffn_inp_capture->shape[2],
                 ffn_inp_capture->shape[3]);
  }
  const std::string ffn_out_key = "ffn_out-" + std::to_string(layer_index);
  const reference_tensor_capture * ffn_out_capture =
      find_reference_capture(graph_capture, ffn_out_key.c_str());
  if (ffn_out_capture != nullptr) {
    std::fprintf(stdout,
                 "%s.ffn_out_shape: ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 layer_prefix.c_str(),
                 ffn_out_capture->shape[0],
                 ffn_out_capture->shape[1],
                 ffn_out_capture->shape[2],
                 ffn_out_capture->shape[3]);
  }

  if (!reference_ctx.empty()) {
    auto dump_ctx_variant = [&](const char * suffix, std::span<const float> values) {
      float ctx_max_abs = 0.0f;
      int32_t ctx_max_idx = 0;
      for (int32_t idx = 0; idx < static_cast<int32_t>(values.size()); ++idx) {
        const float diff = std::fabs(values[static_cast<size_t>(idx)] -
                                     reference_ctx[static_cast<size_t>(idx)]);
        if (diff > ctx_max_abs) {
          ctx_max_abs = diff;
          ctx_max_idx = idx;
        }
      }
      std::fprintf(stdout,
                   "%s.%s: max_abs=%g idx=%d emel=%g reference=%g\n",
                   layer_prefix.c_str(),
                   suffix,
                   ctx_max_abs,
                   ctx_max_idx,
                   values[static_cast<size_t>(ctx_max_idx)],
                   reference_ctx[static_cast<size_t>(ctx_max_idx)]);
    };

    dump_ctx_variant("kqv_out_from_emel", backend.attn_ctx);
    dump_ctx_variant("kqv_out_from_emel_ggml_f16_dot", ctx_emel_ggml_f16_dot);
    dump_ctx_variant("kqv_out_from_reference_all_ggml_f16_dot", ctx_reference_all_ggml_f16_dot);
    dump_ctx_variant("kqv_out_from_reference_probs", ctx_reference_probs);
    dump_ctx_variant("kqv_out_from_reference_values", ctx_reference_values);
    if (!raw_reference_ctx.empty()) {
      dump_ctx_variant("kqv_raw", raw_reference_ctx);
    }
  }

  if (position_limit == 1 && backend.n_head == backend.n_head_kv &&
      backend.attn_ctx.size() == backend.v.size()) {
    float internal_v_max_abs = 0.0f;
    for (size_t idx = 0; idx < backend.attn_ctx.size(); ++idx) {
      internal_v_max_abs =
          std::max(internal_v_max_abs, std::fabs(backend.attn_ctx[idx] - backend.v[idx]));
    }
    std::fprintf(stdout,
                 "%s.kqv_out_from_internal_v: max_abs=%g\n",
                 layer_prefix.c_str(),
                 internal_v_max_abs);
  }

  return true;
}

void dump_logits_compare(const generation_load_state & state,
                         std::span<const float> emel_embedding,
                         const std::vector<float> & emel_logits,
                         const emel::paritychecker::parity_options & opts) {
  llama_context_params context_params = llama_context_default_params();
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  context_params.n_ctx = 0;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.n_threads = 1;
  context_params.n_threads_batch = 1;
  context_params.embeddings = true;
  reference_graph_capture graph_capture = {};
  context_params.cb_eval = capture_reference_eval_tensor;
  context_params.cb_eval_user_data = &graph_capture;
  llama_context_ptr ctx = llama_context_ptr{
      state.backend.model != nullptr
          ? llama_init_from_model(state.backend.model.get(), context_params)
          : nullptr,
      llama_free,
  };
  if (ctx == nullptr) {
    std::fprintf(stdout, "tensor_compare.logits: reference context failed\n");
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "tensor_compare.logits: tokenize failed\n");
    return;
  }

  llama_batch prompt_batch =
      llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
  if (llama_decode(ctx.get(), prompt_batch) != 0) {
    std::fprintf(stdout, "tensor_compare.logits: decode failed\n");
    return;
  }

  const float * reference_logits = llama_get_logits_ith(ctx.get(), -1);
  if (reference_logits == nullptr) {
    std::fprintf(stdout, "tensor_compare.logits: logits unavailable\n");
    return;
  }
  const float * reference_embedding = llama_get_embeddings_ith(ctx.get(), -1);
  if (reference_embedding != nullptr) {
    float embedding_max_abs = 0.0f;
    int32_t embedding_idx = 0;
    for (int32_t idx = 0; idx < state.model_data->params.n_embd; ++idx) {
      const float diff =
          std::fabs(emel_embedding[static_cast<size_t>(idx)] - reference_embedding[idx]);
      if (diff > embedding_max_abs) {
        embedding_max_abs = diff;
        embedding_idx = idx;
      }
    }
    std::fprintf(stdout,
                 "tensor_compare.embedding: max_abs=%g idx=%d emel=%g reference=%g\n",
                 embedding_max_abs,
                 embedding_idx,
                 emel_embedding[static_cast<size_t>(embedding_idx)],
                 reference_embedding[embedding_idx]);
  }

  float max_abs = 0.0f;
  int32_t max_idx = 0;
  int32_t emel_best = 0;
  int32_t reference_best = 0;
  float emel_best_score = emel_logits[0];
  float reference_best_score = reference_logits[0];
  for (int32_t idx = 0; idx < state.backend.vocab_size; ++idx) {
    const float emel_score = emel_logits[static_cast<size_t>(idx)];
    const float reference_score = reference_logits[idx];
    const float diff = std::fabs(emel_score - reference_score);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
    }
    if (emel_score > emel_best_score) {
      emel_best_score = emel_score;
      emel_best = idx;
    }
    if (reference_score > reference_best_score) {
      reference_best_score = reference_score;
      reference_best = idx;
    }
  }

  std::fprintf(stdout,
               "tensor_compare.logits: max_abs=%g idx=%d emel=%g reference=%g "
               "emel_best=%d reference_best=%d\n",
               max_abs,
               max_idx,
               emel_logits[static_cast<size_t>(max_idx)],
               reference_logits[max_idx],
               emel_best,
               reference_best);
}

void dump_generation_tensor_compare(generation_load_state & state,
                                    const emel::paritychecker::parity_options & opts) {
  emel::generator::detail::native_backend backend = {};
  if (state.model_data == nullptr ||
      emel::generator::detail::prepare(backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "tensor_compare: backend prepare failed\n");
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "tensor_compare: tokenize failed\n");
    return;
  }

  std::fprintf(stdout,
               "tensor_compare: prompt_tokens=%zu first=%d\n",
               prompt_tokens.size(),
               static_cast<int>(prompt_tokens.front()));
  std::fprintf(stdout,
               "tensor_compare.params: n_embd=%d n_head=%d n_head_kv=%d n_layer=%d "
               "n_rot=%d rms_eps=%g rope_freq_base=%g use_parallel_residual=%d embedding_scale=%g residual_scale=%g "
               "attn_clamp=%g attn_softcap=%g final_softcap=%g\n",
               state.model_data->params.n_embd,
               state.model_data->params.n_head,
               state.model_data->params.n_head_kv,
               state.model_data->params.n_layer,
               state.model_data->params.n_rot,
               state.model_data->params.attention_layer_norm_rms_epsilon,
               state.model_data->params.rope_freq_base,
               state.model_data->params.use_parallel_residual ? 1 : 0,
               state.model_data->params.embedding_scale,
               state.model_data->params.residual_scale,
               state.model_data->params.attention_clamp_kqv,
               state.model_data->params.attn_logit_softcapping,
               state.model_data->params.final_logit_softcapping);
  int32_t bias_tensor_count = 0;
  int32_t extra_norm_tensor_count = 0;
  for (uint32_t idx = 0; idx < state.model_data->n_tensors; ++idx) {
    const auto name = emel::model::tensor_name_view(*state.model_data, state.model_data->tensors[idx]);
    if (name.find("bias") != std::string_view::npos) {
      if (bias_tensor_count < 8) {
        std::fprintf(stdout, "tensor_compare.bias[%d]: %.*s\n",
                     bias_tensor_count,
                     static_cast<int>(name.size()),
                     name.data());
      }
      bias_tensor_count += 1;
    }
    if (name.find("q_norm") != std::string_view::npos ||
        name.find("k_norm") != std::string_view::npos ||
        name.find("attn_q_norm") != std::string_view::npos ||
        name.find("attn_k_norm") != std::string_view::npos) {
      if (extra_norm_tensor_count < 8) {
        std::fprintf(stdout, "tensor_compare.extra[%d]: %.*s\n",
                     extra_norm_tensor_count,
                     static_cast<int>(name.size()),
                     name.data());
      }
      extra_norm_tensor_count += 1;
    }
  }
  std::fprintf(stdout, "tensor_compare: bias_tensor_count=%d\n", bias_tensor_count);
  std::fprintf(stdout, "tensor_compare: extra_norm_tensor_count=%d\n", extra_norm_tensor_count);

  int32_t conditioned_count = 0;
  int32_t conditioned_error = 0;
  std::array<int32_t, 64> conditioned_tokens = {};
  emel::text::conditioner::event::prepare prepare_ev{conditioned_count, conditioned_error};
  prepare_ev.input = opts.text;
  prepare_ev.token_ids_out = conditioned_tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(conditioned_tokens.size());
  const bool conditioned_accepted = state.conditioner.process_event(prepare_ev);
  std::fprintf(stdout,
               "tensor_compare: conditioner accepted=%d count=%d error=%d",
               conditioned_accepted ? 1 : 0,
               conditioned_count,
               conditioned_error);
  for (int32_t idx = 0; idx < conditioned_count; ++idx) {
    std::fprintf(stdout, " %d", conditioned_tokens[static_cast<size_t>(idx)]);
  }
  std::fprintf(stdout, "\n");

  dump_row_compare(
      "tensor_compare.token_embedding",
      *backend.token_embedding.tensor,
      prompt_tokens.front());
  emel::model::llama::detail::block_view block_view = {};
  if (emel::model::llama::detail::lookup_block_view(backend.execution, 0, block_view) ==
      emel::error::cast(emel::model::loader::error::none)) {
    dump_vector_compare(
        "tensor_compare.layer0.attn_norm",
        *block_view.attention_norm.tensor,
        backend.blocks[0].attention_norm);
    dump_vector_compare(
        "tensor_compare.layer0.ffn_norm",
        *block_view.feed_forward_norm.tensor,
        backend.blocks[0].feed_forward_norm);
  }
  if (emel::model::llama::detail::lookup_block_view(backend.execution, 1, block_view) ==
      emel::error::cast(emel::model::loader::error::none)) {
    dump_vector_compare(
        "tensor_compare.layer1.attn_norm",
        *block_view.attention_norm.tensor,
        backend.blocks[1].attention_norm);
    dump_vector_compare(
        "tensor_compare.layer1.ffn_norm",
        *block_view.feed_forward_norm.tensor,
        backend.blocks[1].feed_forward_norm);
  }
  dump_vector_compare("tensor_compare.output_norm", *backend.execution.output_norm.tensor, backend.output_norm);

  reference_graph_capture graph_capture = {};
  const bool have_reference_graph = capture_reference_graph(state, opts, graph_capture);
  std::fprintf(stdout, "tensor_compare: reference_graph=%d\n", have_reference_graph ? 1 : 0);

  if (!emel::generator::detail::copy_tensor_row(
          *backend.token_embedding.tensor, prompt_tokens.front(), backend.hidden) ||
      !emel::generator::detail::rms_norm(
          backend.hidden, backend.blocks[0].attention_norm, backend.rms_epsilon, backend.norm)) {
    std::fprintf(stdout, "tensor_compare: prefill prep failed\n");
    return;
  }

  auto compare_layer = [&](const int32_t layer, const int32_t position) {
    auto & block = backend.blocks[static_cast<size_t>(layer)];
    const std::string prefix = "tensor_compare.layer" + std::to_string(layer);
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.attn_norm_state",
          backend.norm,
          find_reference_tensor(graph_capture, "attn_norm-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.attn_norm_state",
          backend.norm,
          find_reference_tensor(graph_capture, "attn_norm-1"));
    }

    dump_matrix_compare((prefix + ".attn_q").c_str(), backend, block.attention_q, backend.norm);
    dump_matrix_compare((prefix + ".attn_k").c_str(), backend, block.attention_k, backend.norm);
    dump_matrix_compare((prefix + ".attn_v").c_str(), backend, block.attention_v, backend.norm);

    if (!emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
      std::fprintf(stdout, "%s: qkv matmul failed\n", prefix.c_str());
      return false;
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.v_state",
          backend.v,
          find_reference_tensor(graph_capture, "Vcur-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.v_state",
          backend.v,
          find_reference_tensor(graph_capture, "Vcur-1"));
    }

    emel::generator::detail::apply_rope(
        backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
    emel::generator::detail::apply_rope(
        backend.k, backend.n_head_kv, backend.head_dim_kv, backend.n_rot, position, backend.rope_freq_base);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
        backend.q_attn.data());

    const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
    const size_t cache_offset = emel::generator::detail::layer_cache_offset(backend, layer, position, kv_dim);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
        backend.key_cache.data() + cache_offset);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
        backend.value_cache.data() + cache_offset);

    if (!compute_attention_with_softmax_debug(backend, graph_capture, layer, position + 1, prefix)) {
      std::fprintf(stdout, "%s: attention failed\n", prefix.c_str());
      return false;
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.kqv_out_state",
          backend.attn_ctx,
          find_reference_tensor(graph_capture, "kqv_out-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.kqv_out_state",
          backend.attn_ctx,
          find_reference_tensor(graph_capture, "kqv_out-1"));
    }

    dump_matrix_compare(
        (prefix + ".attn_out").c_str(), backend, block.attention_output, backend.attn_ctx);
    if (!emel::generator::detail::matmul_vector(
            backend, block.attention_output, backend.attn_ctx, backend.projected)) {
      std::fprintf(stdout, "%s: attention output matmul failed\n", prefix.c_str());
      return false;
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.attn_out_state",
          backend.projected,
          find_reference_tensor(graph_capture, "attn_out-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.attn_out_state",
          backend.projected,
          find_reference_tensor(graph_capture, "attn_out-1"));
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }

    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm)) {
      std::fprintf(stdout, "%s: ffn rms_norm failed\n", prefix.c_str());
      return false;
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.ffn_norm_state",
          backend.norm,
          find_reference_tensor(graph_capture, "ffn_norm-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.ffn_norm_state",
          backend.norm,
          find_reference_tensor(graph_capture, "ffn_norm-1"));
    }

    dump_matrix_compare((prefix + ".ffn_gate").c_str(), backend, block.feed_forward_gate, backend.norm);
    dump_matrix_compare((prefix + ".ffn_up").c_str(), backend, block.feed_forward_up, backend.norm);
    if (!emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
        !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
      std::fprintf(stdout, "%s: ffn gate/up matmul failed\n", prefix.c_str());
      return false;
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.ffn_gate_state",
          backend.gate,
          find_reference_tensor(graph_capture, "ffn_gate-0"));
      dump_state_compare(
          "tensor_compare.layer0.ffn_up_state",
          backend.up,
          find_reference_tensor(graph_capture, "ffn_up-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.ffn_gate_state",
          backend.gate,
          find_reference_tensor(graph_capture, "ffn_gate-1"));
      dump_state_compare(
          "tensor_compare.layer1.ffn_up_state",
          backend.up,
          find_reference_tensor(graph_capture, "ffn_up-1"));
    }

    for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
      backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.ffn_swiglu_state",
          backend.ffn_hidden,
          find_reference_tensor(graph_capture, "ffn_swiglu-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.ffn_swiglu_state",
          backend.ffn_hidden,
          find_reference_tensor(graph_capture, "ffn_swiglu-1"));
    }

    dump_matrix_compare((prefix + ".ffn_down").c_str(), backend, block.feed_forward_down, backend.ffn_hidden);
    if (!emel::generator::detail::matmul_vector(
            backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
      std::fprintf(stdout, "%s: ffn down matmul failed\n", prefix.c_str());
      return false;
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
    if (layer == 0) {
      dump_state_compare(
          "tensor_compare.layer0.l_out_state",
          backend.hidden,
          find_reference_tensor(graph_capture, "l_out-0"));
    } else if (layer == 1) {
      dump_state_compare(
          "tensor_compare.layer1.l_out_state",
          backend.hidden,
          find_reference_tensor(graph_capture, "l_out-1"));
    }
    return true;
  };

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!compare_layer(layer, 0)) {
      return;
    }
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm)) {
    std::fprintf(stdout, "tensor_compare: final rms_norm failed\n");
    return;
  }
  dump_state_compare(
      "tensor_compare.result_norm_state",
      backend.norm,
      find_reference_tensor(graph_capture, "result_norm"));
  dump_matrix_compare("tensor_compare.output", backend, backend.output, backend.norm);
  if (!emel::generator::detail::matmul_vector(backend, backend.output, backend.norm, backend.bound_logits)) {
    std::fprintf(stdout, "tensor_compare: final logits matmul failed\n");
    return;
  }
  dump_logits_compare(state, backend.norm, backend.bound_logits, opts);
}

void dump_generation_prefix_state_debug(const generation_load_state & state,
                                        const emel::paritychecker::parity_options & opts,
                                        const generation_result & emel_result,
                                        const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    std::fprintf(stdout, "generation_debug.state: tokenize failed\n");
    return;
  }

  std::vector<int32_t> prefix_tokens;
  prefix_tokens.reserve(prompt_tokens.size() + static_cast<size_t>(token_mismatch_index));
  for (const llama_token token : prompt_tokens) {
    prefix_tokens.push_back(static_cast<int32_t>(token));
  }
  for (int32_t idx = 0; idx < token_mismatch_index; ++idx) {
    prefix_tokens.push_back(reference_result.trace.token_ids[static_cast<size_t>(idx)]);
  }
  if (prefix_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.state: empty prefix\n");
    return;
  }

  reference_graph_capture graph_capture = {};
  const std::span<const int32_t> generated_prefix{
      prefix_tokens.data() + prompt_tokens.size(),
      prefix_tokens.size() - prompt_tokens.size(),
  };
  if (!capture_reference_graph_for_generation_prefix(
          state, prompt_tokens, generated_prefix, graph_capture)) {
    std::fprintf(stdout, "generation_debug.state: reference graph capture failed\n");
    return;
  }

  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx == nullptr ||
      !run_reference_prefix_decode(reference_ctx.get(), prompt_tokens, generated_prefix)) {
    std::fprintf(stdout, "generation_debug.state: reference decode replay failed\n");
    return;
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.state: backend prepare failed\n");
    return;
  }

  std::vector<std::vector<float>> reference_value_cache_rows(
      static_cast<size_t>(backend.n_layer));
  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!capture_reference_value_cache_rows(
            reference_ctx.get(),
            layer,
            reference_value_cache_rows[static_cast<size_t>(layer)])) {
      std::fprintf(stdout,
                   "generation_debug.state.layer%d.reference_value_cache: unavailable\n",
                   layer);
    }
  }

  if (prefix_tokens.size() > 1u) {
    const std::span<const int32_t> prior_tokens{prefix_tokens.data(), prefix_tokens.size() - 1u};
    if (!run_prefill_from_token_prefix(backend, prior_tokens)) {
      std::fprintf(stdout, "generation_debug.state: prior prefix replay failed\n");
      return;
    }
  } else {
    backend.kv_cache_tokens = 0;
  }

  const int32_t token_id = prefix_tokens.back();
  const int32_t position = static_cast<int32_t>(prefix_tokens.size() - 1u);
  if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
    std::fprintf(stdout, "generation_debug.state: token embedding replay failed\n");
    return;
  }

  auto compare_layer = [&](const int32_t layer) -> bool {
    auto & block = backend.blocks[static_cast<size_t>(layer)];
    const std::string layer_prefix = "generation_debug.state.layer" + std::to_string(layer);
    const std::string layer_suffix = std::to_string(layer);

    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm)) {
      std::fprintf(stdout, "%s.attn_norm: rms_norm failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".attn_norm").c_str(),
                       backend.norm,
                       find_reference_tensor(graph_capture, ("attn_norm-" + layer_suffix).c_str()));
    dump_q8_quantize_compare((layer_prefix + ".attn_norm_q8").c_str(), backend.norm);
    dump_matrix_compare((layer_prefix + ".attn_q").c_str(), backend, block.attention_q, backend.norm);
    dump_matrix_compare_reference_q8(
        (layer_prefix + ".attn_q_refq8").c_str(), backend, block.attention_q, backend.norm);
    dump_matrix_compare((layer_prefix + ".attn_k").c_str(), backend, block.attention_k, backend.norm);
    dump_matrix_compare_reference_q8(
        (layer_prefix + ".attn_k_refq8").c_str(), backend, block.attention_k, backend.norm);
    dump_matrix_compare((layer_prefix + ".attn_v").c_str(), backend, block.attention_v, backend.norm);
    dump_matrix_compare_reference_q8(
        (layer_prefix + ".attn_v_refq8").c_str(), backend, block.attention_v, backend.norm);

    if (!emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
      std::fprintf(stdout, "%s.qkv: matmul failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".v").c_str(),
                       backend.v,
                       find_reference_tensor(graph_capture, ("Vcur-" + layer_suffix).c_str()));

    emel::generator::detail::apply_rope(
        backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
    emel::generator::detail::apply_rope(backend.k,
                                        backend.n_head_kv,
                                        backend.head_dim_kv,
                                        backend.n_rot,
                                        position,
                                        backend.rope_freq_base);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
        backend.q_attn.data());
    dump_state_compare((layer_prefix + ".q").c_str(),
                       backend.q,
                       find_reference_tensor(graph_capture, ("Qcur-" + layer_suffix).c_str()));
    dump_state_compare((layer_prefix + ".k").c_str(),
                       backend.k,
                       find_reference_tensor(graph_capture, ("Kcur-" + layer_suffix).c_str()));

    const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
    const size_t cache_offset =
        emel::generator::detail::layer_cache_offset(backend, layer, position, kv_dim);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
        backend.key_cache.data() + cache_offset);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
        backend.value_cache.data() + cache_offset);

    if (!compute_attention_with_softmax_debug(
            backend,
            graph_capture,
            layer,
            position + 1,
            layer_prefix,
            reference_value_cache_rows[static_cast<size_t>(layer)])) {
      std::fprintf(stdout, "%s.attention: compute failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".kqv_out").c_str(),
                       backend.attn_ctx,
                       find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
    dump_q8_quantize_compare((layer_prefix + ".attn_ctx_q8").c_str(), backend.attn_ctx);
    dump_matrix_compare(
        (layer_prefix + ".attn_out_matmul").c_str(), backend, block.attention_output, backend.attn_ctx);
    dump_matrix_compare_reference_q8((layer_prefix + ".attn_out_matmul_refq8").c_str(),
                                     backend,
                                     block.attention_output,
                                     backend.attn_ctx);

    if (!emel::generator::detail::matmul_vector(
            backend, block.attention_output, backend.attn_ctx, backend.projected)) {
      std::fprintf(stdout, "%s.attn_out: matmul failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".attn_out").c_str(),
                       backend.projected,
                       find_reference_tensor(graph_capture, ("attn_out-" + layer_suffix).c_str()));

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
    dump_state_compare((layer_prefix + ".ffn_inp").c_str(),
                       backend.hidden,
                       find_reference_tensor(graph_capture, ("ffn_inp-" + layer_suffix).c_str()));

    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm)) {
      std::fprintf(stdout, "%s.ffn_norm: rms_norm failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".ffn_norm").c_str(),
                       backend.norm,
                       find_reference_tensor(graph_capture, ("ffn_norm-" + layer_suffix).c_str()));
    dump_q8_quantize_compare((layer_prefix + ".ffn_norm_q8").c_str(), backend.norm);
    dump_matrix_compare((layer_prefix + ".ffn_gate_matmul").c_str(),
                        backend,
                        block.feed_forward_gate,
                        backend.norm);
    dump_matrix_compare_reference_q8((layer_prefix + ".ffn_gate_matmul_refq8").c_str(),
                                     backend,
                                     block.feed_forward_gate,
                                     backend.norm);
    dump_matrix_compare((layer_prefix + ".ffn_up_matmul").c_str(),
                        backend,
                        block.feed_forward_up,
                        backend.norm);
    dump_matrix_compare_reference_q8((layer_prefix + ".ffn_up_matmul_refq8").c_str(),
                                     backend,
                                     block.feed_forward_up,
                                     backend.norm);

    if (!emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
        !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
      std::fprintf(stdout, "%s.ffn_gate_up: matmul failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".ffn_gate").c_str(),
                       backend.gate,
                       find_reference_tensor(graph_capture, ("ffn_gate-" + layer_suffix).c_str()));
    dump_state_compare((layer_prefix + ".ffn_up").c_str(),
                       backend.up,
                       find_reference_tensor(graph_capture, ("ffn_up-" + layer_suffix).c_str()));

    for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
      backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
    }
    dump_state_compare((layer_prefix + ".ffn_swiglu").c_str(),
                       backend.ffn_hidden,
                       find_reference_tensor(graph_capture, ("ffn_swiglu-" + layer_suffix).c_str()));
    dump_q8_quantize_compare((layer_prefix + ".ffn_hidden_q8").c_str(), backend.ffn_hidden);
    dump_matrix_compare((layer_prefix + ".ffn_down_matmul").c_str(),
                        backend,
                        block.feed_forward_down,
                        backend.ffn_hidden);
    dump_matrix_compare_reference_q8((layer_prefix + ".ffn_down_matmul_refq8").c_str(),
                                     backend,
                                     block.feed_forward_down,
                                     backend.ffn_hidden);

    if (!emel::generator::detail::matmul_vector(
            backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
      std::fprintf(stdout, "%s.ffn_down: matmul failed\n", layer_prefix.c_str());
      return false;
    }
    dump_state_compare((layer_prefix + ".ffn_out").c_str(),
                       backend.projected,
                       find_reference_tensor(graph_capture, ("ffn_out-" + layer_suffix).c_str()));

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
    dump_state_compare((layer_prefix + ".l_out").c_str(),
                       backend.hidden,
                       find_reference_tensor(graph_capture, ("l_out-" + layer_suffix).c_str()));
    return true;
  };

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!compare_layer(layer)) {
      return;
    }
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm)) {
    std::fprintf(stdout, "generation_debug.state.result_norm: rms_norm failed\n");
    return;
  }
  dump_state_compare(
      "generation_debug.state.result_norm",
      backend.norm,
      find_reference_tensor(graph_capture, "result_norm"));
}

double l2_norm_diff(std::span<const float> emel_values, std::span<const float> reference_values) {
  if (emel_values.size() != reference_values.size()) {
    return -1.0;
  }

  double square_sum = 0.0;
  for (size_t idx = 0; idx < emel_values.size(); ++idx) {
    const double diff = static_cast<double>(emel_values[idx]) -
                        static_cast<double>(reference_values[idx]);
    square_sum += diff * diff;
  }
  return std::sqrt(square_sum);
}

float max_abs_diff(std::span<const float> emel_values, std::span<const float> reference_values) {
  if (emel_values.size() != reference_values.size()) {
    return -1.0f;
  }

  float max_abs = 0.0f;
  for (size_t idx = 0; idx < emel_values.size(); ++idx) {
    max_abs = std::max(max_abs, std::fabs(emel_values[idx] - reference_values[idx]));
  }
  return max_abs;
}

std::span<const float> reference_token_row(std::span<const float> reference_values,
                                           const size_t row_width,
                                           const size_t row_index) {
  if (row_width == 0 || reference_values.empty() || reference_values.size() % row_width != 0) {
    return {};
  }

  const size_t row_count = reference_values.size() / row_width;
  if (row_index >= row_count) {
    return {};
  }

  return reference_values.subspan(row_index * row_width, row_width);
}

std::span<const float> reference_token_tensor_slice(const reference_tensor_capture & capture,
                                                    const int32_t token_index) {
  const int64_t ne0 = std::max<int64_t>(capture.shape[0], 1);
  const int64_t ne1 = std::max<int64_t>(capture.shape[1], 1);
  const int64_t ne2 = std::max<int64_t>(capture.shape[2], 1);
  if (token_index < 0 || token_index >= ne2) {
    return {};
  }

  const size_t row_width = static_cast<size_t>(ne0 * ne1);
  const size_t offset = static_cast<size_t>(token_index) * row_width;
  if (offset + row_width > capture.values.size()) {
    return {};
  }
  return std::span<const float>(capture.values).subspan(offset, row_width);
}

std::span<const float> reference_last_token_row(std::span<const float> reference_values,
                                                const size_t row_width) {
  if (row_width == 0 || reference_values.empty()) {
    return {};
  }
  if (reference_values.size() == row_width) {
    return reference_values;
  }
  if (reference_values.size() % row_width != 0) {
    return {};
  }

  const size_t row_count = reference_values.size() / row_width;
  const size_t row_offset = (row_count - 1u) * row_width;
  return reference_values.subspan(row_offset, row_width);
}

void dump_generation_prefix_timeline_debug(const generation_load_state & state,
                                           const emel::paritychecker::parity_options & opts,
                                           const generation_result & emel_result,
                                           const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  const int32_t prefix_generated_tokens = std::min<int32_t>(12, token_mismatch_index);
  if (state.model_data == nullptr || prefix_generated_tokens <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    std::fprintf(stdout, "generation_debug.timeline: tokenize failed\n");
    return;
  }

  std::vector<int32_t> prefix_tokens;
  prefix_tokens.reserve(prompt_tokens.size() + static_cast<size_t>(prefix_generated_tokens));
  for (const llama_token token : prompt_tokens) {
    prefix_tokens.push_back(static_cast<int32_t>(token));
  }
  for (int32_t idx = 0; idx < prefix_generated_tokens; ++idx) {
    prefix_tokens.push_back(reference_result.trace.token_ids[static_cast<size_t>(idx)]);
  }

  std::vector<llama_token> prefix_tokens_llama;
  prefix_tokens_llama.reserve(prefix_tokens.size());
  for (const int32_t token : prefix_tokens) {
    prefix_tokens_llama.push_back(static_cast<llama_token>(token));
  }

  reference_graph_capture graph_capture = {};
  if (!capture_reference_graph_for_tokens(state, prefix_tokens_llama, graph_capture)) {
    std::fprintf(stdout, "generation_debug.timeline: reference graph capture failed\n");
    return;
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.timeline: backend prepare failed\n");
    return;
  }

  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx == nullptr) {
    std::fprintf(stdout, "generation_debug.timeline: reference context failed\n");
    return;
  }

  const reference_tensor_capture * layer0_v_capture = find_reference_capture(graph_capture, "Vcur-0");
  const reference_tensor_capture * layer0_q_capture = find_reference_capture(graph_capture, "Qcur-0");
  const reference_tensor_capture * layer0_k_capture = find_reference_capture(graph_capture, "Kcur-0");
  const std::span<const float> layer0_kqv_out_capture =
      find_reference_tensor(graph_capture, "kqv_out-0");
  const std::span<const float> layer0_attn_out_capture =
      find_reference_tensor(graph_capture, "attn_out-0");
  const reference_tensor_capture * layer1_v_capture = find_reference_capture(graph_capture, "Vcur-1");
  const size_t prompt_count = prompt_tokens.size();
  std::vector<float> layer0_value_cache0_baseline;
  std::vector<float> layer1_value_cache0_baseline;

  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    llama_token reference_token = static_cast<llama_token>(token_id);
    llama_batch decode_batch = llama_batch_get_one(&reference_token, 1);
    if (llama_decode(reference_ctx.get(), decode_batch) != 0) {
      std::fprintf(stdout, "generation_debug.timeline: reference decode failed\n");
      return;
    }
    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      std::fprintf(stdout, "generation_debug.timeline: token embedding replay failed\n");
      return;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention(backend, layer, position)) {
        std::fprintf(stdout, "generation_debug.timeline: layer replay failed\n");
        return;
      }

      const bool is_prompt_token = token_index < prompt_count;
      const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
      const std::string token_prefix = is_prompt_token
                                           ? "generation_debug.timeline.prompt" +
                                                 std::to_string(token_index)
                                           : "generation_debug.timeline.gen" +
                                                 std::to_string(generated_index);
      const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer, 0, kv_dim);
      const std::span<const float> cache_row0{
          backend.value_cache.data() + cache_offset,
          static_cast<size_t>(kv_dim),
      };
      if (layer == 0) {
        std::vector<float> reference_layer0_key_cache;
        std::vector<float> reference_layer0_value_cache;
        if (capture_reference_key_cache_rows(reference_ctx.get(), 0, reference_layer0_key_cache) &&
            reference_layer0_key_cache.size() == static_cast<size_t>((position + 1) * kv_dim)) {
          float max_cache_abs = 0.0f;
          for (int32_t cache_pos = 0; cache_pos <= position; ++cache_pos) {
            const size_t cache_offset_compare = emel::generator::detail::layer_cache_offset(
                backend, layer, cache_pos, kv_dim);
            const std::span<const float> emel_cache_row{
                backend.key_cache.data() + cache_offset_compare,
                static_cast<size_t>(kv_dim),
            };
            const std::span<const float> reference_cache_row =
                std::span<const float>(reference_layer0_key_cache)
                    .subspan(static_cast<size_t>(cache_pos) * static_cast<size_t>(kv_dim),
                             static_cast<size_t>(kv_dim));
            max_cache_abs = std::max(max_cache_abs, max_abs_diff(emel_cache_row, reference_cache_row));
          }
          std::fprintf(stdout,
                       "%s.layer0_key_cache: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_cache_abs);
        }
        if (capture_reference_value_cache_rows(reference_ctx.get(), 0, reference_layer0_value_cache) &&
            reference_layer0_value_cache.size() == static_cast<size_t>((position + 1) * kv_dim)) {
          float max_cache_abs = 0.0f;
          for (int32_t cache_pos = 0; cache_pos <= position; ++cache_pos) {
            const size_t cache_offset_compare = emel::generator::detail::layer_cache_offset(
                backend, layer, cache_pos, kv_dim);
            const std::span<const float> emel_cache_row{
                backend.value_cache.data() + cache_offset_compare,
                static_cast<size_t>(kv_dim),
            };
            const std::span<const float> reference_cache_row =
                std::span<const float>(reference_layer0_value_cache)
                    .subspan(static_cast<size_t>(cache_pos) * static_cast<size_t>(kv_dim),
                             static_cast<size_t>(kv_dim));
            max_cache_abs = std::max(max_cache_abs, max_abs_diff(emel_cache_row, reference_cache_row));
          }
          std::fprintf(stdout,
                       "%s.layer0_value_cache: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_cache_abs);
        }
        if (token_index == 0u) {
          layer0_value_cache0_baseline.assign(cache_row0.begin(), cache_row0.end());
        } else if (!layer0_value_cache0_baseline.empty()) {
          std::fprintf(stdout,
                       "%s.layer0_value_cache0_drift: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(cache_row0, layer0_value_cache0_baseline));
        }
        if (layer0_v_capture != nullptr) {
          const std::span<const float> reference_v =
              reference_token_tensor_slice(*layer0_v_capture, position);
          std::fprintf(stdout,
                       "%s.layer0_v: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(backend.v, reference_v));
        }
        if (layer0_q_capture != nullptr) {
          const std::span<const float> reference_q =
              reference_token_tensor_slice(*layer0_q_capture, position);
          std::fprintf(stdout,
                       "%s.layer0_q: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(backend.q, reference_q));
        }
        if (layer0_k_capture != nullptr) {
          const std::span<const float> reference_k =
              reference_token_tensor_slice(*layer0_k_capture, position);
          std::fprintf(stdout,
                       "%s.layer0_k: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(backend.k, reference_k));
        }
        const std::span<const float> reference_kqv_out = reference_token_row(
            layer0_kqv_out_capture, backend.attn_ctx.size(), token_index);
        std::fprintf(stdout,
                     "%s.layer0_kqv_out: max_abs=%g\n",
                     token_prefix.c_str(),
                     max_abs_diff(backend.attn_ctx, reference_kqv_out));
        const std::span<const float> reference_attn_out = reference_token_row(
            layer0_attn_out_capture, backend.projected.size(), token_index);
        std::fprintf(stdout,
                     "%s.layer0_attn_out: max_abs=%g\n",
                     token_prefix.c_str(),
                     max_abs_diff(backend.projected, reference_attn_out));
        const std::span<const float> reference_l_out = reference_token_row(
            find_reference_tensor(graph_capture, "l_out-0"), backend.hidden.size(), token_index);
        std::fprintf(stdout,
                     "%s.layer0_l_out: max_abs=%g\n",
                     token_prefix.c_str(),
                     max_abs_diff(backend.hidden, reference_l_out));
      } else if (layer == 1) {
        if (token_index == 0u) {
          layer1_value_cache0_baseline.assign(cache_row0.begin(), cache_row0.end());
        } else if (!layer1_value_cache0_baseline.empty()) {
          std::fprintf(stdout,
                       "%s.layer1_value_cache0_drift: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(cache_row0, layer1_value_cache0_baseline));
        }
        if (layer1_v_capture != nullptr) {
          const std::span<const float> reference_v =
              reference_token_tensor_slice(*layer1_v_capture, position);
          std::fprintf(stdout,
                       "%s.layer1_v: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(backend.v, reference_v));
        }
        const std::span<const float> reference_l_out = reference_token_row(
            find_reference_tensor(graph_capture, "l_out-1"), backend.hidden.size(), token_index);
        std::fprintf(stdout,
                     "%s.layer1_l_out: max_abs=%g\n",
                     token_prefix.c_str(),
                     max_abs_diff(backend.hidden, reference_l_out));
      }
    }
    backend.kv_cache_tokens = position + 1;
  }
}

void dump_generation_residual_l2_debug(const generation_load_state & state,
                                       const emel::paritychecker::parity_options & opts,
                                       const generation_result & emel_result,
                                       const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  const int32_t prefix_generated_tokens = std::min<int32_t>(12, token_mismatch_index);
  if (state.model_data == nullptr || prefix_generated_tokens <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    std::fprintf(stdout, "generation_debug.residual_l2: tokenize failed\n");
    return;
  }

  std::vector<int32_t> prefix_tokens;
  prefix_tokens.reserve(prompt_tokens.size() + static_cast<size_t>(prefix_generated_tokens));
  for (const llama_token token : prompt_tokens) {
    prefix_tokens.push_back(static_cast<int32_t>(token));
  }
  for (int32_t idx = 0; idx < prefix_generated_tokens; ++idx) {
    prefix_tokens.push_back(reference_result.trace.token_ids[static_cast<size_t>(idx)]);
  }
  if (prefix_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.residual_l2: empty prefix\n");
    return;
  }

  reference_graph_capture graph_capture = {};
  const std::span<const int32_t> generated_prefix{
      prefix_tokens.data() + prompt_tokens.size(),
      prefix_tokens.size() - prompt_tokens.size(),
  };
  if (!capture_reference_graph_for_generation_prefix(
          state, prompt_tokens, generated_prefix, graph_capture)) {
    std::fprintf(stdout, "generation_debug.residual_l2: reference graph capture failed\n");
    return;
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.residual_l2: backend prepare failed\n");
    return;
  }

  if (prefix_tokens.size() > 1u) {
    const std::span<const int32_t> prior_tokens{prefix_tokens.data(), prefix_tokens.size() - 1u};
    if (!run_prefill_from_token_prefix(backend, prior_tokens)) {
      std::fprintf(stdout, "generation_debug.residual_l2: prior prefix replay failed\n");
      return;
    }
  } else {
    backend.kv_cache_tokens = 0;
  }

  const int32_t token_id = prefix_tokens.back();
  const int32_t position = static_cast<int32_t>(prefix_tokens.size() - 1u);
  if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
    std::fprintf(stdout, "generation_debug.residual_l2: token embedding replay failed\n");
    return;
  }

  std::fprintf(stdout,
               "generation_debug.residual_l2: generated_prefix=%d total_prefix_tokens=%zu\n",
               prefix_generated_tokens,
               prefix_tokens.size());

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    auto & block = backend.blocks[static_cast<size_t>(layer)];
    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
      std::fprintf(stdout, "generation_debug.residual_l2.layer%d: qkv prep failed\n", layer);
      return;
    }

    emel::generator::detail::apply_rope(
        backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
    emel::generator::detail::apply_rope(backend.k,
                                        backend.n_head_kv,
                                        backend.head_dim_kv,
                                        backend.n_rot,
                                        position,
                                        backend.rope_freq_base);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
        backend.q_attn.data());

    const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
    const size_t cache_offset =
        emel::generator::detail::layer_cache_offset(backend, layer, position, kv_dim);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
        backend.key_cache.data() + cache_offset);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
        backend.value_cache.data() + cache_offset);

    if (!emel::generator::detail::compute_attention(
            backend, layer, position + 1, backend.q_attn) ||
        !emel::generator::detail::matmul_vector(
            backend, block.attention_output, backend.attn_ctx, backend.projected)) {
      std::fprintf(stdout, "generation_debug.residual_l2.layer%d: attention failed\n", layer);
      return;
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }

    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
        !emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
        !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
      std::fprintf(stdout, "generation_debug.residual_l2.layer%d: ffn gate/up failed\n", layer);
      return;
    }

    for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
      backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
    }
    if (!emel::generator::detail::matmul_vector(
            backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
      std::fprintf(stdout, "generation_debug.residual_l2.layer%d: ffn down failed\n", layer);
      return;
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }

    const std::string key = "l_out-" + std::to_string(layer);
    const std::span<const float> reference_hidden =
        reference_last_token_row(find_reference_tensor(graph_capture, key.c_str()), backend.hidden.size());
    const double l2 = l2_norm_diff(backend.hidden, reference_hidden);
    std::fprintf(stdout,
                 "generation_debug.residual_l2.layer%d: l2=%g\n",
                 layer,
                 l2);
  }
}

const char * kernel_kind_name(const emel::kernel::kernel_kind kind) {
  switch (kind) {
    case emel::kernel::kernel_kind::x86_64:
      return "x86_64";
    case emel::kernel::kernel_kind::aarch64:
      return "aarch64";
    case emel::kernel::kernel_kind::wasm:
      return "wasm";
    case emel::kernel::kernel_kind::cuda:
      return "cuda";
    case emel::kernel::kernel_kind::metal:
      return "metal";
    case emel::kernel::kernel_kind::vulkan:
      return "vulkan";
  }
  return "unknown";
}

void dump_reference_decode_seam(const generation_load_state & state) {
  const emel::kernel::kernel_kind kernel_kind = state.generator->generation_kernel_kind();
  const uint64_t optimized_flash_dispatch_calls =
      state.generator->generation_optimized_flash_dispatch_calls();
  const uint64_t shared_flash_dispatch_calls =
      state.generator->generation_shared_flash_dispatch_calls();
  const uint64_t optimized_q2_dispatch_calls =
      state.generator->generation_optimized_q2_dispatch_calls();
  const uint64_t shared_q2_dispatch_calls =
      state.generator->generation_shared_q2_dispatch_calls();
  const uint64_t optimized_q3_dispatch_calls =
      state.generator->generation_optimized_q3_dispatch_calls();
  const uint64_t shared_q3_dispatch_calls =
      state.generator->generation_shared_q3_dispatch_calls();
  const uint64_t optimized_q6_dispatch_calls =
      state.generator->generation_optimized_q6_dispatch_calls();
  const uint64_t shared_q6_dispatch_calls =
      state.generator->generation_shared_q6_dispatch_calls();
  std::fprintf(stdout,
               "reference_impl: source=%.*s ref=%.*s\n",
               static_cast<int>(k_reference_impl_source.size()),
               k_reference_impl_source.data(),
               static_cast<int>(k_reference_impl_ref.size()),
               k_reference_impl_ref.data());
  std::fprintf(stdout,
               "reference_decode_seams: emel_decode_calls=%d emel_logits_calls=%d "
               "reference_decode_calls=%d reference_logits_calls=%d\n",
               state.backend.emel_reference_decode_calls,
               state.backend.emel_reference_logits_calls,
               state.backend.direct_reference_decode_calls,
               state.backend.direct_reference_logits_calls);
  std::fprintf(stdout,
               "kernel_dispatch: kind=%s calls=%" PRIu64 "\n",
               kernel_kind_name(kernel_kind),
               state.generator->generation_kernel_dispatch_calls());
  std::fprintf(stdout,
               "flash_dispatch: calls=%" PRIu64 " optimized=%" PRIu64 " shared=%" PRIu64 "\n",
               state.generator->generation_flash_attention_dispatch_calls(),
               optimized_flash_dispatch_calls,
               shared_flash_dispatch_calls);
  std::fprintf(stdout,
               "quantized_dispatch: optimized_q2_dispatch_calls=%" PRIu64
               " shared_q2_dispatch_calls=%" PRIu64
               " optimized_q3_dispatch_calls=%" PRIu64
               " shared_q3_dispatch_calls=%" PRIu64
               " optimized_q6_dispatch_calls=%" PRIu64
               " shared_q6_dispatch_calls=%" PRIu64 "\n",
               optimized_q2_dispatch_calls,
               shared_q2_dispatch_calls,
               optimized_q3_dispatch_calls,
               shared_q3_dispatch_calls,
               optimized_q6_dispatch_calls,
               shared_q6_dispatch_calls);
}

void dump_generation_failure_surface(generation_load_state & state,
                                     const generation_result * emel_result,
                                     const generation_result * reference_result,
                                     const emel::paritychecker::parity_options & opts) {
  dump_reference_decode_seam(state);
  if (emel_result != nullptr) {
    dump_generation_result("emel", *emel_result);
  }
  if (reference_result != nullptr) {
    dump_generation_result("reference", *reference_result);
  }
  if (emel_result != nullptr && reference_result != nullptr) {
    const int32_t token_mismatch_index = first_token_mismatch_index(*emel_result, *reference_result);
    const bool have_emel_token =
        token_mismatch_index >= 0 && token_mismatch_index < emel_result->trace.token_count;
    const bool have_reference_token =
        token_mismatch_index >= 0 && token_mismatch_index < reference_result->trace.token_count;
    std::fprintf(stdout,
                 "generation_trace: emel_tokens=%d reference_tokens=%d first_token_mismatch=%d\n",
                 emel_result->trace.token_count,
                 reference_result->trace.token_count,
                 token_mismatch_index);
    if (have_emel_token && have_reference_token) {
      std::fprintf(stdout,
                   "generation_trace.mismatch: emel_token=%d emel_gap=%g reference_token=%d "
                   "reference_gap=%g\n",
                   emel_result->trace.token_ids[static_cast<size_t>(token_mismatch_index)],
                   emel_result->trace.top_score_gaps[static_cast<size_t>(token_mismatch_index)],
                   reference_result->trace.token_ids[static_cast<size_t>(token_mismatch_index)],
                   reference_result->trace.top_score_gaps[static_cast<size_t>(token_mismatch_index)]);
      dump_generation_trace_window("emel", state.backend, *emel_result, token_mismatch_index);
      dump_generation_trace_window("reference", state.backend, *reference_result, token_mismatch_index);
    }
    dump_scalar_attention_debug(state, opts, *emel_result, *reference_result);
    dump_generation_residual_l2_debug(state, opts, *emel_result, *reference_result);
    dump_generation_prefix_timeline_debug(state, opts, *emel_result, *reference_result);
    dump_generation_prefix_state_debug(state, opts, *emel_result, *reference_result);
  }
  if (opts.dump && reference_result != nullptr) {
    dump_generation_tensor_compare(state, opts);
  }
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

bool decode_float_value(const generation_load_state & state,
                        const emel::gguf::loader::kv_entry & entry,
                        float & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(state, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  switch (entry.value_type) {
    case constants::gguf_type_float32: {
      if (bytes.size() != sizeof(uint32_t)) {
        return false;
      }
      const uint32_t bits = read_u32_le(bytes);
      std::memcpy(&value_out, &bits, sizeof(value_out));
      return true;
    }
    case constants::gguf_type_float64: {
      if (bytes.size() != sizeof(uint64_t)) {
        return false;
      }
      const uint64_t bits = read_u64_le(bytes);
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      value_out = static_cast<float>(value);
      return true;
    }
    default:
      return false;
  }
}

bool decode_bool_value(const generation_load_state & state,
                       const emel::gguf::loader::kv_entry & entry,
                       bool & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(state, entry);
  namespace constants = emel::gguf::loader::detail::constants;
  if (entry.value_type != constants::gguf_type_bool || bytes.size() != 1u) {
    return false;
  }
  value_out = bytes[0] != 0u;
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

  const auto assign_f32 = [&](const std::string_view key, float & field) {
    const auto * entry = find_kv_entry(state, key);
    if (entry == nullptr) {
      return true;
    }

    float value = 0.0f;
    if (!decode_float_value(state, *entry, value)) {
      return false;
    }

    field = value;
    return true;
  };

  const auto assign_bool = [&](const std::string_view key, bool & field) {
    const auto * entry = find_kv_entry(state, key);
    if (entry == nullptr) {
      return true;
    }

    bool value = false;
    if (!decode_bool_value(state, *entry, value)) {
      return false;
    }

    field = value;
    return true;
  };

  if (!assign_i32("llama.context_length", model_data.params.n_ctx) ||
      !assign_i32("llama.embedding_length", model_data.params.n_embd) ||
      !assign_i32("llama.embedding_length_out", model_data.params.n_embd_out) ||
      !assign_i32("llama.feed_forward_length", model_data.params.n_ff) ||
      !assign_i32("llama.attention.head_count", model_data.params.n_head) ||
      !assign_i32("llama.attention.head_count_kv", model_data.params.n_head_kv) ||
      !assign_i32("llama.rope.dimension_count", model_data.params.n_rot) ||
      !assign_i32("llama.block_count", model_data.params.n_layer) ||
      !assign_i32("llama.vocab_size", model_data.params.n_vocab) ||
      !assign_f32("llama.attention.layer_norm_epsilon", model_data.params.attention_layer_norm_epsilon) ||
      !assign_f32(
          "llama.attention.layer_norm_rms_epsilon",
          model_data.params.attention_layer_norm_rms_epsilon) ||
      !assign_f32("llama.attention.clamp_kqv", model_data.params.attention_clamp_kqv) ||
      !assign_f32("llama.attn_logit_softcapping", model_data.params.attn_logit_softcapping) ||
      !assign_f32("llama.final_logit_softcapping", model_data.params.final_logit_softcapping) ||
      !assign_f32("llama.residual_scale", model_data.params.residual_scale) ||
      !assign_f32("llama.embedding_scale", model_data.params.embedding_scale) ||
      !assign_f32("llama.rope.freq_base", model_data.params.rope_freq_base) ||
      !assign_f32("llama.rope.freq_base_swa", model_data.params.rope_freq_base_swa) ||
      !assign_bool("llama.use_parallel_residual", model_data.params.use_parallel_residual)) {
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
    if (emel::model::try_parse_block_index(
            emel::model::tensor_name_view(req.model_data, req.model_data.tensors[i]),
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
  return emel::model::architecture_name_view(req.model_data) == "llama"
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

  reset_reference_decode_seam(state.backend);
  generation_result emel_result{};
  const emel::error::type generation_err =
      run_emel_generate(state,
                        opts,
                        std::span<char>{emel_result.output},
                        emel_result.output_length,
                        emel_result.trace);
  if (generation_err != emel::error::cast(emel::generator::error::none)) {
    std::fprintf(stderr,
                 "generation error (fixture=%s err=%s generated_tokens=%d output_bytes=%zu)\n",
                 k_generation_fixture_name,
                 generator_error_name(generation_err),
                 state.generation.tokens_generated,
                 state.generation.output_length);
    dump_generation_failure_surface(state, nullptr, nullptr, opts);
    return 1;
  }
  emel_result.tokens_generated = state.generation.tokens_generated;
  emel_result.output_length = state.generation.output_length;
  const emel::kernel::kernel_kind generation_kernel_kind =
      state.generator->generation_kernel_kind();
  const uint64_t flash_dispatch_calls =
      state.generator->generation_flash_attention_dispatch_calls();
  const uint64_t optimized_flash_dispatch_calls =
      state.generator->generation_optimized_flash_dispatch_calls();
  const uint64_t shared_flash_dispatch_calls =
      state.generator->generation_shared_flash_dispatch_calls();
  const uint64_t optimized_q2_dispatch_calls =
      state.generator->generation_optimized_q2_dispatch_calls();
  const uint64_t shared_q2_dispatch_calls =
      state.generator->generation_shared_q2_dispatch_calls();
  const uint64_t optimized_q3_dispatch_calls =
      state.generator->generation_optimized_q3_dispatch_calls();
  const uint64_t shared_q3_dispatch_calls =
      state.generator->generation_shared_q3_dispatch_calls();
  const uint64_t optimized_q6_dispatch_calls =
      state.generator->generation_optimized_q6_dispatch_calls();
  const uint64_t shared_q6_dispatch_calls =
      state.generator->generation_shared_q6_dispatch_calls();
  if (flash_dispatch_calls == 0u) {
    std::fprintf(stderr,
                 "generation flash proof failed (fixture=%s flash_dispatch_calls=%" PRIu64 ")\n",
                 k_generation_fixture_name,
                 flash_dispatch_calls);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  if (generation_kernel_kind == emel::kernel::kernel_kind::aarch64 &&
      (optimized_flash_dispatch_calls == 0u || shared_flash_dispatch_calls != 0u)) {
    std::fprintf(stderr,
                 "generation flash proof failed (fixture=%s kernel_kind=%s "
                 "flash_dispatch_calls=%" PRIu64 " optimized_flash_dispatch_calls=%" PRIu64
                 " shared_flash_dispatch_calls=%" PRIu64 ")\n",
                 k_generation_fixture_name,
                 kernel_kind_name(generation_kernel_kind),
                 flash_dispatch_calls,
                 optimized_flash_dispatch_calls,
                 shared_flash_dispatch_calls);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  if (generation_kernel_kind == emel::kernel::kernel_kind::aarch64 &&
      (optimized_q2_dispatch_calls == 0u || shared_q2_dispatch_calls != 0u ||
       optimized_q3_dispatch_calls == 0u || shared_q3_dispatch_calls != 0u ||
       optimized_q6_dispatch_calls == 0u || shared_q6_dispatch_calls != 0u)) {
    std::fprintf(stderr,
                 "generation quantized proof failed (fixture=%s kernel_kind=%s "
                 "optimized_q2_dispatch_calls=%" PRIu64
                 " shared_q2_dispatch_calls=%" PRIu64
                 " optimized_q3_dispatch_calls=%" PRIu64
                 " shared_q3_dispatch_calls=%" PRIu64
                 " optimized_q6_dispatch_calls=%" PRIu64
                 " shared_q6_dispatch_calls=%" PRIu64 ")\n",
                 k_generation_fixture_name,
                 kernel_kind_name(generation_kernel_kind),
                 optimized_q2_dispatch_calls,
                 shared_q2_dispatch_calls,
                 optimized_q3_dispatch_calls,
                 shared_q3_dispatch_calls,
                 optimized_q6_dispatch_calls,
                 shared_q6_dispatch_calls);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }

  generation_result reference_result{};
  const emel::error::type reference_err =
      run_reference_generate(state.backend, opts, reference_result);
  if (reference_err != emel::error::cast(emel::generator::error::none)) {
    std::fprintf(stderr,
                 "generation reference failed (fixture=%s err=%s)\n",
                 k_generation_fixture_name,
                 generator_error_name(reference_err));
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
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
    dump_generation_failure_surface(state, &emel_result, &reference_result, opts);
    return 1;
  }

  std::fprintf(stdout,
               "generation parity ok (fixture=%s prompt_bytes=%zu max_tokens=%d generated_tokens=%d "
               "output_bytes=%zu flash_dispatch_calls=%" PRIu64
               " optimized_flash_dispatch_calls=%" PRIu64
               " shared_flash_dispatch_calls=%" PRIu64 " text=%.*s)\n",
               k_generation_fixture_name,
               opts.text.size(),
               opts.max_tokens,
               emel_result.tokens_generated,
               emel_result.output_length,
               flash_dispatch_calls,
               optimized_flash_dispatch_calls,
               shared_flash_dispatch_calls,
               static_cast<int>(emel_result.output_length),
               emel_result.output.data());
  dump_reference_decode_seam(state);
  if (opts.dump) {
    dump_generation_result("emel", emel_result);
    dump_generation_result("reference", reference_result);
    dump_generation_tensor_compare(state, opts);
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
