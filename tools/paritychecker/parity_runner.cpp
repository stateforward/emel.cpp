#include "parity_runner.hpp"
#include "tokenizer_parity.hpp"
#include "../generation_formatter_contract.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
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
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/detokenizer/actions.hpp"
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

using reference_ggml_float = double;

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
reference_ggml_float ggml_vec_soft_max_f32(int n, float * y, const float * x, float max);
void ggml_vec_dot_q2_K_q8_K(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q2_K_q8_K_generic(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K_generic(
    int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(
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
constexpr const char * k_generation_fixture_name = "Qwen3-0.6B-Q8_0.gguf";
constexpr const char * k_generation_fixture_slug = "qwen3_0_6b_q8_0";
constexpr size_t k_generation_output_capacity = 65536u;
constexpr size_t k_generation_token_trace_capacity = 4096u;
constexpr std::string_view k_generation_baseline_source = "maintained_generation_baseline";
constexpr std::string_view k_generation_baseline_format = "emel_generation_baseline_v1";
constexpr std::string_view k_generation_baseline_contract =
    "generation_online_f16_final_normalize_v1";
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

struct quantized_contract_summary {
  uint32_t native_quantized = 0u;
  uint32_t approved_dense_f32_by_contract = 0u;
  uint32_t disallowed_fallback = 0u;
  uint32_t explicit_no_claim = 0u;
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

quantized_contract_summary build_quantized_contract_summary(
    const emel::model::llama::detail::quantized_path_audit & audit) {
  quantized_contract_summary summary{};
  for (const auto & stage : audit.stages) {
    summary.native_quantized += static_cast<uint32_t>(
        stage.contract == emel::model::llama::detail::quantized_contract_kind::native_quantized);
    summary.approved_dense_f32_by_contract += static_cast<uint32_t>(
        stage.contract ==
        emel::model::llama::detail::quantized_contract_kind::approved_dense_f32_by_contract);
    summary.disallowed_fallback += static_cast<uint32_t>(
        stage.contract == emel::model::llama::detail::quantized_contract_kind::disallowed_fallback);
    summary.explicit_no_claim += static_cast<uint32_t>(
        stage.contract == emel::model::llama::detail::quantized_contract_kind::explicit_no_claim);
  }
  return summary;
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
  bool trace_available = true;
  generation_trace trace = {};
};

struct generation_baseline_record {
  std::string fixture_name;
  std::string prompt;
  int32_t max_tokens = 0;
  generation_result result = {};
};

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out);

std::filesystem::path paritychecker_repo_root_path() {
#ifdef PARITYCHECKER_REPO_ROOT
  return std::filesystem::path(PARITYCHECKER_REPO_ROOT);
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path generation_baseline_directory_path() {
  return paritychecker_repo_root_path() / "snapshots" / "parity";
}

std::string generation_prompt_slug(std::string_view text) {
  std::string slug;
  slug.reserve(text.size());

  bool last_was_underscore = false;
  for (const char ch : text) {
    const unsigned char byte = static_cast<unsigned char>(ch);
    if (std::isalnum(byte) != 0) {
      slug.push_back(static_cast<char>(std::tolower(byte)));
      last_was_underscore = false;
      continue;
    }
    if (!last_was_underscore) {
      slug.push_back('_');
      last_was_underscore = true;
    }
  }

  while (!slug.empty() && slug.front() == '_') {
    slug.erase(slug.begin());
  }
  while (!slug.empty() && slug.back() == '_') {
    slug.pop_back();
  }
  if (slug.empty()) {
    slug = "empty";
  }
  return slug;
}

std::filesystem::path default_generation_baseline_path(
    const emel::paritychecker::parity_options & opts) {
  const std::string prompt_slug = generation_prompt_slug(opts.text);
  return generation_baseline_directory_path() /
      ("generation_" + std::string(k_generation_fixture_slug) + "_prompt_" + prompt_slug +
       "_max_tokens_" + std::to_string(opts.max_tokens) + ".txt");
}

char hex_digit(const uint8_t nibble) {
  return static_cast<char>(nibble < 10u ? ('0' + nibble) : ('a' + (nibble - 10u)));
}

std::string encode_hex(std::span<const std::byte> bytes) {
  std::string out;
  out.resize(bytes.size() * 2u);
  for (size_t idx = 0; idx < bytes.size(); ++idx) {
    const uint8_t value = static_cast<uint8_t>(bytes[idx]);
    out[idx * 2u + 0u] = hex_digit(static_cast<uint8_t>((value >> 4u) & 0x0fu));
    out[idx * 2u + 1u] = hex_digit(static_cast<uint8_t>(value & 0x0fu));
  }
  return out;
}

bool decode_hex_char(const char ch, uint8_t & value_out) {
  if (ch >= '0' && ch <= '9') {
    value_out = static_cast<uint8_t>(ch - '0');
    return true;
  }
  if (ch >= 'a' && ch <= 'f') {
    value_out = static_cast<uint8_t>(10 + ch - 'a');
    return true;
  }
  if (ch >= 'A' && ch <= 'F') {
    value_out = static_cast<uint8_t>(10 + ch - 'A');
    return true;
  }
  return false;
}

bool decode_hex(std::string_view text, std::vector<uint8_t> & out) {
  out.clear();
  if ((text.size() % 2u) != 0u) {
    return false;
  }

  out.resize(text.size() / 2u);
  for (size_t idx = 0; idx < out.size(); ++idx) {
    uint8_t high = 0u;
    uint8_t low = 0u;
    if (!decode_hex_char(text[idx * 2u + 0u], high) ||
        !decode_hex_char(text[idx * 2u + 1u], low)) {
      return false;
    }
    out[idx] = static_cast<uint8_t>((high << 4u) | low);
  }
  return true;
}

bool find_named_line_value(std::string_view text,
                           std::string_view key,
                           std::string_view & value_out) {
  size_t line_begin = 0u;
  while (line_begin <= text.size()) {
    size_t line_end = text.find('\n', line_begin);
    if (line_end == std::string_view::npos) {
      line_end = text.size();
    }
    std::string_view line = text.substr(line_begin, line_end - line_begin);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1u);
    }
    if (line.size() > key.size() && line.substr(0u, key.size()) == key &&
        line[key.size()] == '=') {
      value_out = line.substr(key.size() + 1u);
      return true;
    }
    if (line_end == text.size()) {
      break;
    }
    line_begin = line_end + 1u;
  }
  return false;
}

bool parse_i32_text(std::string_view text, int32_t & out) {
  if (text.empty()) {
    return false;
  }
  std::string buffer(text);
  char * end = nullptr;
  const long parsed = std::strtol(buffer.c_str(), &end, 10);
  if (end == buffer.c_str() || *end != '\0' || parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  out = static_cast<int32_t>(parsed);
  return true;
}

bool parse_size_text(std::string_view text, size_t & out) {
  if (text.empty()) {
    return false;
  }
  std::string buffer(text);
  char * end = nullptr;
  const unsigned long long parsed = std::strtoull(buffer.c_str(), &end, 10);
  if (end == buffer.c_str() || *end != '\0') {
    return false;
  }
  out = static_cast<size_t>(parsed);
  return true;
}

bool parse_float_text(std::string_view text, float & out) {
  if (text.empty()) {
    return false;
  }
  std::string buffer(text);
  char * end = nullptr;
  const float parsed = std::strtof(buffer.c_str(), &end);
  if (end == buffer.c_str() || *end != '\0') {
    return false;
  }
  out = parsed;
  return true;
}

template <class value_type, class parse_fn>
bool parse_csv_values(std::string_view text,
                      std::span<value_type> out,
                      int32_t & count_out,
                      parse_fn parse_value) {
  count_out = 0;
  if (text.empty()) {
    return true;
  }

  size_t token_begin = 0u;
  while (token_begin <= text.size()) {
    const size_t token_end = text.find(',', token_begin);
    const size_t safe_end = token_end == std::string_view::npos ? text.size() : token_end;
    if (count_out < 0 || static_cast<size_t>(count_out) >= out.size()) {
      return false;
    }

    value_type value{};
    if (!parse_value(text.substr(token_begin, safe_end - token_begin), value)) {
      return false;
    }
    out[static_cast<size_t>(count_out)] = value;
    count_out += 1;

    if (token_end == std::string_view::npos) {
      break;
    }
    token_begin = token_end + 1u;
  }
  return true;
}

bool write_generation_baseline_file(const std::filesystem::path & path,
                                    const emel::paritychecker::parity_options & opts,
                                    const generation_result & result) {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code create_error;
    std::filesystem::create_directories(parent, create_error);
    if (create_error) {
      return false;
    }
  }

  std::FILE * file = std::fopen(path.string().c_str(), "wb");
  if (file == nullptr) {
    return false;
  }

  const std::string prompt_hex = encode_hex(std::as_bytes(std::span<const char>(opts.text.data(), opts.text.size())));
  const std::string output_hex = encode_hex(std::as_bytes(std::span<const char>(
      result.output.data(), result.output_length)));

  std::fprintf(file, "format=%.*s\n", static_cast<int>(k_generation_baseline_format.size()),
               k_generation_baseline_format.data());
  std::fprintf(file, "contract=%.*s\n", static_cast<int>(k_generation_baseline_contract.size()),
               k_generation_baseline_contract.data());
  std::fprintf(file, "fixture=%s\n", k_generation_fixture_name);
  std::fprintf(file, "prompt_hex=%s\n", prompt_hex.c_str());
  std::fprintf(file, "max_tokens=%d\n", opts.max_tokens);
  std::fprintf(file, "tokens_generated=%d\n", result.tokens_generated);
  std::fprintf(file, "output_length=%zu\n", result.output_length);
  std::fprintf(file, "trace_token_count=%d\n", result.trace.token_count);
  std::fprintf(file, "token_ids=");
  for (int32_t idx = 0; idx < result.trace.token_count; ++idx) {
    if (idx > 0) {
      std::fputc(',', file);
    }
    std::fprintf(file, "%d", result.trace.token_ids[static_cast<size_t>(idx)]);
  }
  std::fputc('\n', file);
  std::fprintf(file, "top_score_gaps=");
  for (int32_t idx = 0; idx < result.trace.token_count; ++idx) {
    if (idx > 0) {
      std::fputc(',', file);
    }
    std::fprintf(file, "%.9g", result.trace.top_score_gaps[static_cast<size_t>(idx)]);
  }
  std::fputc('\n', file);
  std::fprintf(file, "output_hex=%s\n", output_hex.c_str());

  return std::fclose(file) == 0;
}

bool load_generation_baseline_file(const std::filesystem::path & path,
                                   generation_baseline_record & record_out) {
  std::vector<uint8_t> file_bytes;
  if (!read_file_bytes(path.string(), file_bytes)) {
    return false;
  }

  const std::string text(file_bytes.begin(), file_bytes.end());
  std::string_view value = {};

  if (!find_named_line_value(text, "format", value) || value != k_generation_baseline_format) {
    return false;
  }
  if (!find_named_line_value(text, "contract", value) || value != k_generation_baseline_contract) {
    return false;
  }
  if (!find_named_line_value(text, "fixture", value)) {
    return false;
  }
  record_out.fixture_name.assign(value.begin(), value.end());

  if (!find_named_line_value(text, "prompt_hex", value)) {
    return false;
  }
  std::vector<uint8_t> prompt_bytes;
  if (!decode_hex(value, prompt_bytes)) {
    return false;
  }
  record_out.prompt.assign(reinterpret_cast<const char *>(prompt_bytes.data()), prompt_bytes.size());

  if (!find_named_line_value(text, "max_tokens", value) ||
      !parse_i32_text(value, record_out.max_tokens)) {
    return false;
  }
  if (!find_named_line_value(text, "tokens_generated", value) ||
      !parse_i32_text(value, record_out.result.tokens_generated)) {
    return false;
  }
  if (!find_named_line_value(text, "output_length", value) ||
      !parse_size_text(value, record_out.result.output_length) ||
      record_out.result.output_length > record_out.result.output.size()) {
    return false;
  }
  if (!find_named_line_value(text, "trace_token_count", value) ||
      !parse_i32_text(value, record_out.result.trace.token_count) ||
      record_out.result.trace.token_count < 0 ||
      static_cast<size_t>(record_out.result.trace.token_count) >
          record_out.result.trace.token_ids.size()) {
    return false;
  }
  int32_t token_count = 0;
  if (!find_named_line_value(text, "token_ids", value) ||
      !parse_csv_values<int32_t>(
          value,
          std::span<int32_t>(record_out.result.trace.token_ids),
          token_count,
          [](std::string_view token_text, int32_t & token_out) {
            return parse_i32_text(token_text, token_out);
          }) ||
      token_count != record_out.result.trace.token_count) {
    return false;
  }

  int32_t gap_count = 0;
  if (!find_named_line_value(text, "top_score_gaps", value) ||
      !parse_csv_values<float>(
          value,
          std::span<float>(record_out.result.trace.top_score_gaps),
          gap_count,
          [](std::string_view gap_text, float & gap_out) {
            return parse_float_text(gap_text, gap_out);
          }) ||
      gap_count != record_out.result.trace.token_count) {
    return false;
  }

  if (!find_named_line_value(text, "output_hex", value)) {
    return false;
  }
  std::vector<uint8_t> output_bytes;
  if (!decode_hex(value, output_bytes) || output_bytes.size() != record_out.result.output_length) {
    return false;
  }
  if (!output_bytes.empty()) {
    std::memcpy(record_out.result.output.data(), output_bytes.data(), output_bytes.size());
  }

  return record_out.fixture_name == k_generation_fixture_name;
}

struct generation_attribution_bucket {
  const char * name = nullptr;
  uint64_t ns = 0u;
  uint64_t calls = 0u;
};

struct generation_attribution {
  int32_t prompt_tokens = 0;
  int32_t generated_tokens = 0;
  uint64_t total_ns = 0u;
  uint64_t prefill_ns = 0u;
  uint64_t decode_ns = 0u;

  generation_attribution_bucket embedding_lookup{"embedding_lookup"};
  generation_attribution_bucket rms_norm{"rms_norm"};
  generation_attribution_bucket qkv_matmul{"qkv_matmul"};
  generation_attribution_bucket rope{"rope"};
  generation_attribution_bucket cache_store{"cache_store"};
  generation_attribution_bucket attention{"attention"};
  generation_attribution_bucket attention_output_proj{"attention_output_proj"};
  generation_attribution_bucket residual_add{"residual_add"};
  generation_attribution_bucket ffn_gate_up_matmul{"ffn_gate_up_matmul"};
  generation_attribution_bucket swiglu{"swiglu"};
  generation_attribution_bucket ffn_down_matmul{"ffn_down_matmul"};
  generation_attribution_bucket logits_q8_prepare{"logits_q8_prepare"};
  generation_attribution_bucket logits_matmul{"logits_matmul"};
  generation_attribution_bucket logits_argmax{"logits_argmax"};
  generation_attribution_bucket output_append{"output_append"};
};

bool run_compute_logits_with_attribution(
    emel::generator::detail::native_backend & backend,
    generation_attribution & attribution);

bool run_compute_logits_preselected_argmax_with_attribution(
    emel::generator::detail::native_backend & backend,
    generation_attribution & attribution,
    int32_t & selected_index,
    float & selected_score);

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

using attribution_clock = std::chrono::steady_clock;

uint64_t elapsed_ns(const attribution_clock::time_point begin,
                    const attribution_clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

void add_bucket_sample(generation_attribution_bucket & bucket, const uint64_t ns) {
  bucket.ns += ns;
  bucket.calls += 1u;
}

template <class fn_t>
bool time_bucket_bool(generation_attribution_bucket & bucket, fn_t && fn) {
  const auto begin = attribution_clock::now();
  const bool ok = fn();
  add_bucket_sample(bucket, elapsed_ns(begin, attribution_clock::now()));
  return ok;
}

template <class fn_t>
void time_bucket_void(generation_attribution_bucket & bucket, fn_t && fn) {
  const auto begin = attribution_clock::now();
  fn();
  add_bucket_sample(bucket, elapsed_ns(begin, attribution_clock::now()));
}

uint64_t bucketed_generation_ns(const generation_attribution & attribution) {
  return attribution.embedding_lookup.ns +
      attribution.rms_norm.ns +
      attribution.qkv_matmul.ns +
      attribution.rope.ns +
      attribution.cache_store.ns +
      attribution.attention.ns +
      attribution.attention_output_proj.ns +
      attribution.residual_add.ns +
      attribution.ffn_gate_up_matmul.ns +
      attribution.swiglu.ns +
      attribution.ffn_down_matmul.ns +
      attribution.logits_q8_prepare.ns +
      attribution.logits_matmul.ns +
      attribution.logits_argmax.ns +
      attribution.output_append.ns;
}

void print_generation_attribution_bucket(const generation_attribution_bucket & bucket,
                                         const uint64_t total_ns) {
  const double pct = total_ns == 0u
      ? 0.0
      : (100.0 * static_cast<double>(bucket.ns) / static_cast<double>(total_ns));
  std::fprintf(stdout,
               "generation_attribution.bucket: name=%s ns=%" PRIu64
               " ms=%.3f pct_total=%.2f calls=%" PRIu64 "\n",
               bucket.name,
               bucket.ns,
               static_cast<double>(bucket.ns) / 1.0e6,
               pct,
               bucket.calls);
}

void dump_generation_attribution(const generation_attribution & attribution) {
  const uint64_t bucketed_ns = bucketed_generation_ns(attribution);
  const uint64_t uncategorized_ns =
      attribution.total_ns > bucketed_ns ? attribution.total_ns - bucketed_ns : 0u;
  const double prefill_pct = attribution.total_ns == 0u
      ? 0.0
      : (100.0 * static_cast<double>(attribution.prefill_ns) /
         static_cast<double>(attribution.total_ns));
  const double decode_pct = attribution.total_ns == 0u
      ? 0.0
      : (100.0 * static_cast<double>(attribution.decode_ns) /
         static_cast<double>(attribution.total_ns));

  std::fprintf(stdout,
               "generation_attribution: prompt_tokens=%d generated_tokens=%d total_ns=%" PRIu64
               " total_ms=%.3f prefill_ns=%" PRIu64 " prefill_pct=%.2f decode_ns=%" PRIu64
               " decode_pct=%.2f bucketed_ns=%" PRIu64 " uncategorized_ns=%" PRIu64 "\n",
               attribution.prompt_tokens,
               attribution.generated_tokens,
               attribution.total_ns,
               static_cast<double>(attribution.total_ns) / 1.0e6,
               attribution.prefill_ns,
               prefill_pct,
               attribution.decode_ns,
               decode_pct,
               bucketed_ns,
               uncategorized_ns);

  print_generation_attribution_bucket(attribution.embedding_lookup, attribution.total_ns);
  print_generation_attribution_bucket(attribution.rms_norm, attribution.total_ns);
  print_generation_attribution_bucket(attribution.qkv_matmul, attribution.total_ns);
  print_generation_attribution_bucket(attribution.rope, attribution.total_ns);
  print_generation_attribution_bucket(attribution.cache_store, attribution.total_ns);
  print_generation_attribution_bucket(attribution.attention, attribution.total_ns);
  print_generation_attribution_bucket(attribution.attention_output_proj, attribution.total_ns);
  print_generation_attribution_bucket(attribution.residual_add, attribution.total_ns);
  print_generation_attribution_bucket(attribution.ffn_gate_up_matmul, attribution.total_ns);
  print_generation_attribution_bucket(attribution.swiglu, attribution.total_ns);
  print_generation_attribution_bucket(attribution.ffn_down_matmul, attribution.total_ns);
  print_generation_attribution_bucket(attribution.logits_q8_prepare, attribution.total_ns);
  print_generation_attribution_bucket(attribution.logits_matmul, attribution.total_ns);
  print_generation_attribution_bucket(attribution.logits_argmax, attribution.total_ns);
  print_generation_attribution_bucket(attribution.output_append, attribution.total_ns);
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
  emel::tools::generation_formatter_contract::formatter_binding formatter_binding = {};

  generation_load_state()
      : samplers{emel::logits::sampler::fn::from<generation_load_state, sampler_select_argmax>(
            this)} {}
};

quantized_contract_summary runtime_quantized_contract_summary(
    const generation_load_state & state) {
  return quantized_contract_summary{
      .native_quantized = state.generator->generation_native_quantized_stage_count(),
      .approved_dense_f32_by_contract =
          state.generator->generation_approved_dense_f32_stage_count(),
      .disallowed_fallback = state.generator->generation_disallowed_fallback_stage_count(),
      .explicit_no_claim = state.generator->generation_explicit_no_claim_stage_count(),
  };
}

bool quantized_contract_matches(const quantized_contract_summary & lhs,
                                const quantized_contract_summary & rhs) {
  return lhs.native_quantized == rhs.native_quantized &&
         lhs.approved_dense_f32_by_contract == rhs.approved_dense_f32_by_contract &&
         lhs.disallowed_fallback == rhs.disallowed_fallback &&
         lhs.explicit_no_claim == rhs.explicit_no_claim;
}

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

  std::string formatted_prompt = {};
  if (!emel::tools::generation_formatter_contract::format_single_user_prompt(
          state.formatter_binding, opts.text, formatted_prompt)) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  const int32_t prompt_capacity =
      std::max<int32_t>(32, static_cast<int32_t>(formatted_prompt.size()) + 8);
  const int32_t decode_capacity = std::max<int32_t>(4, opts.max_tokens);
  const int32_t block_capacity = std::max<int32_t>(8, prompt_capacity + decode_capacity);

  state.generator = std::make_unique<emel::generator::sm>(
      *state.model_data,
      state.conditioner,
      state.formatter_binding.formatter_ctx,
      state.formatter_binding.format_prompt);

  reset_initialize_capture(state);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::event::initialize request{
    &state.tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
    std::span<emel::logits::sampler::fn>{},
  };
  request.preprocessor_variant = generation_preprocessor_variant(*state.model_data);
  request.encoder_variant = generation_encoder_variant(*state.model_data);
  request.add_special = false;
  request.parse_special = false;
  request.selection_mode = emel::generator::selection_mode::preselected_argmax;
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
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  emel::generator::event::generate request{
    emel::tools::generation_formatter_contract::single_user_messages(
        message_storage, opts.text),
    opts.max_tokens,
    output,
    output_length_out,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
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

  const std::string_view piece_view = piece;
  const llama_token_attr attr = llama_vocab_get_attr(backend.vocab, token);
  const bool is_byte_token = (attr & LLAMA_TOKEN_ATTR_BYTE) != 0;
  if (is_byte_token) {
    uint8_t byte_value = 0;
    const bool parsed =
        emel::text::detokenizer::action::detail::parse_plamo2_byte_token(piece_view, byte_value);
    if (!parsed || result_out.output_length + 1u > result_out.output.size()) {
      return false;
    }
    result_out.output[result_out.output_length] = static_cast<char>(byte_value);
    result_out.output_length += 1u;
    return true;
  }

  const size_t piece_len = piece_view.size();
  if (result_out.output_length + piece_len > result_out.output.size()) {
    return false;
  }
  if (piece_len > 0u) {
    std::memcpy(result_out.output.data() + result_out.output_length, piece_view.data(), piece_len);
  }
  result_out.output_length += piece_len;
  return true;
}

llama_context_ptr make_reference_context(initialize_backend & backend) {
  llama_context_params context_params = llama_context_default_params();
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  context_params.n_ctx = 0;
  const int32_t batch_capacity =
      backend.model != nullptr ? std::max(512, llama_model_n_ctx_train(backend.model.get())) : 512;
  context_params.n_batch = batch_capacity;
  context_params.n_ubatch = batch_capacity;
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
  if (emel_result.tokens_generated != reference_result.tokens_generated ||
      emel_result.output_length != reference_result.output_length) {
    return false;
  }
  if (emel_result.trace_available && reference_result.trace_available) {
    if (emel_result.trace.token_count != reference_result.trace.token_count) {
      return false;
    }
    for (int32_t idx = 0; idx < emel_result.trace.token_count; ++idx) {
      if (emel_result.trace.token_ids[static_cast<size_t>(idx)] !=
          reference_result.trace.token_ids[static_cast<size_t>(idx)]) {
        return false;
      }
    }
  }
  return
         std::string_view{emel_result.output.data(), emel_result.output_length} ==
             std::string_view{reference_result.output.data(), reference_result.output_length};
}

using native_layer_runner =
    bool (*)(emel::generator::detail::native_backend &, int32_t, int32_t);

bool run_ggml_nonflash_attn_case(std::span<const float> q_data,
                                 std::span<const float> k_data,
                                 std::span<const float> v_data,
                                 int64_t head_dim,
                                 int64_t kv_tokens,
                                 int64_t active_kv_tokens,
                                 int64_t head_count,
                                 int64_t kv_head_count,
                                 float scale,
                                 std::vector<float> & out);
bool run_emel_nonflash_f16_ggml_softmax_case(std::span<const float> q_data,
                                             std::span<const float> k_data,
                                             std::span<const float> v_data,
                                             int64_t head_dim,
                                             int64_t kv_tokens,
                                             int64_t active_kv_tokens,
                                             int64_t head_count,
                                             int64_t kv_head_count,
                                             float scale,
                                             std::vector<float> & out);
bool run_emel_prod_style_attn_case(std::span<const float> q_data,
                                   std::span<const float> k_data,
                                   std::span<const float> v_data,
                                   int64_t head_dim,
                                   int64_t kv_tokens,
                                   int64_t active_kv_tokens,
                                   int64_t head_count,
                                   int64_t kv_head_count,
                                   float scale,
                                   std::vector<float> & out);
bool run_emel_prod_style_float_value_attn_case(std::span<const float> q_data,
                                               std::span<const float> k_data,
                                               std::span<const float> v_data,
                                               int64_t head_dim,
                                               int64_t kv_tokens,
                                               int64_t active_kv_tokens,
                                               int64_t head_count,
                                               int64_t kv_head_count,
                                               float scale,
                                               std::vector<float> & out);

bool run_ggml_flash_attn_ext_case(std::span<const float> q_data,
                                  std::span<const float> k_data,
                                  std::span<const float> v_data,
                                  int64_t head_dim,
                                  int64_t kv_tokens,
                                  int64_t head_count,
                                  int64_t kv_head_count,
                                  float scale,
                                  std::vector<float> & out);
bool run_ggml_flash_attn_ext_case(std::span<const float> q_data,
                                  std::span<const uint16_t> k_data,
                                  std::span<const uint16_t> v_data,
                                  int64_t head_dim,
                                  int64_t kv_tokens,
                                  int64_t head_count,
                                  int64_t kv_head_count,
                                  float scale,
                                  std::vector<float> & out);

bool run_ggml_flash_attn_ext_masked_case(std::span<const float> q_data,
                                         std::span<const float> k_data,
                                         std::span<const float> v_data,
                                         int64_t head_dim,
                                         int64_t kv_tokens,
                                         int64_t active_kv_tokens,
                                         int64_t head_count,
                                         int64_t kv_head_count,
                                         float scale,
                                         std::vector<float> & out);
bool run_ggml_flash_attn_ext_masked_case(std::span<const float> q_data,
                                         std::span<const uint16_t> k_data,
                                         std::span<const uint16_t> v_data,
                                         int64_t head_dim,
                                         int64_t kv_tokens,
                                         int64_t active_kv_tokens,
                                         int64_t head_count,
                                         int64_t kv_head_count,
                                         float scale,
                                         std::vector<float> & out);

bool run_ggml_nonflash_attn_case(std::span<const float> q_data,
                                 std::span<const uint16_t> k_data,
                                 std::span<const uint16_t> v_data,
                                 int64_t head_dim,
                                 int64_t kv_tokens,
                                 int64_t active_kv_tokens,
                                 int64_t head_count,
                                 int64_t kv_head_count,
                                 float scale,
                                 std::vector<float> & out);

bool run_emel_nonflash_f16_ggml_softmax_case(std::span<const float> q_data,
                                             std::span<const uint16_t> k_data,
                                             std::span<const uint16_t> v_data,
                                             int64_t head_dim,
                                             int64_t kv_tokens,
                                             int64_t active_kv_tokens,
                                             int64_t head_count,
                                             int64_t kv_head_count,
                                             float scale,
                                             std::vector<float> & out);

bool run_emel_prod_style_attn_case(std::span<const float> q_data,
                                   std::span<const uint16_t> k_data,
                                   std::span<const uint16_t> v_data,
                                   int64_t head_dim,
                                   int64_t kv_tokens,
                                   int64_t active_kv_tokens,
                                   int64_t head_count,
                                   int64_t kv_head_count,
                                   float scale,
                                   std::vector<float> & out);

bool run_emel_prod_style_float_value_attn_case(std::span<const float> q_data,
                                               std::span<const uint16_t> k_data,
                                               std::span<const uint16_t> v_data,
                                               int64_t head_dim,
                                               int64_t kv_tokens,
                                               int64_t active_kv_tokens,
                                               int64_t head_count,
                                               int64_t kv_head_count,
                                               float scale,
                                               std::vector<float> & out);

inline float fp16_storage_to_fp32(const uint16_t value) {
  return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(value));
}

std::vector<float> decode_fp16_storage(std::span<const uint16_t> values) {
  std::vector<float> decoded(values.size(), 0.0f);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    decoded[idx] = fp16_storage_to_fp32(values[idx]);
  }
  return decoded;
}

inline float read_debug_value(std::span<const float> values, const size_t idx) {
  return values[idx];
}

inline float read_debug_value(std::span<const uint16_t> values, const size_t idx) {
  return fp16_storage_to_fp32(values[idx]);
}

emel::error::type run_custom_native_generate(
    const initialize_backend & backend_ref,
    const emel::model::data & model_data,
    const emel::paritychecker::parity_options & opts,
    const native_layer_runner run_layer_fn,
    generation_result & result_out) {
  if (backend_ref.vocab == nullptr || backend_ref.vocab_size <= 0 || run_layer_fn == nullptr) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(backend_ref, opts, prompt_tokens) || prompt_tokens.empty()) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return emel::error::cast(emel::generator::error::backend);
  }

  result_out = {};

  for (size_t token_index = 0; token_index < prompt_tokens.size(); ++token_index) {
    const int32_t token_id = static_cast<int32_t>(prompt_tokens[token_index]);
    const int32_t position = static_cast<int32_t>(token_index);
    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      return emel::error::cast(emel::generator::error::backend);
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_fn(backend, layer, position)) {
        return emel::error::cast(emel::generator::error::backend);
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  if (!emel::generator::detail::compute_logits(backend)) {
    return emel::error::cast(emel::generator::error::backend);
  }

  for (int32_t step = 0; step < opts.max_tokens; ++step) {
    const argmax_summary summary =
        select_argmax_from_logits(backend.bound_logits.data(), backend.n_vocab);
    const llama_token selected = static_cast<llama_token>(summary.selected_token);
    append_trace_token(
        result_out.trace, summary.selected_token, summary.best_score, summary.second_best_score);
    result_out.tokens_generated += 1;
    if (!append_reference_piece(backend_ref, selected, result_out)) {
      return emel::error::cast(emel::generator::error::backend);
    }
    if (llama_vocab_is_eog(backend_ref.vocab, selected)) {
      break;
    }

    const int32_t position =
        static_cast<int32_t>(prompt_tokens.size()) + result_out.tokens_generated - 1;
    if (!emel::generator::detail::copy_tensor_row(
            *backend.token_embedding.tensor, summary.selected_token, backend.hidden)) {
      return emel::error::cast(emel::generator::error::backend);
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_fn(backend, layer, position)) {
        return emel::error::cast(emel::generator::error::backend);
      }
    }
    backend.kv_cache_tokens = position + 1;
    if (!emel::generator::detail::compute_logits(backend)) {
      return emel::error::cast(emel::generator::error::backend);
    }
  }

  return emel::error::cast(emel::generator::error::none);
}

bool run_layer_with_flash_attribution(emel::generator::detail::native_backend & backend,
                                      const int32_t layer_index,
                                      const int32_t position,
                                      generation_attribution & attribution) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];

  if (!time_bucket_bool(attribution.rms_norm, [&] {
        return emel::generator::detail::rms_norm(
            backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm);
      })) {
    return false;
  }
  if (!time_bucket_bool(attribution.qkv_matmul, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.attention_q, backend.norm, backend.q);
      }) ||
      !time_bucket_bool(attribution.qkv_matmul, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.attention_k, backend.norm, backend.k);
      }) ||
      !time_bucket_bool(attribution.qkv_matmul, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.attention_v, backend.norm, backend.v);
      })) {
    return false;
  }

  time_bucket_void(attribution.rope, [&] {
    emel::generator::detail::apply_rope(
        backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
    emel::generator::detail::apply_rope(
        backend.k,
        backend.n_head_kv,
        backend.head_dim_kv,
        backend.n_rot,
        position,
        backend.rope_freq_base);
  });

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset = emel::generator::detail::layer_cache_offset(
      backend, layer_index, position, kv_dim);
  time_bucket_void(attribution.cache_store, [&] {
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
        backend.q_attn.data());
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
        backend.key_cache.data() + cache_offset);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
        backend.value_cache.data() + cache_offset);
    for (int32_t kv_head = 0; kv_head < backend.n_head_kv; ++kv_head) {
      const size_t src_offset =
          static_cast<size_t>(kv_head) * static_cast<size_t>(backend.head_dim_kv);
      const size_t flash_cache_offset =
          emel::generator::detail::flash_layer_cache_head_position_offset(
              backend, layer_index, kv_head, position, backend.head_dim_kv);
      emel::generator::detail::store_fp16_rounded_cache(
          std::span<const float>(
              backend.k.data() + static_cast<std::ptrdiff_t>(src_offset),
              static_cast<size_t>(backend.head_dim_kv)),
          backend.flash_key_cache.data() + flash_cache_offset);
      emel::generator::detail::store_fp16_rounded_cache(
          std::span<const float>(
              backend.v.data() + static_cast<std::ptrdiff_t>(src_offset),
              static_cast<size_t>(backend.head_dim_kv)),
          backend.flash_value_cache.data() + flash_cache_offset);
    }
  });

  if (!time_bucket_bool(attribution.attention, [&] {
        return emel::generator::detail::dispatch_flash_attention(backend, layer_index, position);
      })) {
    return false;
  }
  if (!time_bucket_bool(attribution.attention_output_proj, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.attention_output, backend.attn_ctx, backend.projected);
      })) {
    return false;
  }

  time_bucket_void(attribution.residual_add, [&] {
    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
  });

  if (!time_bucket_bool(attribution.rms_norm, [&] {
        return emel::generator::detail::rms_norm(
            backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm);
      })) {
    return false;
  }
  if (!time_bucket_bool(attribution.ffn_gate_up_matmul, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.feed_forward_gate, backend.norm, backend.gate);
      }) ||
      !time_bucket_bool(attribution.ffn_gate_up_matmul, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.feed_forward_up, backend.norm, backend.up);
      })) {
    return false;
  }

  time_bucket_void(attribution.swiglu, [&] {
    for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
      backend.ffn_hidden[idx] =
          emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
    }
  });

  if (!time_bucket_bool(attribution.ffn_down_matmul, [&] {
        return emel::generator::detail::matmul_vector(
            backend, block.feed_forward_down, backend.ffn_hidden, backend.projected);
      })) {
    return false;
  }

  time_bucket_void(attribution.residual_add, [&] {
    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
  });

  return true;
}

emel::error::type run_attributed_native_generate(
    const initialize_backend & backend_ref,
    const emel::model::data & model_data,
    const emel::paritychecker::parity_options & opts,
    generation_result & result_out,
    generation_attribution & attribution_out) {
  if (backend_ref.vocab == nullptr || backend_ref.vocab_size <= 0) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(backend_ref, opts, prompt_tokens) || prompt_tokens.empty()) {
    return emel::error::cast(emel::generator::error::invalid_request);
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return emel::error::cast(emel::generator::error::backend);
  }

  result_out = {};
  attribution_out = {};
  attribution_out.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  const auto total_begin = attribution_clock::now();
  int32_t selected_token = 0;
  float selected_score = 0.0f;

  const auto prefill_begin = attribution_clock::now();
  for (size_t token_index = 0; token_index < prompt_tokens.size(); ++token_index) {
    const int32_t token_id = static_cast<int32_t>(prompt_tokens[token_index]);
    const int32_t position = static_cast<int32_t>(token_index);
    if (!time_bucket_bool(attribution_out.embedding_lookup, [&] {
          return emel::generator::detail::copy_tensor_row(
              *backend.token_embedding.tensor, token_id, backend.hidden);
        })) {
      return emel::error::cast(emel::generator::error::backend);
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_flash_attribution(backend, layer, position, attribution_out)) {
        return emel::error::cast(emel::generator::error::backend);
      }
    }
    backend.kv_cache_tokens = position + 1;
  }
  if (!run_compute_logits_preselected_argmax_with_attribution(
          backend, attribution_out, selected_token, selected_score)) {
    return emel::error::cast(emel::generator::error::backend);
  }
  attribution_out.prefill_ns = elapsed_ns(prefill_begin, attribution_clock::now());

  for (int32_t step = 0; step < opts.max_tokens; ++step) {
    const auto decode_begin = attribution_clock::now();
    const llama_token selected = static_cast<llama_token>(selected_token);
    append_trace_token(
        result_out.trace, selected_token, selected_score, selected_score);
    result_out.tokens_generated += 1;
    if (!time_bucket_bool(attribution_out.output_append, [&] {
          return append_reference_piece(backend_ref, selected, result_out);
        })) {
      return emel::error::cast(emel::generator::error::backend);
    }
    if (llama_vocab_is_eog(backend_ref.vocab, selected)) {
      attribution_out.decode_ns += elapsed_ns(decode_begin, attribution_clock::now());
      break;
    }

    const int32_t position =
        static_cast<int32_t>(prompt_tokens.size()) + result_out.tokens_generated - 1;
    if (!time_bucket_bool(attribution_out.embedding_lookup, [&] {
          return emel::generator::detail::copy_tensor_row(
              *backend.token_embedding.tensor, selected_token, backend.hidden);
        })) {
      return emel::error::cast(emel::generator::error::backend);
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_flash_attribution(backend, layer, position, attribution_out)) {
        return emel::error::cast(emel::generator::error::backend);
      }
    }
    backend.kv_cache_tokens = position + 1;
    if (!run_compute_logits_preselected_argmax_with_attribution(
            backend, attribution_out, selected_token, selected_score)) {
      return emel::error::cast(emel::generator::error::backend);
    }
    attribution_out.decode_ns += elapsed_ns(decode_begin, attribution_clock::now());
  }

  attribution_out.generated_tokens = result_out.tokens_generated;
  attribution_out.total_ns = elapsed_ns(total_begin, attribution_clock::now());
  return emel::error::cast(emel::generator::error::none);
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
  return emel::generator::detail::run_prefill_flash(backend);
}

bool run_compute_logits_with_attribution(
    emel::generator::detail::native_backend & backend,
    generation_attribution & attribution) {
  if (!time_bucket_bool(attribution.rms_norm, [&] {
        return emel::generator::detail::rms_norm(
            backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm);
      })) {
    return false;
  }

  const bool packed_q8_logits_path =
      !backend.logits_input_q8_storage.empty() &&
      backend.output.tensor != nullptr &&
      static_cast<uint8_t>(backend.output.tensor->type) ==
          emel::kernel::detail::dtype_q6_k_x8_q8_prepared;
  if (packed_q8_logits_path) {
    return time_bucket_bool(attribution.logits_q8_prepare, [&] {
             return emel::generator::detail::quantize_vector_q8_k(
                 backend.norm, backend.logits_input_q8_storage);
           }) &&
        time_bucket_bool(attribution.logits_matmul, [&] {
          return emel::generator::detail::matmul_vector_q8_input(
              backend,
              backend.output,
              backend.logits_input_q8_storage,
              backend.n_embd,
              backend.bound_logits);
        });
  }

  return time_bucket_bool(attribution.logits_matmul, [&] {
    return emel::generator::detail::matmul_vector(
        backend, backend.output, backend.norm, backend.bound_logits);
  });
}

bool run_compute_logits_preselected_argmax_with_attribution(
    emel::generator::detail::native_backend & backend,
    generation_attribution & attribution,
    int32_t & selected_index,
    float & selected_score) {
  if (!time_bucket_bool(attribution.rms_norm, [&] {
        return emel::generator::detail::rms_norm(
            backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm);
      })) {
    return false;
  }

  const emel::generator::detail::tensor_matrix & output_matrix =
      backend.output_argmax.tensor != nullptr ? backend.output_argmax : backend.output;
  const bool packed_q8_logits_path =
      !backend.logits_input_q8_storage.empty() &&
      output_matrix.tensor != nullptr &&
      (static_cast<uint8_t>(output_matrix.tensor->type) == emel::kernel::detail::dtype_q6_k_x8 ||
       static_cast<uint8_t>(output_matrix.tensor->type) ==
           emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared ||
       static_cast<uint8_t>(output_matrix.tensor->type) ==
           emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  const bool direct_argmax_path =
      packed_q8_logits_path &&
      emel::generator::detail::preselected_argmax_direct_supported(backend);
  if (direct_argmax_path) {
    return time_bucket_bool(attribution.logits_q8_prepare, [&] {
             return emel::generator::detail::quantize_vector_q8_k(
                 backend.norm, backend.logits_input_q8_storage);
           }) &&
        time_bucket_bool(attribution.logits_matmul, [&] {
          return emel::generator::detail::matmul_vector_q8_input_argmax(
              backend,
              output_matrix,
              backend.logits_input_q8_storage,
              backend.n_embd,
              selected_index,
              selected_score);
        });
  }

  if (!run_compute_logits_with_attribution(backend, attribution)) {
    return false;
  }

  argmax_summary summary{};
  time_bucket_void(attribution.logits_argmax, [&] {
    summary = select_argmax_from_logits(backend.bound_logits.data(), backend.n_vocab);
  });
  selected_index = summary.selected_token;
  selected_score = summary.best_score;
  return true;
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
  uint32_t dtype_mask = 0xffffffffu;
  bool use_reference_q8 = false;
  bool use_scalar_quantized = false;
};

struct score_dot_probe_result {
  float first_abs = 0.0f;
  int32_t first_token = -1;
  int32_t first_layer = -1;
  int32_t first_head = -1;
  int32_t first_position = -1;
  float first_emel = 0.0f;
  float first_reference = 0.0f;
  float max_abs = 0.0f;
  int32_t max_token = -1;
  int32_t max_layer = -1;
  int32_t max_head = -1;
  int32_t max_position = -1;
  float max_emel = 0.0f;
  float max_reference = 0.0f;
};

void dump_generation_q23_stage_debug(const generation_load_state & state,
                                     const emel::paritychecker::parity_options & opts,
                                     const generation_result & emel_result,
                                     const generation_result & reference_result);
void dump_generation_prefix_timeline_debug(const generation_load_state & state,
                                           const emel::paritychecker::parity_options & opts,
                                           const generation_result & emel_result,
                                           const generation_result & reference_result);
void dump_generation_q23_timeline_debug(const generation_load_state & state,
                                        const emel::paritychecker::parity_options & opts,
                                        const generation_result & emel_result,
                                        const generation_result & reference_result);
void dump_generation_reference_q_timeline_debug(const generation_load_state & state,
                                                const emel::paritychecker::parity_options & opts,
                                                const generation_result & emel_result,
                                                const generation_result & reference_result);
void dump_generation_reference_q_stage_debug(const generation_load_state & state,
                                             const emel::paritychecker::parity_options & opts,
                                             const generation_result & emel_result,
                                             const generation_result & reference_result);

bool quantize_input_blocks_reference(std::span<const float> input,
                                     std::array<reference_block_q8_k,
                                                kernel_quant::MAX_Q8_K_BLOCKS> & blocks,
                                     uint64_t & block_count_out);

bool quantize_input_blocks(std::span<const float> input,
                           std::array<kernel_quant::block_q8_k,
                                      kernel_quant::MAX_Q8_K_BLOCKS> & blocks,
                           uint64_t & block_count_out);

void update_score_dot_probe(const emel::generator::detail::native_backend & backend,
                            const int32_t token_index,
                            const int32_t layer_index,
                            const int32_t position_limit,
                            score_dot_probe_result & result) {
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const int32_t kv_dim = backend.n_head_kv * kv_head_dim;
  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));
  for (int32_t head = 0; head < backend.n_head; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(backend.q_attn[q_offset + static_cast<size_t>(dim)]);
    }
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] =
            ggml_fp32_to_fp16(fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
      }
      float reference_score = 0.0f;
      ggml_vec_dot_f16(head_dim,
                       &reference_score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
      const float emel_score = emel::kernel::detail::dot_product_ggml_f16_scores(
          backend.q_attn.data() + static_cast<std::ptrdiff_t>(q_offset),
          backend.key_cache.data() + static_cast<std::ptrdiff_t>(cache_offset),
          static_cast<uint64_t>(head_dim));
      const float diff = std::fabs(emel_score - reference_score);
      if (diff > 0.0f && result.first_token < 0) {
        result.first_abs = diff;
        result.first_token = token_index;
        result.first_layer = layer_index;
        result.first_head = head;
        result.first_position = position;
        result.first_emel = emel_score;
        result.first_reference = reference_score;
      }
      if (diff > result.max_abs) {
        result.max_abs = diff;
        result.max_token = token_index;
        result.max_layer = layer_index;
        result.max_head = head;
        result.max_position = position;
        result.max_emel = emel_score;
        result.max_reference = reference_score;
      }
    }
  }
}

bool capture_reference_value_cache_rows(llama_context * ctx,
                                        int32_t layer_index,
                                        std::vector<float> & values_out);
bool capture_reference_key_cache_rows(llama_context * ctx,
                                      int32_t layer_index,
                                      std::vector<float> & values_out);

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
                        const uint32_t dtype_mask = 0xffffffffu,
                        const bool use_reference_q8 = false,
                        const bool use_scalar_quantized = false) {
  const uint8_t dtype_code =
      matrix.tensor != nullptr ? static_cast<uint8_t>(matrix.tensor->type) : 0xffu;
  const bool dtype_match = only_dtype != 0u
      ? (dtype_code == only_dtype)
      : (dtype_code < 32u && ((dtype_mask >> dtype_code) & 1u) != 0u);
  const bool exact_enabled = exact && dtype_match;
  if (use_reference_q8 && dtype_match) {
    return matmul_vector_reference_q8(matrix, input, output);
  }
  if (use_scalar_quantized && dtype_match) {
    return matmul_vector_scalar_quantized(matrix, input, output);
  }
  if (exact_enabled) {
    return matmul_vector_dequantized(matrix, input, output);
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
                            mode.dtype_mask,
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
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.attention_k,
          backend.norm,
          backend.k,
          mode.attention,
          mode.only_dtype,
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.attention_v,
          backend.norm,
          backend.v,
          mode.attention,
          mode.only_dtype,
          mode.dtype_mask,
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
          mode.dtype_mask,
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
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.feed_forward_up,
          backend.norm,
          backend.up,
          mode.ffn,
          mode.only_dtype,
          mode.dtype_mask,
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
          mode.dtype_mask,
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

bool run_layer_with_exact_all_scalar_attention(emel::generator::detail::native_backend & backend,
                                               const int32_t layer_index,
                                               const int32_t position) {
  const exact_matmul_mode mode{.attention = true, .ffn = true, .output = true};
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_exact_attention_scalar_attention(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position) {
  const exact_matmul_mode mode{.attention = true, .ffn = false, .output = false};
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_exact_ffn_scalar_attention(emel::generator::detail::native_backend & backend,
                                               const int32_t layer_index,
                                               const int32_t position) {
  const exact_matmul_mode mode{.attention = false, .ffn = true, .output = false};
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_exact_output_scalar_attention(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position) {
  const exact_matmul_mode mode{.attention = false, .ffn = false, .output = true};
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_reference_q236_scalar_attention(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q6_k)),
      .use_reference_q8 = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_exact_q2_scalar_attention(emel::generator::detail::native_backend & backend,
                                              const int32_t layer_index,
                                              const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_exact_q3_scalar_attention(emel::generator::detail::native_backend & backend,
                                              const int32_t layer_index,
                                              const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_exact_q6_scalar_attention(emel::generator::detail::native_backend & backend,
                                              const int32_t layer_index,
                                              const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q6_k),
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_scalar_q2_scalar_attention(emel::generator::detail::native_backend & backend,
                                               const int32_t layer_index,
                                               const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
      .use_scalar_quantized = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_scalar_q3_scalar_attention(emel::generator::detail::native_backend & backend,
                                               const int32_t layer_index,
                                               const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_scalar_quantized = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_scalar_q6_scalar_attention(emel::generator::detail::native_backend & backend,
                                               const int32_t layer_index,
                                               const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q6_k),
      .use_scalar_quantized = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_reference_q2_scalar_attention(emel::generator::detail::native_backend & backend,
                                                  const int32_t layer_index,
                                                  const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
      .use_reference_q8 = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_reference_q6_scalar_attention(emel::generator::detail::native_backend & backend,
                                                  const int32_t layer_index,
                                                  const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q6_k),
      .use_reference_q8 = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_reference_q3_scalar_attention(emel::generator::detail::native_backend & backend,
                                                  const int32_t layer_index,
                                                  const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_reference_q8 = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
}

bool run_layer_with_scalar_q236_scalar_attention(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q6_k)),
      .use_scalar_quantized = true,
  };
  return run_layer_with_matmul_mode_scalar_attention(backend, layer_index, position, mode);
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

bool dispatch_flash_attention_with_q(emel::generator::detail::native_backend & backend,
                                     const int32_t layer_index,
                                     const int32_t position,
                                     std::span<const float> q_vector) {
  auto request = emel::generator::detail::make_flash_attn_request(backend, layer_index, position);
  request.src0 = emel::generator::detail::make_src_view_3d(
      q_vector.data(),
      static_cast<uint64_t>(backend.head_dim),
      1u,
      static_cast<uint64_t>(backend.n_head));
  backend.kernel.set_kind(backend.kernel_kind);
  return backend.kernel.process_event(request);
}

bool run_layer_with_flash_request_q(emel::generator::detail::native_backend & backend,
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

  if (!dispatch_flash_attention_with_q(
          backend,
          layer_index,
          position,
          std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd))) ||
      !emel::generator::detail::matmul_vector(
          backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(
          backend, block.feed_forward_gate, backend.norm, backend.gate) ||
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

bool run_layer_with_flash_request_q_attn(emel::generator::detail::native_backend & backend,
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

  if (!dispatch_flash_attention_with_q(
          backend,
          layer_index,
          position,
          std::span<const float>(backend.q_attn.data(), static_cast<size_t>(backend.n_embd))) ||
      !emel::generator::detail::matmul_vector(
          backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(
          backend, block.feed_forward_gate, backend.norm, backend.gate) ||
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

bool run_layer_with_scalar_attention_score_dot_probe(
    emel::generator::detail::native_backend & backend,
    const int32_t token_index,
    const int32_t layer_index,
    const int32_t position,
    score_dot_probe_result & probe) {
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

  update_score_dot_probe(backend, token_index, layer_index, position + 1, probe);

  if (!emel::generator::detail::compute_attention(
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
      !emel::generator::detail::matmul_vector(
          backend, block.feed_forward_gate, backend.norm, backend.gate) ||
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

bool run_prefill_with_flash_request_q(emel::generator::detail::native_backend & backend,
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
      if (!run_layer_with_flash_request_q(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_flash_request_q_attn(emel::generator::detail::native_backend & backend,
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
      if (!run_layer_with_flash_request_q_attn(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_score_dot_probe(
    emel::generator::detail::native_backend & backend,
    std::span<const int32_t> prefix_tokens,
    score_dot_probe_result & probe) {
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

    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor,
                                                  token_id,
                                                  backend.hidden)) {
      return false;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention_score_dot_probe(
              backend, position, layer, position, probe)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

enum class reference_stage_injection : uint8_t {
  layer0_kqv_out,
  layer0_attn_out,
  layer0_ffn_norm,
  layer0_ffn_out,
  layer0_l_out,
  layer1_kqv_out,
  layer1_attn_out,
  layer1_attn_norm,
  layer1_ffn_inp,
  layer1_ffn_norm,
  layer1_ffn_out,
  layer1_l_out,
};

enum class attention_projection_override : uint8_t {
  native,
  exact,
  reference_q8,
};

struct reference_stage_capture_set {
  std::span<const float> layer0_kqv_out = {};
  std::span<const float> layer0_attn_out = {};
  std::span<const float> layer0_ffn_norm = {};
  std::span<const float> layer0_ffn_out = {};
  std::span<const float> layer0_l_out = {};
  std::span<const float> layer1_kqv_out = {};
  std::span<const float> layer1_attn_out = {};
  std::span<const float> layer1_attn_norm = {};
  std::span<const float> layer1_ffn_inp = {};
  std::span<const float> layer1_ffn_norm = {};
  std::span<const float> layer1_ffn_out = {};
  std::span<const float> layer1_l_out = {};
};

bool reference_stage_targets_layer(const reference_stage_injection injection_stage,
                                   const int32_t layer_index) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_kqv_out:
    case reference_stage_injection::layer0_attn_out:
    case reference_stage_injection::layer0_ffn_norm:
    case reference_stage_injection::layer0_ffn_out:
    case reference_stage_injection::layer0_l_out:
      return layer_index == 0;
    case reference_stage_injection::layer1_kqv_out:
    case reference_stage_injection::layer1_attn_out:
    case reference_stage_injection::layer1_attn_norm:
    case reference_stage_injection::layer1_ffn_inp:
    case reference_stage_injection::layer1_ffn_norm:
    case reference_stage_injection::layer1_ffn_out:
    case reference_stage_injection::layer1_l_out:
      return layer_index == 1;
  }
  return false;
}

bool reference_stage_is_kqv_out(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_kqv_out:
    case reference_stage_injection::layer1_kqv_out:
      return true;
    default:
      return false;
  }
}

bool reference_stage_is_attn_out(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_attn_out:
    case reference_stage_injection::layer1_attn_out:
      return true;
    default:
      return false;
  }
}

bool reference_stage_is_attn_norm(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer1_attn_norm:
      return true;
    default:
      return false;
  }
}

bool reference_stage_is_ffn_inp(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer1_ffn_inp:
      return true;
    default:
      return false;
  }
}

bool reference_stage_is_ffn_norm(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_ffn_norm:
    case reference_stage_injection::layer1_ffn_norm:
      return true;
    default:
      return false;
  }
}

bool reference_stage_is_ffn_out(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_ffn_out:
    case reference_stage_injection::layer1_ffn_out:
      return true;
    default:
      return false;
  }
}

bool reference_stage_is_l_out(const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_l_out:
    case reference_stage_injection::layer1_l_out:
      return true;
    default:
      return false;
  }
}

std::span<const float> reference_stage_rows(const reference_stage_capture_set & captures,
                                            const reference_stage_injection injection_stage) {
  switch (injection_stage) {
    case reference_stage_injection::layer0_kqv_out:
      return captures.layer0_kqv_out;
    case reference_stage_injection::layer0_attn_out:
      return captures.layer0_attn_out;
    case reference_stage_injection::layer0_ffn_norm:
      return captures.layer0_ffn_norm;
    case reference_stage_injection::layer0_ffn_out:
      return captures.layer0_ffn_out;
    case reference_stage_injection::layer0_l_out:
      return captures.layer0_l_out;
    case reference_stage_injection::layer1_kqv_out:
      return captures.layer1_kqv_out;
    case reference_stage_injection::layer1_attn_out:
      return captures.layer1_attn_out;
    case reference_stage_injection::layer1_attn_norm:
      return captures.layer1_attn_norm;
    case reference_stage_injection::layer1_ffn_inp:
      return captures.layer1_ffn_inp;
    case reference_stage_injection::layer1_ffn_norm:
      return captures.layer1_ffn_norm;
    case reference_stage_injection::layer1_ffn_out:
      return captures.layer1_ffn_out;
    case reference_stage_injection::layer1_l_out:
      return captures.layer1_l_out;
  }
  return {};
}

std::span<const float> reference_l_out_rows(const reference_stage_capture_set & captures,
                                            const int32_t layer_index) {
  switch (layer_index) {
    case 0:
      return captures.layer0_l_out;
    case 1:
      return captures.layer1_l_out;
    default:
      return {};
  }
}

bool run_layer_with_scalar_attention_reference_stage(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position,
    const reference_stage_injection injection_stage,
    const reference_stage_capture_set & captures,
    const attention_projection_override projection_override =
        attention_projection_override::native,
    const exact_matmul_mode mode = {}) {
  auto capture_row = [](std::span<const float> values,
                        const size_t row_width,
                        const int32_t row_index) {
    if (row_width == 0 || row_index < 0 || values.empty() || values.size() % row_width != 0) {
      return std::span<const float>{};
    }
    const size_t row_count = values.size() / row_width;
    const size_t row = static_cast<size_t>(row_index);
    if (row >= row_count) {
      return std::span<const float>{};
    }
    return values.subspan(row * row_width, row_width);
  };

  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  const bool inject_stage_here = reference_stage_targets_layer(injection_stage, layer_index);
  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  if (inject_stage_here && reference_stage_is_attn_norm(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.norm.size(), position);
    if (reference_row.size() != backend.norm.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.norm.begin());
  }

  if (!matmul_vector_mode(backend,
                          block.attention_q,
                          backend.norm,
                          backend.q,
                          mode.attention,
                          mode.only_dtype,
                          mode.dtype_mask,
                          mode.use_reference_q8,
                          mode.use_scalar_quantized) ||
      !matmul_vector_mode(backend,
                          block.attention_k,
                          backend.norm,
                          backend.k,
                          mode.attention,
                          mode.only_dtype,
                          mode.dtype_mask,
                          mode.use_reference_q8,
                          mode.use_scalar_quantized) ||
      !matmul_vector_mode(backend,
                          block.attention_v,
                          backend.norm,
                          backend.v,
                          mode.attention,
                          mode.only_dtype,
                          mode.dtype_mask,
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
          backend, layer_index, position + 1, backend.q_attn)) {
    return false;
  }

  if (inject_stage_here && reference_stage_is_kqv_out(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.attn_ctx.size(), position);
    if (reference_row.size() != backend.attn_ctx.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.attn_ctx.begin());
  }

  if (inject_stage_here && reference_stage_is_kqv_out(injection_stage) &&
      projection_override != attention_projection_override::native) {
    const exact_matmul_mode exact_attention_only{
        .attention = true, .ffn = false, .output = false};
    const exact_matmul_mode reference_q8_attention_only{
        .attention = true, .ffn = false, .output = false, .use_reference_q8 = true};
    const exact_matmul_mode & projection_mode =
        projection_override == attention_projection_override::exact
            ? exact_attention_only
            : reference_q8_attention_only;
    if (!matmul_vector_mode(backend,
                            block.attention_output,
                            backend.attn_ctx,
                            backend.projected,
                            projection_mode.attention,
                            projection_mode.only_dtype,
                            projection_mode.dtype_mask,
                            projection_mode.use_reference_q8,
                            projection_mode.use_scalar_quantized)) {
      return false;
    }
  } else if (!matmul_vector_mode(backend,
                                 block.attention_output,
                                 backend.attn_ctx,
                                 backend.projected,
                                 mode.attention,
                                 mode.only_dtype,
                                 mode.dtype_mask,
                                 mode.use_reference_q8,
                                 mode.use_scalar_quantized)) {
    return false;
  }

  if (inject_stage_here && reference_stage_is_attn_out(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.projected.size(), position);
    if (reference_row.size() != backend.projected.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.projected.begin());
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (inject_stage_here && reference_stage_is_ffn_inp(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.hidden.size(), position);
    if (reference_row.size() != backend.hidden.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.hidden.begin());
  }

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector_mode(backend,
                          block.feed_forward_gate,
                          backend.norm,
                          backend.gate,
                          mode.ffn,
                          mode.only_dtype,
                          mode.dtype_mask,
                          mode.use_reference_q8,
                          mode.use_scalar_quantized) ||
      !matmul_vector_mode(backend,
                          block.feed_forward_up,
                          backend.norm,
                          backend.up,
                          mode.ffn,
                          mode.only_dtype,
                          mode.dtype_mask,
                          mode.use_reference_q8,
                          mode.use_scalar_quantized)) {
    return false;
  }

  if (inject_stage_here && reference_stage_is_ffn_norm(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.norm.size(), position);
    if (reference_row.size() != backend.norm.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.norm.begin());
    if (!matmul_vector_mode(backend,
                            block.feed_forward_gate,
                            backend.norm,
                            backend.gate,
                            mode.ffn,
                            mode.only_dtype,
                            mode.dtype_mask,
                            mode.use_reference_q8,
                            mode.use_scalar_quantized) ||
        !matmul_vector_mode(backend,
                            block.feed_forward_up,
                            backend.norm,
                            backend.up,
                            mode.ffn,
                            mode.only_dtype,
                            mode.dtype_mask,
                            mode.use_reference_q8,
                            mode.use_scalar_quantized)) {
      return false;
    }
  }

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
  }

  if (!matmul_vector_mode(backend,
                          block.feed_forward_down,
                          backend.ffn_hidden,
                          backend.projected,
                          mode.ffn,
                          mode.only_dtype,
                          mode.dtype_mask,
                          mode.use_reference_q8,
                          mode.use_scalar_quantized)) {
    return false;
  }

  if (inject_stage_here && reference_stage_is_ffn_out(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.projected.size(), position);
    if (reference_row.size() != backend.projected.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.projected.begin());
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (inject_stage_here && reference_stage_is_l_out(injection_stage)) {
    const std::span<const float> reference_row =
        capture_row(reference_stage_rows(captures, injection_stage), backend.hidden.size(), position);
    if (reference_row.size() != backend.hidden.size()) {
      return false;
    }
    std::copy(reference_row.begin(), reference_row.end(), backend.hidden.begin());
  }

  return true;
}

bool run_prefill_with_scalar_attention_reference_stage(
    emel::generator::detail::native_backend & backend,
    std::span<const int32_t> prefix_tokens,
    const reference_stage_injection injection_stage,
    const reference_stage_capture_set & captures,
    const attention_projection_override projection_override =
        attention_projection_override::native,
    const exact_matmul_mode mode = {}) {
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
      if (!run_layer_with_scalar_attention_reference_stage(
              backend,
              layer,
              position,
              injection_stage,
              captures,
              projection_override,
              mode)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return compute_logits_with_matmul_mode(backend, mode);
}

struct reference_stage_replay_diff {
  int32_t first_target_l_out_mismatch = -1;
  float max_abs = 0.0f;
};

reference_stage_replay_diff replay_reference_stage_l_out_diff(
    emel::generator::detail::native_backend & backend,
    std::span<const int32_t> prefix_tokens,
    const int32_t target_layer,
    const reference_stage_injection injection_stage,
    const reference_stage_capture_set & captures,
    const attention_projection_override projection_override =
        attention_projection_override::native,
    const exact_matmul_mode mode = {}) {
  reference_stage_replay_diff result = {};
  if (prefix_tokens.empty()) {
    return result;
  }

  auto capture_row = [](std::span<const float> values,
                        const size_t row_width,
                        const int32_t row_index) {
    if (row_width == 0 || row_index < 0 || values.empty() || values.size() % row_width != 0) {
      return std::span<const float>{};
    }
    const size_t row_count = values.size() / row_width;
    const size_t row = static_cast<size_t>(row_index);
    if (row >= row_count) {
      return std::span<const float>{};
    }
    return values.subspan(row * row_width, row_width);
  };

  backend.kv_cache_tokens = 0;
  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (token_id < 0 ||
        token_id >= backend.token_embedding.rows ||
        position < 0 ||
        position >= backend.n_ctx ||
        !emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor,
                                                  token_id,
                                                  backend.hidden)) {
      result.first_target_l_out_mismatch = -2;
      return result;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention_reference_stage(
              backend, layer, position, injection_stage, captures, projection_override, mode)) {
        result.first_target_l_out_mismatch = -2;
        return result;
      }
      if (layer == target_layer) {
        const std::span<const float> reference_row =
            capture_row(reference_l_out_rows(captures, target_layer), backend.hidden.size(), position);
        if (reference_row.size() != backend.hidden.size()) {
          result.first_target_l_out_mismatch = -2;
          return result;
        }
        float diff = 0.0f;
        for (size_t idx = 0; idx < backend.hidden.size(); ++idx) {
          diff = std::max(diff, std::fabs(backend.hidden[idx] - reference_row[idx]));
        }
        result.max_abs = std::max(result.max_abs, diff);
        if (result.first_target_l_out_mismatch < 0 && diff > 1.0e-6f) {
          result.first_target_l_out_mismatch = position;
        }
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return result;
}

bool inject_reference_cache_row(emel::generator::detail::native_backend & backend,
                                llama_context * reference_ctx,
                                const int32_t layer_index,
                                const int32_t position,
                                const bool inject_key_cache,
                                const bool inject_value_cache) {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t row_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim);
  const size_t row_width = static_cast<size_t>(kv_dim);

  if (inject_key_cache) {
    std::vector<float> reference_key_cache_rows;
    if (!capture_reference_key_cache_rows(reference_ctx, layer_index, reference_key_cache_rows) ||
        reference_key_cache_rows.size() != static_cast<size_t>(position + 1) * row_width) {
      return false;
    }
    const auto reference_row = std::span<const float>(reference_key_cache_rows)
                                   .subspan(static_cast<size_t>(position) * row_width, row_width);
    std::copy(reference_row.begin(), reference_row.end(), backend.key_cache.begin() + row_offset);
  }

  if (inject_value_cache) {
    std::vector<float> reference_value_cache_rows;
    if (!capture_reference_value_cache_rows(reference_ctx, layer_index, reference_value_cache_rows) ||
        reference_value_cache_rows.size() != static_cast<size_t>(position + 1) * row_width) {
      return false;
    }
    const auto reference_row = std::span<const float>(reference_value_cache_rows)
                                   .subspan(static_cast<size_t>(position) * row_width, row_width);
    std::copy(reference_row.begin(),
              reference_row.end(),
              backend.value_cache.begin() + row_offset);
  }

  return true;
}

bool run_layer_with_scalar_attention_reference_cache(
    emel::generator::detail::native_backend & backend,
    llama_context * reference_ctx,
    const int32_t layer_index,
    const int32_t position,
    const bool inject_key_cache,
    const bool inject_value_cache) {
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

  if (!inject_reference_cache_row(
          backend, reference_ctx, layer_index, position, inject_key_cache, inject_value_cache) ||
      !emel::generator::detail::compute_attention(
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
      !emel::generator::detail::matmul_vector(
          backend, block.feed_forward_gate, backend.norm, backend.gate) ||
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

bool run_prefill_with_scalar_attention_reference_cache(
    emel::generator::detail::native_backend & backend,
    initialize_backend & reference_backend,
    std::span<const int32_t> prefix_tokens,
    const bool inject_key_cache,
    const bool inject_value_cache) {
  if (prefix_tokens.empty()) {
    return false;
  }

  llama_context_ptr reference_ctx = make_reference_context(reference_backend);
  if (reference_ctx == nullptr) {
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

    llama_token reference_token = static_cast<llama_token>(token_id);
    llama_batch decode_batch = llama_batch_get_one(&reference_token, 1);
    if (llama_decode(reference_ctx.get(), decode_batch) != 0) {
      return false;
    }

    if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      return false;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention_reference_cache(
              backend,
              reference_ctx.get(),
              layer,
              position,
              inject_key_cache,
              inject_value_cache)) {
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
                 fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
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
            fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]));
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

bool compute_attention_without_weight_rounding(
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
                 fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
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
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool compute_attention_with_ggml_nonflash_f16(
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

  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> value_f16(static_cast<size_t>(position_limit));
  std::vector<ggml_fp16_t> weight_f16(static_cast<size_t>(position_limit));

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(q_vector[q_offset + static_cast<size_t>(dim)]);
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
      }
      float score = 0.0f;
      ggml_vec_dot_f16(head_dim,
                       &score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
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
      weight_f16[static_cast<size_t>(position)] = ggml_fp32_to_fp16(weight);
    }

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      for (int32_t position = 0; position < position_limit; ++position) {
        const size_t cache_offset =
            emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
            kv_offset;
        value_f16[static_cast<size_t>(position)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]));
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

bool compute_attention_with_ggml_f16_scores(
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

  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(q_vector[q_offset + static_cast<size_t>(dim)]);
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
      }
      float score = 0.0f;
      ggml_vec_dot_f16(head_dim,
                       &score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
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
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool compute_attention_with_double_softmax_sum(
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

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
                              q_vector.data() + static_cast<std::ptrdiff_t>(q_offset),
                              backend.key_cache.data() + static_cast<std::ptrdiff_t>(cache_offset),
                              static_cast<uint64_t>(head_dim)) *
          inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    double score_sum = 0.0;
    for (int32_t position = 0; position < position_limit; ++position) {
      const float prob = std::exp(backend.attn_scores[static_cast<size_t>(position)] - max_score);
      backend.attn_probs[static_cast<size_t>(position)] = prob;
      score_sum += static_cast<double>(prob);
    }

    const float inv_score_sum =
        score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight = backend.attn_probs[static_cast<size_t>(position)] * inv_score_sum;
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool compute_attention_with_ggml_softmax(
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

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
                              q_vector.data() + static_cast<std::ptrdiff_t>(q_offset),
                              backend.key_cache.data() + static_cast<std::ptrdiff_t>(cache_offset),
                              static_cast<uint64_t>(head_dim)) *
          inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    const reference_ggml_float score_sum = ggml_vec_soft_max_f32(
        position_limit, backend.attn_probs.data(), backend.attn_scores.data(), max_score);
    const float inv_score_sum =
        score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int32_t position = 0; position < position_limit; ++position) {
      backend.attn_probs[static_cast<size_t>(position)] *= inv_score_sum;
    }

    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight = backend.attn_probs[static_cast<size_t>(position)];
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool compute_attention_with_ggml_f16_scores_ggml_softmax(
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

  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(q_vector[q_offset + static_cast<size_t>(dim)]);
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
      }
      float score = 0.0f;
      ggml_vec_dot_f16(head_dim,
                       &score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
      score *= inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    const reference_ggml_float score_sum = ggml_vec_soft_max_f32(
        position_limit, backend.attn_probs.data(), backend.attn_scores.data(), max_score);
    const float inv_score_sum =
        score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight =
          backend.attn_probs[static_cast<size_t>(position)] * inv_score_sum;
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool compute_attention_with_ggml_nonflash_f16_ggml_softmax(
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

  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> value_f16(static_cast<size_t>(position_limit));
  std::vector<ggml_fp16_t> weight_f16(static_cast<size_t>(position_limit));

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(q_vector[q_offset + static_cast<size_t>(dim)]);
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
      }
      float score = 0.0f;
      ggml_vec_dot_f16(head_dim,
                       &score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
      score *= inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    const reference_ggml_float score_sum = ggml_vec_soft_max_f32(
        position_limit, backend.attn_probs.data(), backend.attn_scores.data(), max_score);
    const float inv_score_sum =
        score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight =
          backend.attn_probs[static_cast<size_t>(position)] * inv_score_sum;
      weight_f16[static_cast<size_t>(position)] = ggml_fp32_to_fp16(weight);
    }

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      for (int32_t position = 0; position < position_limit; ++position) {
        const size_t cache_offset =
            emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
            kv_offset;
        value_f16[static_cast<size_t>(position)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]));
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

bool compute_attention_with_rounded_weight_scalar(
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
                 fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
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
      const float weight = kernel_quant::fp16_to_fp32(
          kernel_quant::fp32_to_fp16(
              backend.attn_probs[static_cast<size_t>(position)] / score_sum));
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool compute_attention_with_online_f32(
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

  std::vector<float> acc(static_cast<size_t>(head_dim), 0.0f);
  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    std::fill(acc.begin(), acc.end(), 0.0f);
    float max_score = -std::numeric_limits<float>::infinity();
    float score_sum = 0.0f;

    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      float score = 0.0f;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        score += q_vector[q_offset + static_cast<size_t>(dim)] *
                 fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
      }
      score *= inv_scale;

      const float prior_max = max_score;
      float scale_acc = 1.0f;
      float weight = 1.0f;
      if (score > max_score) {
        max_score = score;
        scale_acc = std::exp(prior_max - max_score);
        for (float & value : acc) {
          value *= scale_acc;
        }
      } else {
        weight = std::exp(score - max_score);
      }

      for (int32_t dim = 0; dim < head_dim; ++dim) {
        acc[static_cast<size_t>(dim)] +=
            fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]) * weight;
      }
      score_sum = score_sum * scale_acc + weight;
    }

    const float inv_score_sum = score_sum == 0.0f ? 0.0f : 1.0f / score_sum;
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      backend.attn_ctx[q_offset + static_cast<size_t>(dim)] =
          acc[static_cast<size_t>(dim)] * inv_score_sum;
    }
  }

  return true;
}

bool compute_attention_with_ggml_flash_ext_helper(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t layer_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, 0, kv_dim);
  std::vector<float> attn_ctx;
  if (!run_ggml_flash_attn_ext_case(
          q_vector,
          std::span<const uint16_t>(backend.key_cache.data() + layer_offset,
                                 static_cast<size_t>(position_limit * kv_dim)),
          std::span<const uint16_t>(backend.value_cache.data() + layer_offset,
                                 static_cast<size_t>(position_limit * kv_dim)),
          static_cast<int64_t>(backend.head_dim),
          static_cast<int64_t>(position_limit),
          static_cast<int64_t>(backend.n_head),
          static_cast<int64_t>(backend.n_head_kv),
          1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
          attn_ctx)) {
    return false;
  }
  if (attn_ctx.size() != backend.attn_ctx.size()) {
    return false;
  }
  std::copy(attn_ctx.begin(), attn_ctx.end(), backend.attn_ctx.begin());
  return true;
}

bool compute_attention_with_ggml_flash_ext_masked_helper(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t layer_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, 0, kv_dim);
  std::vector<float> attn_ctx;
  if (!run_ggml_flash_attn_ext_masked_case(
          q_vector,
          std::span<const uint16_t>(backend.key_cache.data() + layer_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          std::span<const uint16_t>(backend.value_cache.data() + layer_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          static_cast<int64_t>(backend.head_dim),
          static_cast<int64_t>(backend.n_ctx),
          static_cast<int64_t>(position_limit),
          static_cast<int64_t>(backend.n_head),
          static_cast<int64_t>(backend.n_head_kv),
          1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
          attn_ctx)) {
    return false;
  }
  if (attn_ctx.size() != backend.attn_ctx.size()) {
    return false;
  }
  std::copy(attn_ctx.begin(), attn_ctx.end(), backend.attn_ctx.begin());
  return true;
}

inline float round_scalar_to_fp16(const float value) {
  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

inline void scale_fp16_buffer(std::span<ggml_fp16_t> values, const float scale) {
  const float rounded_scale = round_scalar_to_fp16(scale);
  for (ggml_fp16_t & value : values) {
    value = ggml_fp32_to_fp16(ggml_fp16_to_fp32(value) * rounded_scale);
  }
}

inline void mad_fp16_buffer(std::span<ggml_fp16_t> accum,
                            std::span<const ggml_fp16_t> values,
                            const float scale) {
  if (accum.size() != values.size()) {
    return;
  }
  const float rounded_scale = round_scalar_to_fp16(scale);
  for (size_t idx = 0; idx < accum.size(); ++idx) {
    const float sum =
        ggml_fp16_to_fp32(accum[idx]) + ggml_fp16_to_fp32(values[idx]) * rounded_scale;
    accum[idx] = ggml_fp32_to_fp16(sum);
  }
}

bool compute_attention_with_ggml_online_f16(
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

  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> v_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> acc_f16(static_cast<size_t>(head_dim));

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(q_vector[q_offset + static_cast<size_t>(dim)]);
    }
    std::fill(acc_f16.begin(), acc_f16.end(), ggml_fp32_to_fp16(0.0f));

    float score_sum = 0.0f;
    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
        v_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]));
      }

      float score = 0.0f;
      ggml_vec_dot_f16(head_dim,
                       &score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
      score *= inv_scale;

      const float prior_max = max_score;
      float scale_acc = 1.0f;
      float weight = 1.0f;
      if (score > max_score) {
        max_score = score;
        scale_acc = std::exp(prior_max - max_score);
        scale_fp16_buffer(acc_f16, scale_acc);
      } else {
        weight = std::exp(score - max_score);
      }

      mad_fp16_buffer(acc_f16, v_f16, weight);
      score_sum = score_sum * scale_acc + weight;
    }

    const float inv_score_sum = score_sum == 0.0f ? 0.0f : 1.0f / score_sum;
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      backend.attn_ctx[q_offset + static_cast<size_t>(dim)] =
          ggml_fp16_to_fp32(acc_f16[static_cast<size_t>(dim)]) * inv_score_sum;
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

bool run_layer_with_scalar_attention_no_weight_rounding(
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

  if (!compute_attention_without_weight_rounding(
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

bool run_layer_with_scalar_attention_rounded_weight_scalar(
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

  if (!compute_attention_with_rounded_weight_scalar(
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

bool run_layer_with_scalar_attention_ggml_online_f16(
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

  if (!compute_attention_with_ggml_online_f16(
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

bool run_layer_with_scalar_attention_online_f32(
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

  if (!compute_attention_with_online_f32(
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

bool run_layer_with_scalar_attention_double_softmax_sum(
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

  if (!compute_attention_with_double_softmax_sum(
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

bool run_layer_with_scalar_attention_ggml_softmax(
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

  if (!compute_attention_with_ggml_softmax(
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

bool run_layer_with_scalar_attention_ggml_flash_ext(
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

  if (!compute_attention_with_ggml_flash_ext_helper(
          backend,
          layer_index,
          position + 1,
          std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd))) ||
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

bool run_layer_with_scalar_attention_ggml_flash_ext_masked(
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

  if (!compute_attention_with_ggml_flash_ext_masked_helper(
          backend,
          layer_index,
          position + 1,
          std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd))) ||
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

bool run_prefill_with_scalar_attention_no_weight_rounding(
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
      if (!run_layer_with_scalar_attention_no_weight_rounding(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_rounded_weight_scalar(
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
      if (!run_layer_with_scalar_attention_rounded_weight_scalar(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_ggml_online_f16(
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
      if (!run_layer_with_scalar_attention_ggml_online_f16(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_online_f32(
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
      if (!run_layer_with_scalar_attention_online_f32(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_ggml_flash_ext(
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
      if (!run_layer_with_scalar_attention_ggml_flash_ext(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_double_softmax_sum(
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
      if (!run_layer_with_scalar_attention_double_softmax_sum(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_ggml_softmax(
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
      if (!run_layer_with_scalar_attention_ggml_softmax(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_layer_with_scalar_attention_full_q_no_weight_rounding(
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

  if (!compute_attention_without_weight_rounding(
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

bool run_layer_with_scalar_attention_ggml_nonflash_f16(
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

  if (!compute_attention_with_ggml_nonflash_f16(
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

bool run_prefill_with_scalar_attention_ggml_nonflash_f16(
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
      if (!run_layer_with_scalar_attention_ggml_nonflash_f16(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_layer_with_scalar_attention_ggml_f16_scores(
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

  if (!compute_attention_with_ggml_f16_scores(
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

bool run_prefill_with_scalar_attention_ggml_f16_scores(
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
      if (!run_layer_with_scalar_attention_ggml_f16_scores(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_layer_with_scalar_attention_ggml_f16_scores_ggml_softmax(
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

  if (!compute_attention_with_ggml_f16_scores_ggml_softmax(
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

bool run_prefill_with_scalar_attention_ggml_f16_scores_ggml_softmax(
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
      if (!run_layer_with_scalar_attention_ggml_f16_scores_ggml_softmax(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_layer_with_scalar_attention_ggml_nonflash_f16_ggml_softmax(
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

  if (!compute_attention_with_ggml_nonflash_f16_ggml_softmax(
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

bool run_prefill_with_scalar_attention_ggml_nonflash_f16_ggml_softmax(
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
      if (!run_layer_with_scalar_attention_ggml_nonflash_f16_ggml_softmax(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool compute_attention_with_emel_prod_style(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, 0, kv_dim);
  std::vector<float> attn_ctx;
  if (!run_emel_prod_style_attn_case(
          q_vector,
          std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          static_cast<int64_t>(backend.head_dim),
          static_cast<int64_t>(backend.n_ctx),
          static_cast<int64_t>(position_limit),
          static_cast<int64_t>(backend.n_head),
          static_cast<int64_t>(backend.n_head_kv),
          1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
          attn_ctx)) {
    return false;
  }
  if (attn_ctx.size() != backend.attn_ctx.size()) {
    return false;
  }
  std::copy(attn_ctx.begin(), attn_ctx.end(), backend.attn_ctx.begin());
  return true;
}

bool compute_attention_with_emel_prod_style_float_value(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, 0, kv_dim);
  std::vector<float> attn_ctx;
  if (!run_emel_prod_style_float_value_attn_case(
          q_vector,
          std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          static_cast<int64_t>(backend.head_dim),
          static_cast<int64_t>(backend.n_ctx),
          static_cast<int64_t>(position_limit),
          static_cast<int64_t>(backend.n_head),
          static_cast<int64_t>(backend.n_head_kv),
          1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
          attn_ctx)) {
    return false;
  }
  if (attn_ctx.size() != backend.attn_ctx.size()) {
    return false;
  }
  std::copy(attn_ctx.begin(), attn_ctx.end(), backend.attn_ctx.begin());
  return true;
}

bool run_layer_with_scalar_attention_emel_prod_style(
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

  if (!compute_attention_with_emel_prod_style(
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

bool run_layer_with_scalar_attention_emel_prod_style_float_value(
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

  if (!compute_attention_with_emel_prod_style_float_value(
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

bool run_layer_with_scalar_attention_emel_prod_style_float_value_reference_q2(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position) {
  const exact_matmul_mode mode{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
      .use_reference_q8 = true,
  };

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
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.attention_k,
          backend.norm,
          backend.k,
          mode.attention,
          mode.only_dtype,
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.attention_v,
          backend.norm,
          backend.v,
          mode.attention,
          mode.only_dtype,
          mode.dtype_mask,
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

  if (!compute_attention_with_emel_prod_style_float_value(
          backend, layer_index, position + 1, backend.q_attn) ||
      !matmul_vector_mode(
          backend,
          block.attention_output,
          backend.attn_ctx,
          backend.projected,
          mode.attention,
          mode.only_dtype,
          mode.dtype_mask,
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
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized) ||
      !matmul_vector_mode(
          backend,
          block.feed_forward_up,
          backend.norm,
          backend.up,
          mode.ffn,
          mode.only_dtype,
          mode.dtype_mask,
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
          mode.dtype_mask,
          mode.use_reference_q8,
          mode.use_scalar_quantized)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

bool run_prefill_with_scalar_attention_emel_prod_style(
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
      if (!run_layer_with_scalar_attention_emel_prod_style(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_emel_prod_style_float_value(
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
      if (!run_layer_with_scalar_attention_emel_prod_style_float_value(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool compute_attention_with_ggml_nonflash_exact_masked(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, 0, kv_dim);
  std::vector<float> attn_ctx;
  if (!run_ggml_nonflash_attn_case(
          q_vector,
          std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                 static_cast<size_t>(backend.n_ctx * kv_dim)),
          static_cast<int64_t>(backend.head_dim),
          static_cast<int64_t>(backend.n_ctx),
          static_cast<int64_t>(position_limit),
          static_cast<int64_t>(backend.n_head),
          static_cast<int64_t>(backend.n_head_kv),
          1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
          attn_ctx)) {
    return false;
  }
  if (attn_ctx.size() != backend.attn_ctx.size()) {
    return false;
  }
  std::copy(attn_ctx.begin(), attn_ctx.end(), backend.attn_ctx.begin());
  return true;
}

bool compute_attention_with_ggml_nonflash_exact_scores_prod_value(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector);

bool run_layer_with_scalar_attention_ggml_nonflash_exact_masked(
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

  if (!compute_attention_with_ggml_nonflash_exact_masked(
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

bool run_layer_with_scalar_attention_ggml_nonflash_exact_scores_prod_value(
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

  if (!compute_attention_with_ggml_nonflash_exact_scores_prod_value(
          backend, layer_index, position + 1, backend.q) ||
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

bool run_layer_with_scalar_attention_ggml_nonflash_exact_masked_full_q(
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

  if (!compute_attention_with_ggml_nonflash_exact_masked(
          backend, layer_index, position + 1, backend.q) ||
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

bool run_layer_with_scalar_attention_full_q(emel::generator::detail::native_backend & backend,
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

bool run_prefill_with_scalar_attention_full_q(emel::generator::detail::native_backend & backend,
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
      if (!run_layer_with_scalar_attention_full_q(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_prefill_with_scalar_attention_full_q_no_weight_rounding(
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
      if (!run_layer_with_scalar_attention_full_q_no_weight_rounding(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_layer_with_scalar_attention_full_q_rounded_weight(
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

  if (!compute_attention_with_rounded_weight_scalar(
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

bool run_prefill_with_scalar_attention_full_q_rounded_weight(
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
      if (!run_layer_with_scalar_attention_full_q_rounded_weight(
              backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return emel::generator::detail::compute_logits(backend);
}

bool run_layer_with_scalar_attention_full_q_ggml_f16_value_contraction(
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

bool run_prefill_with_scalar_attention_full_q_ggml_f16_value_contraction(
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
      if (!run_layer_with_scalar_attention_full_q_ggml_f16_value_contraction(
              backend, layer, position)) {
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

struct reference_tensor_capture {
  const char * name = nullptr;
  std::vector<float> values = {};
  std::array<int64_t, 4> shape = {1, 1, 1, 1};
};

struct reference_graph_capture {
  std::array<reference_tensor_capture, 33> entries = {{
      {"attn_norm-0", {}},
      {"Qcur-0", {}},
      {"Kcur-0", {}},
      {"Vcur-0", {}},
      {"kq-0", {}},
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
      {"kq-1", {}},
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

std::span<const float> find_reference_tensor(const reference_graph_capture & capture,
                                             const char * name);
std::span<const float> reference_last_token_row(std::span<const float> reference_values,
                                                size_t row_width);
bool capture_reference_graph_for_tokens(const generation_load_state & state,
                                        std::span<const llama_token> prompt_tokens,
                                        reference_graph_capture & graph_capture);
bool capture_reference_graph_for_generation_prefix(const generation_load_state & state,
                                                   std::span<const llama_token> prompt_tokens,
                                                   std::span<const int32_t> generated_tokens,
                                                   reference_graph_capture & graph_capture);
void dump_state_compare(const char * label,
                        std::span<const float> emel_values,
                        std::span<const float> reference_values);
void dump_state_compare(const char * label,
                        std::span<const uint16_t> emel_values,
                        std::span<const float> reference_values);
void dump_state_compare(const char * label,
                        std::span<const float> emel_values,
                        std::span<const uint16_t> reference_values);
void dump_state_compare(const char * label,
                        std::span<const uint16_t> emel_values,
                        std::span<const uint16_t> reference_values);
void dump_q8_quantize_compare(const char * label, std::span<const float> input);

void dump_prompt0_reference_attn_out_ffn_debug(
    const generation_load_state & state,
    std::span<const llama_token> prompt_tokens) {
  if (state.model_data == nullptr || prompt_tokens.empty()) {
    return;
  }

  reference_graph_capture graph_capture = {};
  if (!capture_reference_graph_for_generation_prefix(
          state,
          std::span<const llama_token>(prompt_tokens.data(), 1u),
          std::span<const int32_t>{},
          graph_capture)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "reference graph capture failed\n");
    return;
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "backend prepare failed\n");
    return;
  }

  const int32_t token_id = static_cast<int32_t>(prompt_tokens.front());
  if (!emel::generator::detail::copy_tensor_row(
          *backend.token_embedding.tensor, token_id, backend.hidden)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "token embedding replay failed\n");
    return;
  }

  auto dump_ffn_matmul_modes = [&](const char * label_base,
                                   const emel::generator::detail::tensor_matrix & matrix,
                                   std::span<const float> input,
                                   std::span<const float> reference_values) {
    std::vector<float> exact_values(reference_values.size(), 0.0f);
    if (matmul_vector_dequantized(matrix, input, exact_values)) {
      dump_state_compare((std::string(label_base) + ".exact").c_str(),
                         exact_values,
                         reference_values);
    }

    std::vector<float> reference_q8_values(reference_values.size(), 0.0f);
    if (matmul_vector_reference_q8(matrix, input, reference_q8_values)) {
      dump_state_compare((std::string(label_base) + ".reference_q8").c_str(),
                         reference_q8_values,
                         reference_values);
    }

    std::vector<float> scalar_quant_values(reference_values.size(), 0.0f);
    if (matmul_vector_scalar_quantized(matrix, input, scalar_quant_values)) {
      dump_state_compare((std::string(label_base) + ".scalar_quantized").c_str(),
                         scalar_quant_values,
                         reference_values);
    }
  };

  const auto reference_row = [&](const char * key, const size_t width) {
    return reference_last_token_row(find_reference_tensor(graph_capture, key), width);
  };

  const std::span<const float> reference_attn_out =
      reference_row("attn_out-0", backend.projected.size());
  const std::span<const float> reference_ffn_inp =
      reference_row("ffn_inp-0", backend.hidden.size());
  const std::span<const float> reference_ffn_norm =
      reference_row("ffn_norm-0", backend.norm.size());
  const std::span<const float> reference_ffn_gate =
      reference_row("ffn_gate-0", backend.gate.size());
  const std::span<const float> reference_ffn_up =
      reference_row("ffn_up-0", backend.up.size());
  const std::span<const float> reference_ffn_swiglu =
      reference_row("ffn_swiglu-0", backend.ffn_hidden.size());
  const std::span<const float> reference_ffn_out =
      reference_row("ffn_out-0", backend.projected.size());
  const std::span<const float> reference_l_out =
      reference_row("l_out-0", backend.hidden.size());

  if (reference_attn_out.size() != backend.projected.size() ||
      reference_ffn_inp.size() != backend.hidden.size() ||
      reference_ffn_norm.size() != backend.norm.size() ||
      reference_ffn_gate.size() != backend.gate.size() ||
      reference_ffn_up.size() != backend.up.size() ||
      reference_ffn_swiglu.size() != backend.ffn_hidden.size() ||
      reference_ffn_out.size() != backend.projected.size() ||
      reference_l_out.size() != backend.hidden.size()) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "reference tensors unavailable\n");
    return;
  }

  constexpr int32_t layer_index = 0;
  constexpr int32_t position = 0;
  auto & block = backend.blocks[0];
  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: qkv matmul failed\n");
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
      emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
      backend.key_cache.data() + cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
      backend.value_cache.data() + cache_offset);

  const auto build_single_token_attn_ctx =
      [&](const auto value_cache_row) {
        std::vector<float> out(static_cast<size_t>(backend.n_embd), 0.0f);
        if (value_cache_row.size() != static_cast<size_t>(kv_dim)) {
          return out;
        }
        for (int32_t head = 0; head < backend.n_head; ++head) {
          const int32_t kv_head = head / backend.n_rep;
          const size_t out_offset =
              static_cast<size_t>(head) * static_cast<size_t>(backend.head_dim);
          const size_t cache_row_offset =
              static_cast<size_t>(kv_head) * static_cast<size_t>(backend.head_dim_kv);
          for (int32_t dim = 0; dim < backend.head_dim; ++dim) {
            out[out_offset + static_cast<size_t>(dim)] = read_debug_value(
                value_cache_row, cache_row_offset + static_cast<size_t>(dim));
          }
        }
        return out;
      };

  const std::span<const uint16_t> runtime_value_cache_row(
      backend.value_cache.data() + cache_offset, static_cast<size_t>(kv_dim));
  const std::vector<float> expected_runtime_attn_ctx =
      build_single_token_attn_ctx(runtime_value_cache_row);
  std::vector<float> expected_reference_attn_ctx = {};

  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx != nullptr &&
      run_reference_prefix_decode(reference_ctx.get(),
                                  std::span<const llama_token>(prompt_tokens.data(), 1u),
                                  std::span<const int32_t>{})) {
    std::vector<float> reference_value_cache_rows;
    if (capture_reference_value_cache_rows(reference_ctx.get(), 0, reference_value_cache_rows) &&
        reference_value_cache_rows.size() >= static_cast<size_t>(kv_dim)) {
      const std::span<const float> reference_value_cache_row(
          reference_value_cache_rows.data(), static_cast<size_t>(kv_dim));
      expected_reference_attn_ctx = build_single_token_attn_ctx(reference_value_cache_row);
      dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.value_cache.runtime",
                         runtime_value_cache_row,
                         reference_value_cache_row);

      auto dump_value_cache_mode = [&](const char * label,
                                       const auto matmul_fn) {
        std::vector<float> values(static_cast<size_t>(kv_dim), 0.0f);
        if (!matmul_fn(block.attention_v, backend.norm, values)) {
          return;
        }
        std::vector<float> rounded_values(static_cast<size_t>(kv_dim), 0.0f);
        emel::generator::detail::store_fp16_rounded_cache(values, rounded_values.data());
        dump_state_compare(label, rounded_values, reference_value_cache_row);
      };

      dump_value_cache_mode(
          "generation_debug.reference_attn_out.prompt0.layer0.value_cache.exact",
          [&](const emel::generator::detail::tensor_matrix & matrix,
              std::span<const float> input,
              std::span<float> output) {
            return matmul_vector_dequantized(matrix, input, output);
          });
      dump_value_cache_mode(
          "generation_debug.reference_attn_out.prompt0.layer0.value_cache.reference_q8",
          [&](const emel::generator::detail::tensor_matrix & matrix,
              std::span<const float> input,
              std::span<float> output) {
            return matmul_vector_reference_q8(matrix, input, output);
          });
      dump_value_cache_mode(
          "generation_debug.reference_attn_out.prompt0.layer0.value_cache.scalar_quantized",
          [&](const emel::generator::detail::tensor_matrix & matrix,
              std::span<const float> input,
              std::span<float> output) {
            return matmul_vector_scalar_quantized(matrix, input, output);
          });
    }
  }

  if (!emel::generator::detail::compute_attention(
          backend, layer_index, position + 1, backend.q_attn) ||
      !emel::generator::detail::matmul_vector(
          backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "attention replay failed\n");
    return;
  }

  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.attn_ctx.runtime_value_cache",
                     backend.attn_ctx,
                     expected_runtime_attn_ctx);
  if (!expected_reference_attn_ctx.empty()) {
    dump_state_compare(
        "generation_debug.reference_attn_out.prompt0.layer0.attn_ctx.reference_value_cache",
        backend.attn_ctx,
        expected_reference_attn_ctx);
  }
  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.attn_out.runtime",
                     backend.projected,
                     reference_attn_out);
  dump_ffn_matmul_modes("generation_debug.reference_attn_out.prompt0.layer0.attn_out",
                        block.attention_output,
                        expected_runtime_attn_ctx,
                        reference_attn_out);

  std::copy(reference_attn_out.begin(), reference_attn_out.end(), backend.projected.begin());
  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_inp",
                     backend.hidden,
                     reference_ffn_inp);

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(
          backend, block.feed_forward_gate, backend.norm, backend.gate) ||
      !emel::generator::detail::matmul_vector(
          backend, block.feed_forward_up, backend.norm, backend.up)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "ffn gate/up replay failed\n");
    return;
  }

  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_norm",
                     backend.norm,
                     reference_ffn_norm);
  dump_q8_quantize_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_norm_q8",
                           backend.norm);
  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_gate",
                     backend.gate,
                     reference_ffn_gate);
  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_up",
                     backend.up,
                     reference_ffn_up);
  dump_ffn_matmul_modes("generation_debug.reference_attn_out.prompt0.layer0.ffn_gate",
                        block.feed_forward_gate,
                        backend.norm,
                        reference_ffn_gate);
  dump_ffn_matmul_modes("generation_debug.reference_attn_out.prompt0.layer0.ffn_up",
                        block.feed_forward_up,
                        backend.norm,
                        reference_ffn_up);

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
  }
  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_swiglu",
                     backend.ffn_hidden,
                     reference_ffn_swiglu);
  dump_q8_quantize_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_hidden_q8",
                           backend.ffn_hidden);

  if (!emel::generator::detail::matmul_vector(
          backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
    std::fprintf(stdout,
                 "generation_debug.reference_attn_out.prompt0.layer0: "
                 "ffn down replay failed\n");
    return;
  }

  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.ffn_out",
                     backend.projected,
                     reference_ffn_out);
  dump_ffn_matmul_modes("generation_debug.reference_attn_out.prompt0.layer0.ffn_out",
                        block.feed_forward_down,
                        backend.ffn_hidden,
                        reference_ffn_out);

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }
  dump_state_compare("generation_debug.reference_attn_out.prompt0.layer0.l_out",
                     backend.hidden,
                     reference_l_out);
}

void dump_matrix_compare(const char * label,
                         emel::generator::detail::native_backend & backend,
                         const emel::generator::detail::tensor_matrix & matrix,
                         std::span<const float> input);

void dump_matrix_compare_reference_q8(const char * label,
                                      emel::generator::detail::native_backend & backend,
                                      const emel::generator::detail::tensor_matrix & matrix,
                                      std::span<const float> input);

float ggml_row_dot(const emel::generator::detail::tensor_matrix & matrix,
                   uint32_t row,
                   const kernel_quant::block_q8_k * q8_blocks,
                   uint64_t block_count);

void dump_generation_selected_step_stage_debug(
    const generation_load_state & state,
    const emel::paritychecker::parity_options & opts,
    const generation_result & reference_result,
    const int32_t generated_index) {
  if (state.model_data == nullptr || generated_index < 0 ||
      generated_index >= static_cast<int32_t>(reference_result.trace.token_ids.size())) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.gen_step: tokenize failed\n");
    return;
  }

  std::vector<int32_t> generated_prefix(
      reference_result.trace.token_ids.begin(),
      reference_result.trace.token_ids.begin() + generated_index + 1);
  reference_graph_capture graph_capture = {};
  if (!capture_reference_graph_for_generation_prefix(
          state, prompt_tokens, generated_prefix, graph_capture)) {
    std::fprintf(stdout, "generation_debug.gen_step: reference graph capture failed\n");
    return;
  }

  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx == nullptr ||
      !run_reference_prefix_decode(
          reference_ctx.get(), prompt_tokens, std::span<const int32_t>(generated_prefix))) {
    std::fprintf(stdout, "generation_debug.gen_step: reference decode failed\n");
    return;
  }

  std::vector<float> reference_layer0_key_cache;
  std::vector<float> reference_layer0_value_cache;
  std::vector<float> reference_layer1_key_cache;
  std::vector<float> reference_layer1_value_cache;
  const bool have_layer0_key_cache =
      capture_reference_key_cache_rows(reference_ctx.get(), 0, reference_layer0_key_cache);
  const bool have_layer0_value_cache =
      capture_reference_value_cache_rows(reference_ctx.get(), 0, reference_layer0_value_cache);
  const bool have_layer1_key_cache =
      capture_reference_key_cache_rows(reference_ctx.get(), 1, reference_layer1_key_cache);
  const bool have_layer1_value_cache =
      capture_reference_value_cache_rows(reference_ctx.get(), 1, reference_layer1_value_cache);

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.gen_step: backend prepare failed\n");
    return;
  }

  std::vector<int32_t> prefix_tokens;
  prefix_tokens.reserve(prompt_tokens.size() + generated_prefix.size());
  for (const llama_token token : prompt_tokens) {
    prefix_tokens.push_back(static_cast<int32_t>(token));
  }
  prefix_tokens.insert(prefix_tokens.end(), generated_prefix.begin(), generated_prefix.end());
  if (prefix_tokens.empty()) {
    return;
  }

  if (prefix_tokens.size() > 1u) {
    const std::span<const int32_t> prior_tokens{
        prefix_tokens.data(), prefix_tokens.size() - 1u};
    if (!run_prefill_from_token_prefix(backend, prior_tokens)) {
      std::fprintf(stdout, "generation_debug.gen_step: prior prefix replay failed\n");
      return;
    }
  } else {
    backend.kv_cache_tokens = 0;
  }

  const int32_t token_id = prefix_tokens.back();
  const int32_t position = static_cast<int32_t>(prefix_tokens.size() - 1u);
  if (!emel::generator::detail::copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
    std::fprintf(stdout, "generation_debug.gen_step: token embedding replay failed\n");
    return;
  }

  std::fprintf(stdout,
               "generation_debug.gen_step: generated_index=%d total_prefix_tokens=%zu\n",
               generated_index,
               prefix_tokens.size());

  const auto dump_cache_rows_compare =
      [](const std::string & label_prefix,
         const char * suffix,
         const auto runtime_rows,
         std::span<const float> reference_rows,
         const int32_t row_width) {
        if (runtime_rows.size() != reference_rows.size() || row_width <= 0) {
          std::fprintf(stdout, "%s.%s: unavailable\n", label_prefix.c_str(), suffix);
          return;
        }

        float max_abs = 0.0f;
        int32_t max_pos = 0;
        int32_t max_dim = 0;
        float max_emel = 0.0f;
        float max_reference = 0.0f;
        for (size_t idx = 0; idx < runtime_rows.size(); ++idx) {
          const float emel_value = read_debug_value(runtime_rows, idx);
          const float reference_value = reference_rows[idx];
          const float diff = std::fabs(emel_value - reference_value);
          if (diff > max_abs) {
            max_abs = diff;
            max_pos = static_cast<int32_t>(idx / static_cast<size_t>(row_width));
            max_dim = static_cast<int32_t>(idx % static_cast<size_t>(row_width));
            max_emel = emel_value;
            max_reference = reference_value;
          }
        }

        std::fprintf(stdout,
                     "%s.%s: max_abs=%g pos=%d dim=%d emel=%g reference=%g\n",
                     label_prefix.c_str(),
                     suffix,
                     max_abs,
                     max_pos,
                     max_dim,
                     max_emel,
                     max_reference);
      };

  const auto dump_kernel_row_compare =
      [&](const std::string & label_prefix,
          const char * suffix,
          const emel::generator::detail::tensor_matrix & matrix,
          std::span<const float> input,
          const uint32_t row,
          std::span<const float> reference_values) {
        std::array<kernel_quant::block_q8_k, kernel_quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
        uint64_t block_count = 0;
        if (!quantize_input_blocks(input, q8_blocks, block_count) ||
            row >= static_cast<uint32_t>(matrix.rows)) {
          std::fprintf(stdout, "%s.%s: unavailable\n", label_prefix.c_str(), suffix);
          return;
        }

        std::vector<float> emel_out(static_cast<size_t>(matrix.rows));
        if (!emel::generator::detail::matmul_vector(
                backend, matrix, input, std::span<float>(emel_out))) {
          std::fprintf(stdout, "%s.%s: emel matmul failed\n", label_prefix.c_str(), suffix);
          return;
        }

        const auto * row_ptr =
            static_cast<const uint8_t *>(matrix.tensor->data) +
            static_cast<size_t>(row) *
                emel::generator::detail::row_storage_bytes(*matrix.tensor, matrix.cols);
        const float ggml_value = ggml_row_dot(matrix, row, q8_blocks.data(), block_count);
        float scalar_value = std::numeric_limits<float>::quiet_NaN();
        float neon_value = std::numeric_limits<float>::quiet_NaN();
        switch (static_cast<emel::kernel::event::dtype>(matrix.tensor->type)) {
          case emel::kernel::event::dtype::q2_k: {
            const auto * lhs =
                reinterpret_cast<const kernel_quant::block_q2_k *>(row_ptr);
            scalar_value =
                emel::kernel::detail::dot_q2_k_q8_k_row_scalar(lhs, q8_blocks.data(), block_count);
#if defined(__aarch64__) || defined(__ARM_NEON)
            neon_value = emel::kernel::aarch64::detail::dot_q2_k_q8_k_row_neon(
                lhs, q8_blocks.data(), block_count);
#endif
            break;
          }
          case emel::kernel::event::dtype::q3_k: {
            const auto * lhs =
                reinterpret_cast<const kernel_quant::block_q3_k *>(row_ptr);
            scalar_value =
                emel::kernel::detail::dot_q3_k_q8_k_row_scalar(lhs, q8_blocks.data(), block_count);
#if defined(__aarch64__) || defined(__ARM_NEON)
            neon_value = emel::kernel::aarch64::detail::dot_q3_k_q8_k_row_neon(
                lhs, q8_blocks.data(), block_count);
#endif
            break;
          }
          case emel::kernel::event::dtype::q6_k: {
            const auto * lhs =
                reinterpret_cast<const kernel_quant::block_q6_k *>(row_ptr);
            scalar_value =
                emel::kernel::detail::dot_q6_k_q8_k_row_scalar(lhs, q8_blocks.data(), block_count);
#if defined(__aarch64__) || defined(__ARM_NEON)
            neon_value = emel::kernel::aarch64::detail::dot_q6_k_q8_k_row_neon(
                lhs, q8_blocks.data(), block_count);
#endif
            break;
          }
          default:
            break;
        }

        const float reference_value =
            row < reference_values.size() ? reference_values[static_cast<size_t>(row)]
                                          : std::numeric_limits<float>::quiet_NaN();
        std::fprintf(stdout,
                     "%s.%s: row=%u emel=%0.9g emel_bits=0x%08x "
                     "ggml=%0.9g ggml_bits=0x%08x "
                     "scalar=%0.9g scalar_bits=0x%08x "
                     "neon=%0.9g neon_bits=0x%08x "
                     "reference=%0.9g reference_bits=0x%08x\n",
                     label_prefix.c_str(),
                     suffix,
                     row,
                     emel_out[static_cast<size_t>(row)],
                     kernel_quant::fp32_to_bits(emel_out[static_cast<size_t>(row)]),
                     ggml_value,
                     kernel_quant::fp32_to_bits(ggml_value),
                     scalar_value,
                     kernel_quant::fp32_to_bits(scalar_value),
                     neon_value,
                     kernel_quant::fp32_to_bits(neon_value),
                     reference_value,
                     kernel_quant::fp32_to_bits(reference_value));
      };

  const auto reference_stage_row = [&](const char * key, const size_t width) {
    return reference_last_token_row(find_reference_tensor(graph_capture, key), width);
  };

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    const std::string label_prefix =
        "generation_debug.gen_step.gen" + std::to_string(generated_index) +
        ".layer" + std::to_string(layer);
    auto & block = backend.blocks[static_cast<size_t>(layer)];
    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm)) {
      std::fprintf(stdout, "%s.attn_norm: replay failed\n", label_prefix.c_str());
      return;
    }
    if (const std::span<const float> reference_attn_norm =
            reference_stage_row(("attn_norm-" + std::to_string(layer)).c_str(), backend.norm.size());
        !reference_attn_norm.empty()) {
      dump_state_compare((label_prefix + ".attn_norm").c_str(), backend.norm, reference_attn_norm);
    }
    dump_q8_quantize_compare((label_prefix + ".attn_norm_q8").c_str(), backend.norm);
    dump_matrix_compare((label_prefix + ".attn_q_matmul").c_str(), backend, block.attention_q, backend.norm);
    dump_matrix_compare_reference_q8(
        (label_prefix + ".attn_q_matmul_refq8").c_str(), backend, block.attention_q, backend.norm);
    dump_matrix_compare((label_prefix + ".attn_k_matmul").c_str(), backend, block.attention_k, backend.norm);
    dump_matrix_compare_reference_q8(
        (label_prefix + ".attn_k_matmul_refq8").c_str(), backend, block.attention_k, backend.norm);
    dump_matrix_compare((label_prefix + ".attn_v_matmul").c_str(), backend, block.attention_v, backend.norm);
    dump_matrix_compare_reference_q8(
        (label_prefix + ".attn_v_matmul_refq8").c_str(), backend, block.attention_v, backend.norm);

    if (!emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
        !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
      std::fprintf(stdout, "%s.qkv: replay failed\n", label_prefix.c_str());
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

    const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
    const size_t layer_cache_base_offset =
        emel::generator::detail::layer_cache_offset(backend, layer, 0, kv_dim);
    const size_t layer_cache_row_offset =
        static_cast<size_t>(position) * static_cast<size_t>(kv_dim);
    const std::vector<float> * reference_key_cache = nullptr;
    const std::vector<float> * reference_value_cache = nullptr;
    bool have_reference_key_cache = false;
    bool have_reference_value_cache = false;
    if (layer == 0) {
      reference_key_cache = &reference_layer0_key_cache;
      reference_value_cache = &reference_layer0_value_cache;
      have_reference_key_cache = have_layer0_key_cache;
      have_reference_value_cache = have_layer0_value_cache;
    } else if (layer == 1) {
      reference_key_cache = &reference_layer1_key_cache;
      reference_value_cache = &reference_layer1_value_cache;
      have_reference_key_cache = have_layer1_key_cache;
      have_reference_value_cache = have_layer1_value_cache;
    }

    if (const std::span<const float> reference_q =
            reference_stage_row(("Qcur-" + std::to_string(layer)).c_str(), backend.q.size());
        !reference_q.empty()) {
      dump_state_compare((label_prefix + ".q").c_str(), backend.q, reference_q);
    }
    if (const std::span<const float> reference_k =
            reference_stage_row(("Kcur-" + std::to_string(layer)).c_str(), backend.k.size());
        !reference_k.empty()) {
      dump_state_compare((label_prefix + ".k").c_str(), backend.k, reference_k);

      std::vector<float> rounded_reference_k(reference_k.begin(), reference_k.end());
      emel::generator::detail::store_fp16_rounded_cache(reference_k, rounded_reference_k.data());
      if (layer == 0 && have_layer0_key_cache &&
          reference_layer0_key_cache.size() >=
              layer_cache_row_offset + static_cast<size_t>(backend.head_dim_kv * backend.n_head_kv)) {
        dump_state_compare((label_prefix + ".k.reference_round_to_cache").c_str(),
                           rounded_reference_k,
                           std::span<const float>(reference_layer0_key_cache.data() +
                                                      layer_cache_row_offset,
                                                  static_cast<size_t>(backend.head_dim_kv *
                                                                      backend.n_head_kv)));
      }
      if (layer == 1 && have_layer1_key_cache &&
          reference_layer1_key_cache.size() >=
              layer_cache_row_offset + static_cast<size_t>(backend.head_dim_kv * backend.n_head_kv)) {
        dump_state_compare((label_prefix + ".k.reference_round_to_cache").c_str(),
                           rounded_reference_k,
                           std::span<const float>(reference_layer1_key_cache.data() +
                                                      layer_cache_row_offset,
                                                  static_cast<size_t>(backend.head_dim_kv *
                                                                      backend.n_head_kv)));
      }
    }
    if (const std::span<const float> reference_v =
            reference_stage_row(("Vcur-" + std::to_string(layer)).c_str(), backend.v.size());
        !reference_v.empty()) {
      dump_state_compare((label_prefix + ".v").c_str(), backend.v, reference_v);

      std::vector<float> rounded_reference_v(reference_v.begin(), reference_v.end());
      emel::generator::detail::store_fp16_rounded_cache(reference_v, rounded_reference_v.data());
      if (layer == 0 && have_layer0_value_cache &&
          reference_layer0_value_cache.size() >=
              layer_cache_row_offset + static_cast<size_t>(backend.head_dim_kv * backend.n_head_kv)) {
        dump_state_compare((label_prefix + ".v.reference_round_to_cache").c_str(),
                           rounded_reference_v,
                           std::span<const float>(reference_layer0_value_cache.data() +
                                                      layer_cache_row_offset,
                                                  static_cast<size_t>(backend.head_dim_kv *
                                                                      backend.n_head_kv)));
      }
      if (layer == 1 && have_layer1_value_cache &&
          reference_layer1_value_cache.size() >=
              layer_cache_row_offset + static_cast<size_t>(backend.head_dim_kv * backend.n_head_kv)) {
        dump_state_compare((label_prefix + ".v.reference_round_to_cache").c_str(),
                           rounded_reference_v,
                           std::span<const float>(reference_layer1_value_cache.data() +
                                                      layer_cache_row_offset,
                                                  static_cast<size_t>(backend.head_dim_kv *
                                                                      backend.n_head_kv)));
      }

      if (generated_index == 2 && layer == 0) {
        dump_kernel_row_compare(
            label_prefix, "attn_v_row319", block.attention_v, backend.norm, 319u, reference_v);
      }
    }
    if (generated_index == 2 && layer == 1) {
      if (const std::span<const float> reference_k =
              reference_stage_row(("Kcur-" + std::to_string(layer)).c_str(), backend.k.size());
          !reference_k.empty()) {
        dump_kernel_row_compare(
            label_prefix, "attn_k_row475", block.attention_k, backend.norm, 475u, reference_k);
      }
    }

    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
        backend.q_attn.data());
    if (have_reference_key_cache && reference_key_cache != nullptr && position > 0 &&
        reference_key_cache->size() >= static_cast<size_t>(position) * static_cast<size_t>(kv_dim)) {
      dump_cache_rows_compare(
          label_prefix,
          "key_cache_prior",
          std::span<const uint16_t>(backend.key_cache.data() + layer_cache_base_offset,
                                 static_cast<size_t>(position) * static_cast<size_t>(kv_dim)),
          std::span<const float>(reference_key_cache->data(),
                                 static_cast<size_t>(position) * static_cast<size_t>(kv_dim)),
          kv_dim);
    }
    if (have_reference_value_cache && reference_value_cache != nullptr && position > 0 &&
        reference_value_cache->size() >=
            static_cast<size_t>(position) * static_cast<size_t>(kv_dim)) {
      dump_cache_rows_compare(
          label_prefix,
          "value_cache_prior",
          std::span<const uint16_t>(backend.value_cache.data() + layer_cache_base_offset,
                                 static_cast<size_t>(position) * static_cast<size_t>(kv_dim)),
          std::span<const float>(reference_value_cache->data(),
                                 static_cast<size_t>(position) * static_cast<size_t>(kv_dim)),
          kv_dim);
    }
    const size_t cache_offset =
        emel::generator::detail::layer_cache_offset(backend, layer, position, kv_dim);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
        backend.key_cache.data() + cache_offset);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
        backend.value_cache.data() + cache_offset);
    if (have_reference_key_cache && reference_key_cache != nullptr &&
        reference_key_cache->size() >=
            static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)) {
      dump_cache_rows_compare(
          label_prefix,
          "key_cache_full",
          std::span<const uint16_t>(backend.key_cache.data() + layer_cache_base_offset,
                                 static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)),
          std::span<const float>(reference_key_cache->data(),
                                 static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)),
          kv_dim);
    }
    if (have_reference_value_cache && reference_value_cache != nullptr &&
        reference_value_cache->size() >=
            static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)) {
      dump_cache_rows_compare(
          label_prefix,
          "value_cache_full",
          std::span<const uint16_t>(backend.value_cache.data() + layer_cache_base_offset,
                                 static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)),
          std::span<const float>(reference_value_cache->data(),
                                 static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)),
          kv_dim);
    }

    if (layer == 0 && have_layer0_key_cache &&
        reference_layer0_key_cache.size() >= layer_cache_row_offset + static_cast<size_t>(kv_dim)) {
      dump_state_compare((label_prefix + ".key_cache").c_str(),
                         std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                                static_cast<size_t>(kv_dim)),
                         std::span<const float>(reference_layer0_key_cache.data() +
                                                    layer_cache_row_offset,
                                                static_cast<size_t>(kv_dim)));
    }
    if (layer == 0 && have_layer0_value_cache &&
        reference_layer0_value_cache.size() >= layer_cache_row_offset + static_cast<size_t>(kv_dim)) {
      dump_state_compare((label_prefix + ".value_cache").c_str(),
                         std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                                static_cast<size_t>(kv_dim)),
                         std::span<const float>(reference_layer0_value_cache.data() +
                                                    layer_cache_row_offset,
                                                static_cast<size_t>(kv_dim)));
    }
    if (layer == 1 && have_layer1_key_cache &&
        reference_layer1_key_cache.size() >= layer_cache_row_offset + static_cast<size_t>(kv_dim)) {
      dump_state_compare((label_prefix + ".key_cache").c_str(),
                         std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                                static_cast<size_t>(kv_dim)),
                         std::span<const float>(reference_layer1_key_cache.data() +
                                                    layer_cache_row_offset,
                                                static_cast<size_t>(kv_dim)));
    }
    if (layer == 1 && have_layer1_value_cache &&
        reference_layer1_value_cache.size() >= layer_cache_row_offset + static_cast<size_t>(kv_dim)) {
      dump_state_compare((label_prefix + ".value_cache").c_str(),
                         std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                                static_cast<size_t>(kv_dim)),
                         std::span<const float>(reference_layer1_value_cache.data() +
                                                    layer_cache_row_offset,
                                                static_cast<size_t>(kv_dim)));
    }

    const std::span<const float> reference_kqv_out =
        reference_stage_row(("kqv_out-" + std::to_string(layer)).c_str(), backend.attn_ctx.size());
    if (!reference_kqv_out.empty()) {
      const size_t layer_cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer, 0, kv_dim);
      const std::span<const float> q_attn_rows(
          backend.q_attn.data(), static_cast<size_t>(backend.n_head * backend.head_dim));
      const std::span<const uint16_t> key_rows(
          backend.key_cache.data() + layer_cache_offset,
          static_cast<size_t>(backend.n_ctx * kv_dim));
      const std::span<const uint16_t> value_rows(
          backend.value_cache.data() + layer_cache_offset,
          static_cast<size_t>(backend.n_ctx * kv_dim));
      std::vector<float> ggml_nonflash_exact_ctx;
      if (run_ggml_nonflash_attn_case(q_attn_rows,
                                      key_rows,
                                      value_rows,
                                      static_cast<int64_t>(backend.head_dim),
                                      static_cast<int64_t>(backend.n_ctx),
                                      static_cast<int64_t>(position + 1),
                                      static_cast<int64_t>(backend.n_head),
                                      static_cast<int64_t>(backend.n_head_kv),
                                      1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
                                      ggml_nonflash_exact_ctx)) {
        dump_state_compare((label_prefix + ".kqv_out.ggml_nonflash_exact").c_str(),
                           ggml_nonflash_exact_ctx,
                           reference_kqv_out);
      } else {
        std::fprintf(stdout,
                     "%s.kqv_out.ggml_nonflash_exact: replay failed\n",
                     label_prefix.c_str());
      }

      std::vector<float> emel_nonflash_f16_ctx;
      if (run_emel_nonflash_f16_ggml_softmax_case(q_attn_rows,
                                                  key_rows,
                                                  value_rows,
                                                  static_cast<int64_t>(backend.head_dim),
                                                  static_cast<int64_t>(backend.n_ctx),
                                                  static_cast<int64_t>(position + 1),
                                                  static_cast<int64_t>(backend.n_head),
                                                  static_cast<int64_t>(backend.n_head_kv),
                                                  1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
                                                  emel_nonflash_f16_ctx)) {
        dump_state_compare((label_prefix + ".kqv_out.emel_nonflash_f16").c_str(),
                           emel_nonflash_f16_ctx,
                           reference_kqv_out);
        if (!ggml_nonflash_exact_ctx.empty()) {
          dump_state_compare((label_prefix + ".kqv_out.emel_vs_exact_nonflash").c_str(),
                             emel_nonflash_f16_ctx,
                             ggml_nonflash_exact_ctx);
        }
      } else {
        std::fprintf(stdout,
                     "%s.kqv_out.emel_nonflash_f16: replay failed\n",
                     label_prefix.c_str());
      }
    }

    if (!emel::generator::detail::compute_attention(
            backend, layer, position + 1, backend.q_attn) ||
        !emel::generator::detail::matmul_vector(
            backend, block.attention_output, backend.attn_ctx, backend.projected)) {
      std::fprintf(stdout, "%s.attention: replay failed\n", label_prefix.c_str());
      return;
    }
    if (!reference_kqv_out.empty()) {
      dump_state_compare((label_prefix + ".kqv_out").c_str(), backend.attn_ctx, reference_kqv_out);
    }
    if (const std::span<const float> reference_attn_out =
            reference_stage_row(("attn_out-" + std::to_string(layer)).c_str(), backend.projected.size());
        !reference_attn_out.empty()) {
      dump_state_compare((label_prefix + ".attn_out").c_str(), backend.projected, reference_attn_out);
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
    if (const std::span<const float> reference_ffn_inp =
            reference_stage_row(("ffn_inp-" + std::to_string(layer)).c_str(), backend.hidden.size());
        !reference_ffn_inp.empty()) {
      dump_state_compare((label_prefix + ".ffn_inp").c_str(), backend.hidden, reference_ffn_inp);
    }

    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
        !emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
        !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
      std::fprintf(stdout, "%s.ffn: replay failed\n", label_prefix.c_str());
      return;
    }
    if (const std::span<const float> reference_ffn_norm =
            reference_stage_row(("ffn_norm-" + std::to_string(layer)).c_str(), backend.norm.size());
        !reference_ffn_norm.empty()) {
      dump_state_compare((label_prefix + ".ffn_norm").c_str(), backend.norm, reference_ffn_norm);
    }
    dump_q8_quantize_compare((label_prefix + ".ffn_norm_q8").c_str(), backend.norm);
    dump_matrix_compare(
        (label_prefix + ".ffn_gate_matmul").c_str(), backend, block.feed_forward_gate, backend.norm);
    dump_matrix_compare_reference_q8((label_prefix + ".ffn_gate_matmul_refq8").c_str(),
                                     backend,
                                     block.feed_forward_gate,
                                     backend.norm);
    dump_matrix_compare(
        (label_prefix + ".ffn_up_matmul").c_str(), backend, block.feed_forward_up, backend.norm);
    dump_matrix_compare_reference_q8((label_prefix + ".ffn_up_matmul_refq8").c_str(),
                                     backend,
                                     block.feed_forward_up,
                                     backend.norm);
    if (const std::span<const float> reference_ffn_gate =
            reference_stage_row(("ffn_gate-" + std::to_string(layer)).c_str(), backend.gate.size());
        !reference_ffn_gate.empty()) {
      dump_state_compare((label_prefix + ".ffn_gate").c_str(), backend.gate, reference_ffn_gate);
    }
    if (const std::span<const float> reference_ffn_up =
            reference_stage_row(("ffn_up-" + std::to_string(layer)).c_str(), backend.up.size());
        !reference_ffn_up.empty()) {
      dump_state_compare((label_prefix + ".ffn_up").c_str(), backend.up, reference_ffn_up);
    }

    for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
      backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
    }
    if (const std::span<const float> reference_ffn_swiglu =
            reference_stage_row(("ffn_swiglu-" + std::to_string(layer)).c_str(),
                                backend.ffn_hidden.size());
        !reference_ffn_swiglu.empty()) {
      dump_state_compare(
          (label_prefix + ".ffn_swiglu").c_str(), backend.ffn_hidden, reference_ffn_swiglu);
    }
    if (!emel::generator::detail::matmul_vector(
            backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
      std::fprintf(stdout, "%s.ffn_down: replay failed\n", label_prefix.c_str());
      return;
    }
    dump_q8_quantize_compare((label_prefix + ".ffn_hidden_q8").c_str(), backend.ffn_hidden);
    dump_matrix_compare((label_prefix + ".ffn_down_matmul").c_str(),
                        backend,
                        block.feed_forward_down,
                        backend.ffn_hidden);
    dump_matrix_compare_reference_q8((label_prefix + ".ffn_down_matmul_refq8").c_str(),
                                     backend,
                                     block.feed_forward_down,
                                     backend.ffn_hidden);
    if (const std::span<const float> reference_ffn_out =
            reference_stage_row(("ffn_out-" + std::to_string(layer)).c_str(), backend.projected.size());
        !reference_ffn_out.empty()) {
      dump_state_compare((label_prefix + ".ffn_out").c_str(), backend.projected, reference_ffn_out);

      std::vector<float> exact_ffn_out(backend.projected.size(), 0.0f);
      if (matmul_vector_dequantized(block.feed_forward_down, backend.ffn_hidden, exact_ffn_out)) {
        dump_state_compare((label_prefix + ".ffn_out.exact").c_str(),
                           exact_ffn_out,
                           reference_ffn_out);
      }

      std::vector<float> reference_q8_ffn_out(backend.projected.size(), 0.0f);
      if (matmul_vector_reference_q8(
              block.feed_forward_down, backend.ffn_hidden, reference_q8_ffn_out)) {
        dump_state_compare((label_prefix + ".ffn_out.reference_q8").c_str(),
                           reference_q8_ffn_out,
                           reference_ffn_out);
      }

      std::vector<float> scalar_quant_ffn_out(backend.projected.size(), 0.0f);
      if (matmul_vector_scalar_quantized(
              block.feed_forward_down, backend.ffn_hidden, scalar_quant_ffn_out)) {
        dump_state_compare((label_prefix + ".ffn_out.scalar_quantized").c_str(),
                           scalar_quant_ffn_out,
                           reference_ffn_out);
      }
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
    if (const std::span<const float> reference_l_out =
            reference_stage_row(("l_out-" + std::to_string(layer)).c_str(), backend.hidden.size());
        !reference_l_out.empty()) {
      dump_state_compare((label_prefix + ".l_out").c_str(), backend.hidden, reference_l_out);
    }
  }
}

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
  std::vector<int32_t> prefix_generated_tokens;
  prefix_generated_tokens.reserve(static_cast<size_t>(token_mismatch_index));
  for (int32_t idx = 0; idx < token_mismatch_index; ++idx) {
    prefix_generated_tokens.push_back(reference_result.trace.token_ids[static_cast<size_t>(idx)]);
  }

  if (token_mismatch_index >= 299) {
    emel::generator::detail::native_backend dispatch_backend = {};
    emel::generator::detail::native_backend runtime_flash_q_attn_backend = {};
    emel::generator::detail::native_backend emel_prod_style_backend = {};
    emel::generator::detail::native_backend emel_prod_style_float_value_backend = {};
    emel::generator::detail::native_backend score_dot_probe_backend = {};
    emel::generator::detail::native_backend full_q_backend = {};
    emel::generator::detail::native_backend f32_attention_backend = {};
    emel::generator::detail::native_backend rounded_weight_attention_backend = {};
    emel::generator::detail::native_backend exact_backend = {};
    emel::generator::detail::native_backend attention_exact_backend = {};
    emel::generator::detail::native_backend ffn_exact_backend = {};
    emel::generator::detail::native_backend output_exact_backend = {};
    emel::generator::detail::native_backend ggml_f16_value_backend = {};
    emel::generator::detail::native_backend ggml_online_f16_backend = {};
    emel::generator::detail::native_backend ggml_f16_scores_backend = {};
    emel::generator::detail::native_backend ggml_f16_scores_ggml_softmax_backend = {};
    emel::generator::detail::native_backend ggml_nonflash_f16_backend = {};
    emel::generator::detail::native_backend ggml_nonflash_f16_ggml_softmax_backend = {};
    emel::generator::detail::native_backend q2_exact_backend = {};
    emel::generator::detail::native_backend q3_exact_backend = {};
    emel::generator::detail::native_backend q6_exact_backend = {};
    emel::generator::detail::native_backend q2_scalar_backend = {};
    emel::generator::detail::native_backend q3_scalar_backend = {};
    emel::generator::detail::native_backend q6_scalar_backend = {};
    emel::generator::detail::native_backend q2_reference_backend = {};
    emel::generator::detail::native_backend q3_reference_backend = {};
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
    const exact_matmul_mode scalar_quant_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_scalar_quantized = true,
    };
    const exact_matmul_mode scalar_quant_q2_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
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
    score_dot_probe_result score_dot_probe = {};

    if (emel::generator::detail::prepare(dispatch_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(runtime_flash_q_attn_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(emel_prod_style_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(emel_prod_style_float_value_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(score_dot_probe_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(full_q_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(f32_attention_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(rounded_weight_attention_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(attention_exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(ffn_exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(output_exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(ggml_f16_value_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(ggml_online_f16_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(ggml_f16_scores_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(
            ggml_f16_scores_ggml_softmax_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(ggml_nonflash_f16_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(
            ggml_nonflash_f16_ggml_softmax_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q2_exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q3_exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q6_exact_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q2_scalar_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q3_scalar_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q6_scalar_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q2_reference_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        emel::generator::detail::prepare(q3_reference_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        !run_prefill_from_token_prefix(dispatch_backend, prefix_tokens) ||
        !run_prefill_with_flash_request_q_attn(runtime_flash_q_attn_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_emel_prod_style(emel_prod_style_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_emel_prod_style_float_value(
            emel_prod_style_float_value_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_score_dot_probe(
            score_dot_probe_backend, prefix_tokens, score_dot_probe) ||
        !run_prefill_with_scalar_attention_full_q(full_q_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_no_weight_rounding(
            f32_attention_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_rounded_weight_scalar(
            rounded_weight_attention_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_matmul_mode(exact_backend, prefix_tokens, exact_all) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            attention_exact_backend, prefix_tokens, exact_attention_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(ffn_exact_backend, prefix_tokens, exact_ffn_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            output_exact_backend, prefix_tokens, exact_output_only) ||
        !run_prefill_with_scalar_attention_ggml_f16_value_contraction(
            ggml_f16_value_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_ggml_online_f16(
            ggml_online_f16_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_ggml_f16_scores(
            ggml_f16_scores_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_ggml_f16_scores_ggml_softmax(
            ggml_f16_scores_ggml_softmax_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_ggml_nonflash_f16(
            ggml_nonflash_f16_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_ggml_nonflash_f16_ggml_softmax(
            ggml_nonflash_f16_ggml_softmax_backend, prefix_tokens) ||
        !run_prefill_with_scalar_attention_matmul_mode(q2_exact_backend, prefix_tokens, exact_q2_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(q3_exact_backend, prefix_tokens, exact_q3_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(q6_exact_backend, prefix_tokens, exact_q6_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q2_scalar_backend, prefix_tokens, scalar_quant_q2_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q3_scalar_backend, prefix_tokens, scalar_quant_q3_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q6_scalar_backend, prefix_tokens, scalar_quant_q6_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q2_reference_backend, prefix_tokens, reference_q2_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q3_reference_backend, prefix_tokens, reference_q3_only)) {
      std::fprintf(stdout, "generation_debug.long: unable to replay mismatch prefix\n");
      return;
    }

    const argmax_summary dispatch_summary =
        select_argmax_from_logits(dispatch_backend.bound_logits.data(), dispatch_backend.n_vocab);
    const argmax_summary runtime_flash_q_attn_summary = select_argmax_from_logits(
        runtime_flash_q_attn_backend.bound_logits.data(), runtime_flash_q_attn_backend.n_vocab);
    const argmax_summary emel_prod_style_summary = select_argmax_from_logits(
        emel_prod_style_backend.bound_logits.data(), emel_prod_style_backend.n_vocab);
    const argmax_summary emel_prod_style_float_value_summary = select_argmax_from_logits(
        emel_prod_style_float_value_backend.bound_logits.data(),
        emel_prod_style_float_value_backend.n_vocab);
    const argmax_summary full_q_summary =
        select_argmax_from_logits(full_q_backend.bound_logits.data(), full_q_backend.n_vocab);
    const argmax_summary f32_attention_summary = select_argmax_from_logits(
        f32_attention_backend.bound_logits.data(), f32_attention_backend.n_vocab);
    const argmax_summary rounded_weight_attention_summary = select_argmax_from_logits(
        rounded_weight_attention_backend.bound_logits.data(),
        rounded_weight_attention_backend.n_vocab);
    const argmax_summary exact_summary =
        select_argmax_from_logits(exact_backend.bound_logits.data(), exact_backend.n_vocab);
    const argmax_summary attention_exact_summary = select_argmax_from_logits(
        attention_exact_backend.bound_logits.data(), attention_exact_backend.n_vocab);
    const argmax_summary ffn_exact_summary =
        select_argmax_from_logits(ffn_exact_backend.bound_logits.data(), ffn_exact_backend.n_vocab);
    const argmax_summary output_exact_summary = select_argmax_from_logits(
        output_exact_backend.bound_logits.data(), output_exact_backend.n_vocab);
    const argmax_summary ggml_f16_value_summary = select_argmax_from_logits(
        ggml_f16_value_backend.bound_logits.data(), ggml_f16_value_backend.n_vocab);
    const argmax_summary ggml_online_f16_summary = select_argmax_from_logits(
        ggml_online_f16_backend.bound_logits.data(), ggml_online_f16_backend.n_vocab);
    const argmax_summary ggml_f16_scores_summary = select_argmax_from_logits(
        ggml_f16_scores_backend.bound_logits.data(), ggml_f16_scores_backend.n_vocab);
    const argmax_summary ggml_f16_scores_ggml_softmax_summary = select_argmax_from_logits(
        ggml_f16_scores_ggml_softmax_backend.bound_logits.data(),
        ggml_f16_scores_ggml_softmax_backend.n_vocab);
    const argmax_summary ggml_nonflash_f16_summary = select_argmax_from_logits(
        ggml_nonflash_f16_backend.bound_logits.data(), ggml_nonflash_f16_backend.n_vocab);
    const argmax_summary ggml_nonflash_f16_ggml_softmax_summary = select_argmax_from_logits(
        ggml_nonflash_f16_ggml_softmax_backend.bound_logits.data(),
        ggml_nonflash_f16_ggml_softmax_backend.n_vocab);
    const argmax_summary q2_exact_summary =
        select_argmax_from_logits(q2_exact_backend.bound_logits.data(), q2_exact_backend.n_vocab);
    const argmax_summary q3_exact_summary =
        select_argmax_from_logits(q3_exact_backend.bound_logits.data(), q3_exact_backend.n_vocab);
    const argmax_summary q6_exact_summary =
        select_argmax_from_logits(q6_exact_backend.bound_logits.data(), q6_exact_backend.n_vocab);
    const argmax_summary q2_scalar_summary =
        select_argmax_from_logits(q2_scalar_backend.bound_logits.data(), q2_scalar_backend.n_vocab);
    const argmax_summary q3_scalar_summary =
        select_argmax_from_logits(q3_scalar_backend.bound_logits.data(), q3_scalar_backend.n_vocab);
    const argmax_summary q6_scalar_summary =
        select_argmax_from_logits(q6_scalar_backend.bound_logits.data(), q6_scalar_backend.n_vocab);
    const argmax_summary q2_reference_summary = select_argmax_from_logits(
        q2_reference_backend.bound_logits.data(), q2_reference_backend.n_vocab);
    const argmax_summary q3_reference_summary = select_argmax_from_logits(
        q3_reference_backend.bound_logits.data(), q3_reference_backend.n_vocab);
    const int32_t reference_token =
        reference_result.trace.token_ids[static_cast<size_t>(token_mismatch_index)];
    std::fprintf(stdout,
                 "generation_debug.long: prefix_tokens=%zu dispatch_argmax=%d "
                 "runtime_flash_q_attn_argmax=%d emel_prod_style_argmax=%d "
                 "emel_prod_style_float_value_argmax=%d "
                 "full_q_argmax=%d f32_attn_argmax=%d rounded_weight_attn_argmax=%d "
                 "exact_argmax=%d attention_exact_argmax=%d "
                 "ffn_exact_argmax=%d output_exact_argmax=%d ggml_f16_value_argmax=%d "
                 "ggml_online_f16_argmax=%d ggml_f16_scores_argmax=%d "
                 "ggml_f16_scores_ggml_softmax_argmax=%d ggml_nonflash_f16_argmax=%d "
                 "ggml_nonflash_f16_ggml_softmax_argmax=%d q2_exact_argmax=%d "
                 "q3_exact_argmax=%d q6_exact_argmax=%d q2_scalar_argmax=%d "
                 "q3_scalar_argmax=%d q6_scalar_argmax=%d q2_reference_argmax=%d "
                 "q3_reference_argmax=%d reference_token=%d\n",
                 prefix_tokens.size(),
                 dispatch_summary.selected_token,
                 runtime_flash_q_attn_summary.selected_token,
                 emel_prod_style_summary.selected_token,
                 emel_prod_style_float_value_summary.selected_token,
                 full_q_summary.selected_token,
                 f32_attention_summary.selected_token,
                 rounded_weight_attention_summary.selected_token,
                 exact_summary.selected_token,
                 attention_exact_summary.selected_token,
                 ffn_exact_summary.selected_token,
                 output_exact_summary.selected_token,
                 ggml_f16_value_summary.selected_token,
                 ggml_online_f16_summary.selected_token,
                 ggml_f16_scores_summary.selected_token,
                 ggml_f16_scores_ggml_softmax_summary.selected_token,
                 ggml_nonflash_f16_summary.selected_token,
                 ggml_nonflash_f16_ggml_softmax_summary.selected_token,
                 q2_exact_summary.selected_token,
                 q3_exact_summary.selected_token,
                 q6_exact_summary.selected_token,
                 q2_scalar_summary.selected_token,
                 q3_scalar_summary.selected_token,
                 q6_scalar_summary.selected_token,
                 q2_reference_summary.selected_token,
                 q3_reference_summary.selected_token,
                 reference_token);
    std::fprintf(stdout,
                 "generation_debug.long.score_dot: first_abs=%g token=%d layer=%d head=%d "
                 "position=%d emel=%g reference=%g max_abs=%g max_token=%d max_layer=%d "
                 "max_head=%d max_position=%d max_emel=%g max_reference=%g\n",
                 score_dot_probe.first_abs,
                 score_dot_probe.first_token,
                 score_dot_probe.first_layer,
                 score_dot_probe.first_head,
                 score_dot_probe.first_position,
                 score_dot_probe.first_emel,
                 score_dot_probe.first_reference,
                 score_dot_probe.max_abs,
                 score_dot_probe.max_token,
                 score_dot_probe.max_layer,
                 score_dot_probe.max_head,
                 score_dot_probe.max_position,
                 score_dot_probe.max_emel,
                 score_dot_probe.max_reference);
    dump_generation_prefix_timeline_debug(state, opts, emel_result, reference_result);
    dump_prompt0_reference_attn_out_ffn_debug(state, prompt_tokens);
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 3);
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 8);
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 32);
    if (token_mismatch_index > 0) {
      dump_generation_selected_step_stage_debug(
          state, opts, reference_result, token_mismatch_index - 1);
    }
    dump_generation_selected_step_stage_debug(state, opts, reference_result, token_mismatch_index);
    return;
  }

  const size_t layer0_row_width = static_cast<size_t>(state.model_data->params.n_embd);
  std::vector<float> reference_layer0_kqv_out_rows;
  std::vector<float> reference_layer0_attn_out_rows;
  std::vector<float> reference_layer0_ffn_norm_rows;
  std::vector<float> reference_layer0_ffn_out_rows;
  std::vector<float> reference_layer0_l_out_rows;
  std::vector<float> reference_layer1_kqv_out_rows;
  std::vector<float> reference_layer1_attn_out_rows;
  std::vector<float> reference_layer1_attn_norm_rows;
  std::vector<float> reference_layer1_ffn_inp_rows;
  std::vector<float> reference_layer1_ffn_norm_rows;
  std::vector<float> reference_layer1_ffn_out_rows;
  std::vector<float> reference_layer1_l_out_rows;
  auto build_reference_stage_rows = [&](const char * key, std::vector<float> & rows_out) {
    rows_out.clear();
    rows_out.reserve(prefix_tokens.size() * layer0_row_width);
    std::vector<int32_t> generated_prefix;
    generated_prefix.reserve(prefix_generated_tokens.size());
    for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
      reference_graph_capture token_capture = {};
      const bool is_prompt_token = token_index < prompt_tokens.size();
      std::span<const llama_token> prompt_prefix =
          is_prompt_token
              ? std::span<const llama_token>(prompt_tokens.data(), token_index + 1u)
              : std::span<const llama_token>(prompt_tokens.data(), prompt_tokens.size());
      generated_prefix.clear();
      if (!is_prompt_token) {
        const size_t generated_count = token_index - prompt_tokens.size() + 1u;
        generated_prefix.insert(generated_prefix.end(),
                                prefix_generated_tokens.begin(),
                                prefix_generated_tokens.begin() +
                                    static_cast<std::ptrdiff_t>(generated_count));
      }
      if (!capture_reference_graph_for_generation_prefix(
              state, prompt_prefix, generated_prefix, token_capture)) {
        return false;
      }
      const std::span<const float> row =
          reference_last_token_row(find_reference_tensor(token_capture, key), layer0_row_width);
      if (row.size() != layer0_row_width) {
        return false;
      }
      rows_out.insert(rows_out.end(), row.begin(), row.end());
    }
    return true;
  };
  if (!build_reference_stage_rows("kqv_out-0", reference_layer0_kqv_out_rows) ||
      !build_reference_stage_rows("attn_out-0", reference_layer0_attn_out_rows) ||
      !build_reference_stage_rows("ffn_norm-0", reference_layer0_ffn_norm_rows) ||
      !build_reference_stage_rows("ffn_out-0", reference_layer0_ffn_out_rows) ||
      !build_reference_stage_rows("l_out-0", reference_layer0_l_out_rows) ||
      !build_reference_stage_rows("kqv_out-1", reference_layer1_kqv_out_rows) ||
      !build_reference_stage_rows("attn_out-1", reference_layer1_attn_out_rows) ||
      !build_reference_stage_rows("attn_norm-1", reference_layer1_attn_norm_rows) ||
      !build_reference_stage_rows("ffn_inp-1", reference_layer1_ffn_inp_rows) ||
      !build_reference_stage_rows("ffn_norm-1", reference_layer1_ffn_norm_rows) ||
      !build_reference_stage_rows("ffn_out-1", reference_layer1_ffn_out_rows) ||
      !build_reference_stage_rows("l_out-1", reference_layer1_l_out_rows)) {
    std::fprintf(stdout, "generation_debug.flash: unable to capture reference graph\n");
    return;
  }
  const reference_stage_capture_set reference_stage_captures{
      .layer0_kqv_out = reference_layer0_kqv_out_rows,
      .layer0_attn_out = reference_layer0_attn_out_rows,
      .layer0_ffn_norm = reference_layer0_ffn_norm_rows,
      .layer0_ffn_out = reference_layer0_ffn_out_rows,
      .layer0_l_out = reference_layer0_l_out_rows,
      .layer1_kqv_out = reference_layer1_kqv_out_rows,
      .layer1_attn_out = reference_layer1_attn_out_rows,
      .layer1_attn_norm = reference_layer1_attn_norm_rows,
      .layer1_ffn_inp = reference_layer1_ffn_inp_rows,
      .layer1_ffn_norm = reference_layer1_ffn_norm_rows,
      .layer1_ffn_out = reference_layer1_ffn_out_rows,
      .layer1_l_out = reference_layer1_l_out_rows,
  };

  emel::generator::detail::native_backend dispatch_backend = {};
  emel::generator::detail::native_backend scalar_backend = {};
  emel::generator::detail::native_backend shared_backend = {};
  emel::generator::detail::native_backend f32_attention_backend = {};
  emel::generator::detail::native_backend rounded_weight_attention_backend = {};
  emel::generator::detail::native_backend ggml_f16_attention_backend = {};
  emel::generator::detail::native_backend ggml_online_f16_attention_backend = {};
  emel::generator::detail::native_backend ggml_nonflash_f16_attention_backend = {};
  emel::generator::detail::native_backend ggml_f16_scores_attention_backend = {};
  emel::generator::detail::native_backend ggml_f16_scores_ggml_softmax_attention_backend = {};
  emel::generator::detail::native_backend ggml_nonflash_f16_ggml_softmax_attention_backend = {};
  emel::generator::detail::native_backend double_softmax_sum_attention_backend = {};
  emel::generator::detail::native_backend ggml_softmax_attention_backend = {};
  emel::generator::detail::native_backend full_q_attention_backend = {};
  emel::generator::detail::native_backend full_q_f32_attention_backend = {};
  emel::generator::detail::native_backend full_q_rounded_weight_attention_backend = {};
  emel::generator::detail::native_backend full_q_ggml_f16_attention_backend = {};
  emel::generator::detail::native_backend exact_backend = {};
  emel::generator::detail::native_backend attention_exact_backend = {};
  emel::generator::detail::native_backend ffn_exact_backend = {};
  emel::generator::detail::native_backend output_exact_backend = {};
  emel::generator::detail::native_backend q2_exact_backend = {};
  emel::generator::detail::native_backend q3_exact_backend = {};
  emel::generator::detail::native_backend q6_exact_backend = {};
  emel::generator::detail::native_backend q2_scalar_quant_backend = {};
  emel::generator::detail::native_backend q3_scalar_quant_backend = {};
  emel::generator::detail::native_backend q23_scalar_quant_backend = {};
  emel::generator::detail::native_backend q236_scalar_quant_backend = {};
  emel::generator::detail::native_backend q6_scalar_quant_backend = {};
  emel::generator::detail::native_backend q2_reference_backend = {};
  emel::generator::detail::native_backend q3_reference_backend = {};
  emel::generator::detail::native_backend q236_reference_backend = {};
  emel::generator::detail::native_backend q6_reference_backend = {};
  emel::generator::detail::native_backend reference_key_cache_backend = {};
  emel::generator::detail::native_backend reference_value_cache_backend = {};
  emel::generator::detail::native_backend reference_kv_cache_backend = {};
  emel::generator::detail::native_backend reference_layer0_kqv_out_backend = {};
  emel::generator::detail::native_backend reference_layer0_kqv_out_exact_attention_backend = {};
  emel::generator::detail::native_backend reference_layer0_kqv_out_reference_q8_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_shared_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_q2_scalar_quant_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_q3_scalar_quant_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_q6_scalar_quant_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_q236_scalar_quant_backend = {};
  emel::generator::detail::native_backend reference_layer0_attn_out_q236_reference_backend = {};
  emel::generator::detail::native_backend reference_layer0_ffn_norm_backend = {};
  emel::generator::detail::native_backend reference_layer0_ffn_out_backend = {};
  emel::generator::detail::native_backend reference_layer0_l_out_backend = {};
  emel::generator::detail::native_backend reference_layer1_kqv_out_backend = {};
  emel::generator::detail::native_backend reference_layer1_attn_out_backend = {};
  emel::generator::detail::native_backend reference_layer1_attn_norm_backend = {};
  emel::generator::detail::native_backend reference_layer1_ffn_inp_backend = {};
  emel::generator::detail::native_backend reference_layer1_ffn_norm_backend = {};
  emel::generator::detail::native_backend reference_layer1_ffn_out_backend = {};
  emel::generator::detail::native_backend reference_layer1_l_out_backend = {};
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
  const exact_matmul_mode scalar_quant_q23{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)),
      .use_scalar_quantized = true,
  };
  const exact_matmul_mode scalar_quant_q236{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q6_k)),
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
  const exact_matmul_mode reference_q236{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q6_k)),
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
      emel::generator::detail::prepare(f32_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(rounded_weight_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ggml_f16_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ggml_online_f16_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ggml_nonflash_f16_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ggml_f16_scores_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          ggml_f16_scores_ggml_softmax_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          ggml_nonflash_f16_ggml_softmax_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(double_softmax_sum_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(ggml_softmax_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(full_q_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(full_q_f32_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(full_q_rounded_weight_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(full_q_ggml_f16_attention_backend, *state.model_data) !=
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
      emel::generator::detail::prepare(q23_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q236_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q6_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q236_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q6_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_key_cache_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_value_cache_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_kv_cache_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer0_kqv_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_kqv_out_exact_attention_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_kqv_out_reference_q8_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer0_attn_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_attn_out_q2_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_attn_out_q3_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_attn_out_q6_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_attn_out_q236_scalar_quant_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(
          reference_layer0_attn_out_q236_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer0_ffn_norm_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer0_ffn_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer0_l_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_kqv_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_attn_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_attn_norm_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_ffn_inp_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_ffn_norm_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_ffn_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_layer1_l_out_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      !run_prefill_from_token_prefix(dispatch_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention(scalar_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_no_weight_rounding(
          f32_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_rounded_weight_scalar(
          rounded_weight_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_f16_value_contraction(
          ggml_f16_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_online_f16(
          ggml_online_f16_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_nonflash_f16(
          ggml_nonflash_f16_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_f16_scores(
          ggml_f16_scores_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_f16_scores_ggml_softmax(
          ggml_f16_scores_ggml_softmax_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_nonflash_f16_ggml_softmax(
          ggml_nonflash_f16_ggml_softmax_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_double_softmax_sum(
          double_softmax_sum_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_ggml_softmax(
          ggml_softmax_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_full_q(full_q_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_full_q_no_weight_rounding(
          full_q_f32_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_full_q_rounded_weight(
          full_q_rounded_weight_attention_backend, prefix_tokens) ||
      !run_prefill_with_scalar_attention_full_q_ggml_f16_value_contraction(
          full_q_ggml_f16_attention_backend, prefix_tokens) ||
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
          q23_scalar_quant_backend, prefix_tokens, scalar_quant_q23) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q236_scalar_quant_backend, prefix_tokens, scalar_quant_q236) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q6_scalar_quant_backend, prefix_tokens, scalar_quant_q6_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q2_reference_backend, prefix_tokens, reference_q2_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q3_reference_backend, prefix_tokens, reference_q3_only) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q236_reference_backend, prefix_tokens, reference_q236) ||
      !run_prefill_with_scalar_attention_matmul_mode(
          q6_reference_backend, prefix_tokens, reference_q6_only) ||
      !run_prefill_with_scalar_attention_reference_cache(
          reference_key_cache_backend, const_cast<initialize_backend &>(state.backend), prefix_tokens, true, false) ||
      !run_prefill_with_scalar_attention_reference_cache(
          reference_value_cache_backend,
          const_cast<initialize_backend &>(state.backend),
          prefix_tokens,
          false,
          true) ||
      !run_prefill_with_scalar_attention_reference_cache(
          reference_kv_cache_backend,
          const_cast<initialize_backend &>(state.backend),
          prefix_tokens,
          true,
          true) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_kqv_out_backend,
          prefix_tokens,
          reference_stage_injection::layer0_kqv_out,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_kqv_out_exact_attention_backend,
          prefix_tokens,
          reference_stage_injection::layer0_kqv_out,
          reference_stage_captures,
          attention_projection_override::exact) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_kqv_out_reference_q8_backend,
          prefix_tokens,
          reference_stage_injection::layer0_kqv_out,
          reference_stage_captures,
          attention_projection_override::reference_q8) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_attn_out_backend,
          prefix_tokens,
          reference_stage_injection::layer0_attn_out,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_attn_out_q2_scalar_quant_backend,
          prefix_tokens,
          reference_stage_injection::layer0_attn_out,
          reference_stage_captures,
          attention_projection_override::native,
          scalar_quant_q2_only) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_attn_out_q3_scalar_quant_backend,
          prefix_tokens,
          reference_stage_injection::layer0_attn_out,
          reference_stage_captures,
          attention_projection_override::native,
          scalar_quant_q3_only) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_attn_out_q6_scalar_quant_backend,
          prefix_tokens,
          reference_stage_injection::layer0_attn_out,
          reference_stage_captures,
          attention_projection_override::native,
          scalar_quant_q6_only) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_attn_out_q236_scalar_quant_backend,
          prefix_tokens,
          reference_stage_injection::layer0_attn_out,
          reference_stage_captures,
          attention_projection_override::native,
          scalar_quant_q236) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_attn_out_q236_reference_backend,
          prefix_tokens,
          reference_stage_injection::layer0_attn_out,
          reference_stage_captures,
          attention_projection_override::native,
          reference_q236) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_ffn_norm_backend,
          prefix_tokens,
          reference_stage_injection::layer0_ffn_norm,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer0_ffn_out_backend,
          prefix_tokens,
          reference_stage_injection::layer0_ffn_out,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(reference_layer0_l_out_backend,
                                                        prefix_tokens,
                                                        reference_stage_injection::layer0_l_out,
                                                        reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer1_kqv_out_backend,
          prefix_tokens,
          reference_stage_injection::layer1_kqv_out,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer1_attn_out_backend,
          prefix_tokens,
          reference_stage_injection::layer1_attn_out,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer1_attn_norm_backend,
          prefix_tokens,
          reference_stage_injection::layer1_attn_norm,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer1_ffn_inp_backend,
          prefix_tokens,
          reference_stage_injection::layer1_ffn_inp,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer1_ffn_norm_backend,
          prefix_tokens,
          reference_stage_injection::layer1_ffn_norm,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(
          reference_layer1_ffn_out_backend,
          prefix_tokens,
          reference_stage_injection::layer1_ffn_out,
          reference_stage_captures) ||
      !run_prefill_with_scalar_attention_reference_stage(reference_layer1_l_out_backend,
                                                        prefix_tokens,
                                                        reference_stage_injection::layer1_l_out,
                                                        reference_stage_captures)) {
    std::fprintf(stdout, "generation_debug.flash: unable to replay mismatch prefix\n");
    return;
  }
  shared_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  shared_backend.kernel.set_kind(shared_backend.kernel_kind);
  if (!run_prefill_from_token_prefix(shared_backend, prefix_tokens)) {
    std::fprintf(stdout, "generation_debug.flash: unable to replay shared backend prefix\n");
    return;
  }
  if (emel::generator::detail::prepare(reference_layer0_attn_out_shared_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.flash: unable to prepare shared attn_out replay\n");
    return;
  }
  reference_layer0_attn_out_shared_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  reference_layer0_attn_out_shared_backend.kernel.set_kind(
      reference_layer0_attn_out_shared_backend.kernel_kind);
  if (!run_prefill_with_scalar_attention_reference_stage(reference_layer0_attn_out_shared_backend,
                                                         prefix_tokens,
                                                         reference_stage_injection::layer0_attn_out,
                                                         reference_stage_captures)) {
    std::fprintf(stdout, "generation_debug.flash: unable to replay shared attn_out prefix\n");
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
  const argmax_summary f32_attention_summary = select_argmax_from_logits(
      f32_attention_backend.bound_logits.data(), f32_attention_backend.n_vocab);
  const argmax_summary rounded_weight_attention_summary = select_argmax_from_logits(
      rounded_weight_attention_backend.bound_logits.data(),
      rounded_weight_attention_backend.n_vocab);
  const argmax_summary ggml_f16_attention_summary = select_argmax_from_logits(
      ggml_f16_attention_backend.bound_logits.data(), ggml_f16_attention_backend.n_vocab);
  const argmax_summary ggml_online_f16_attention_summary = select_argmax_from_logits(
      ggml_online_f16_attention_backend.bound_logits.data(),
      ggml_online_f16_attention_backend.n_vocab);
  const argmax_summary ggml_nonflash_f16_attention_summary = select_argmax_from_logits(
      ggml_nonflash_f16_attention_backend.bound_logits.data(),
      ggml_nonflash_f16_attention_backend.n_vocab);
  const argmax_summary ggml_f16_scores_attention_summary = select_argmax_from_logits(
      ggml_f16_scores_attention_backend.bound_logits.data(),
      ggml_f16_scores_attention_backend.n_vocab);
  const argmax_summary ggml_f16_scores_ggml_softmax_attention_summary = select_argmax_from_logits(
      ggml_f16_scores_ggml_softmax_attention_backend.bound_logits.data(),
      ggml_f16_scores_ggml_softmax_attention_backend.n_vocab);
  const argmax_summary ggml_nonflash_f16_ggml_softmax_attention_summary =
      select_argmax_from_logits(
          ggml_nonflash_f16_ggml_softmax_attention_backend.bound_logits.data(),
          ggml_nonflash_f16_ggml_softmax_attention_backend.n_vocab);
  const argmax_summary double_softmax_sum_attention_summary = select_argmax_from_logits(
      double_softmax_sum_attention_backend.bound_logits.data(),
      double_softmax_sum_attention_backend.n_vocab);
  const argmax_summary ggml_softmax_attention_summary = select_argmax_from_logits(
      ggml_softmax_attention_backend.bound_logits.data(),
      ggml_softmax_attention_backend.n_vocab);
  const argmax_summary full_q_attention_summary = select_argmax_from_logits(
      full_q_attention_backend.bound_logits.data(), full_q_attention_backend.n_vocab);
  const argmax_summary full_q_f32_attention_summary = select_argmax_from_logits(
      full_q_f32_attention_backend.bound_logits.data(),
      full_q_f32_attention_backend.n_vocab);
  const argmax_summary full_q_rounded_weight_attention_summary = select_argmax_from_logits(
      full_q_rounded_weight_attention_backend.bound_logits.data(),
      full_q_rounded_weight_attention_backend.n_vocab);
  const argmax_summary full_q_ggml_f16_attention_summary = select_argmax_from_logits(
      full_q_ggml_f16_attention_backend.bound_logits.data(),
      full_q_ggml_f16_attention_backend.n_vocab);
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
  const argmax_summary q23_scalar_quant_summary = select_argmax_from_logits(
      q23_scalar_quant_backend.bound_logits.data(), q23_scalar_quant_backend.n_vocab);
  const argmax_summary q236_scalar_quant_summary = select_argmax_from_logits(
      q236_scalar_quant_backend.bound_logits.data(), q236_scalar_quant_backend.n_vocab);
  const argmax_summary q6_scalar_quant_summary = select_argmax_from_logits(
      q6_scalar_quant_backend.bound_logits.data(), q6_scalar_quant_backend.n_vocab);
  const argmax_summary q2_reference_summary = select_argmax_from_logits(
      q2_reference_backend.bound_logits.data(), q2_reference_backend.n_vocab);
  const argmax_summary q3_reference_summary = select_argmax_from_logits(
      q3_reference_backend.bound_logits.data(), q3_reference_backend.n_vocab);
  const argmax_summary q236_reference_summary = select_argmax_from_logits(
      q236_reference_backend.bound_logits.data(), q236_reference_backend.n_vocab);
  const argmax_summary q6_reference_summary = select_argmax_from_logits(
      q6_reference_backend.bound_logits.data(), q6_reference_backend.n_vocab);
  const argmax_summary reference_key_cache_summary = select_argmax_from_logits(
      reference_key_cache_backend.bound_logits.data(), reference_key_cache_backend.n_vocab);
  const argmax_summary reference_value_cache_summary = select_argmax_from_logits(
      reference_value_cache_backend.bound_logits.data(), reference_value_cache_backend.n_vocab);
  const argmax_summary reference_kv_cache_summary = select_argmax_from_logits(
      reference_kv_cache_backend.bound_logits.data(), reference_kv_cache_backend.n_vocab);
  const argmax_summary reference_layer0_kqv_out_summary = select_argmax_from_logits(
      reference_layer0_kqv_out_backend.bound_logits.data(),
      reference_layer0_kqv_out_backend.n_vocab);
  const argmax_summary reference_layer0_kqv_out_exact_attention_summary =
      select_argmax_from_logits(reference_layer0_kqv_out_exact_attention_backend.bound_logits.data(),
                                reference_layer0_kqv_out_exact_attention_backend.n_vocab);
  const argmax_summary reference_layer0_kqv_out_reference_q8_summary =
      select_argmax_from_logits(reference_layer0_kqv_out_reference_q8_backend.bound_logits.data(),
                                reference_layer0_kqv_out_reference_q8_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_summary = select_argmax_from_logits(
      reference_layer0_attn_out_backend.bound_logits.data(),
      reference_layer0_attn_out_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_shared_summary = select_argmax_from_logits(
      reference_layer0_attn_out_shared_backend.bound_logits.data(),
      reference_layer0_attn_out_shared_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_q2_scalar_quant_summary =
      select_argmax_from_logits(reference_layer0_attn_out_q2_scalar_quant_backend.bound_logits.data(),
                                reference_layer0_attn_out_q2_scalar_quant_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_q3_scalar_quant_summary =
      select_argmax_from_logits(reference_layer0_attn_out_q3_scalar_quant_backend.bound_logits.data(),
                                reference_layer0_attn_out_q3_scalar_quant_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_q6_scalar_quant_summary =
      select_argmax_from_logits(reference_layer0_attn_out_q6_scalar_quant_backend.bound_logits.data(),
                                reference_layer0_attn_out_q6_scalar_quant_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_q236_scalar_quant_summary =
      select_argmax_from_logits(
          reference_layer0_attn_out_q236_scalar_quant_backend.bound_logits.data(),
          reference_layer0_attn_out_q236_scalar_quant_backend.n_vocab);
  const argmax_summary reference_layer0_attn_out_q236_reference_summary =
      select_argmax_from_logits(reference_layer0_attn_out_q236_reference_backend.bound_logits.data(),
                                reference_layer0_attn_out_q236_reference_backend.n_vocab);
  const argmax_summary reference_layer0_ffn_norm_summary = select_argmax_from_logits(
      reference_layer0_ffn_norm_backend.bound_logits.data(),
      reference_layer0_ffn_norm_backend.n_vocab);
  const argmax_summary reference_layer0_ffn_out_summary = select_argmax_from_logits(
      reference_layer0_ffn_out_backend.bound_logits.data(),
      reference_layer0_ffn_out_backend.n_vocab);
  const argmax_summary reference_layer0_l_out_summary = select_argmax_from_logits(
      reference_layer0_l_out_backend.bound_logits.data(),
      reference_layer0_l_out_backend.n_vocab);
  const argmax_summary reference_layer1_kqv_out_summary = select_argmax_from_logits(
      reference_layer1_kqv_out_backend.bound_logits.data(),
      reference_layer1_kqv_out_backend.n_vocab);
  const argmax_summary reference_layer1_attn_out_summary = select_argmax_from_logits(
      reference_layer1_attn_out_backend.bound_logits.data(),
      reference_layer1_attn_out_backend.n_vocab);
  const argmax_summary reference_layer1_attn_norm_summary = select_argmax_from_logits(
      reference_layer1_attn_norm_backend.bound_logits.data(),
      reference_layer1_attn_norm_backend.n_vocab);
  const argmax_summary reference_layer1_ffn_inp_summary = select_argmax_from_logits(
      reference_layer1_ffn_inp_backend.bound_logits.data(),
      reference_layer1_ffn_inp_backend.n_vocab);
  const argmax_summary reference_layer1_ffn_norm_summary = select_argmax_from_logits(
      reference_layer1_ffn_norm_backend.bound_logits.data(),
      reference_layer1_ffn_norm_backend.n_vocab);
  const argmax_summary reference_layer1_ffn_out_summary = select_argmax_from_logits(
      reference_layer1_ffn_out_backend.bound_logits.data(),
      reference_layer1_ffn_out_backend.n_vocab);
  const argmax_summary reference_layer1_l_out_summary = select_argmax_from_logits(
      reference_layer1_l_out_backend.bound_logits.data(),
      reference_layer1_l_out_backend.n_vocab);
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
               "shared_argmax=%d f32_attn_argmax=%d rounded_weight_attn_argmax=%d ggml_f16_attn_argmax=%d ggml_online_f16_attn_argmax=%d ggml_nonflash_f16_attn_argmax=%d ggml_f16_scores_attn_argmax=%d ggml_f16_scores_ggml_softmax_argmax=%d ggml_nonflash_f16_ggml_softmax_argmax=%d double_softmax_argmax=%d ggml_softmax_argmax=%d full_q_attn_argmax=%d full_q_f32_attn_argmax=%d full_q_rounded_weight_attn_argmax=%d full_q_ggml_f16_attn_argmax=%d exact_argmax=%d attention_exact_argmax=%d "
               "ffn_exact_argmax=%d output_exact_argmax=%d q2_exact_argmax=%d "
               "q3_exact_argmax=%d q6_exact_argmax=%d q2_scalar_argmax=%d q3_scalar_argmax=%d "
               "q23_scalar_argmax=%d q236_scalar_argmax=%d q6_scalar_argmax=%d "
               "q2_reference_argmax=%d q3_reference_argmax=%d q236_reference_argmax=%d "
               "q6_reference_argmax=%d reference_key_cache_argmax=%d "
               "reference_value_cache_argmax=%d reference_kv_cache_argmax=%d "
               "reference_layer0_kqv_out_argmax=%d "
               "reference_layer0_kqv_out_exact_attn_argmax=%d "
               "reference_layer0_kqv_out_reference_q8_argmax=%d "
               "reference_layer0_attn_out_argmax=%d "
               "reference_layer0_attn_out_shared_argmax=%d "
               "reference_layer0_attn_out_q2_scalar_argmax=%d "
               "reference_layer0_attn_out_q3_scalar_argmax=%d "
               "reference_layer0_attn_out_q6_scalar_argmax=%d "
               "reference_layer0_attn_out_q236_scalar_argmax=%d "
               "reference_layer0_attn_out_q236_reference_argmax=%d "
               "reference_layer0_ffn_norm_argmax=%d "
               "reference_layer0_ffn_out_argmax=%d "
               "reference_layer0_l_out_argmax=%d "
               "reference_layer1_kqv_out_argmax=%d "
               "reference_layer1_attn_out_argmax=%d "
               "reference_layer1_attn_norm_argmax=%d "
               "reference_layer1_ffn_inp_argmax=%d "
               "reference_layer1_ffn_norm_argmax=%d "
               "reference_layer1_ffn_out_argmax=%d "
               "reference_layer1_l_out_argmax=%d reference_token=%d\n",
               prefix_tokens.size(),
               dispatch_summary.selected_token,
               scalar_summary.selected_token,
               shared_summary.selected_token,
               f32_attention_summary.selected_token,
               rounded_weight_attention_summary.selected_token,
               ggml_f16_attention_summary.selected_token,
               ggml_online_f16_attention_summary.selected_token,
               ggml_nonflash_f16_attention_summary.selected_token,
               ggml_f16_scores_attention_summary.selected_token,
               ggml_f16_scores_ggml_softmax_attention_summary.selected_token,
               ggml_nonflash_f16_ggml_softmax_attention_summary.selected_token,
               double_softmax_sum_attention_summary.selected_token,
               ggml_softmax_attention_summary.selected_token,
               full_q_attention_summary.selected_token,
               full_q_f32_attention_summary.selected_token,
               full_q_rounded_weight_attention_summary.selected_token,
               full_q_ggml_f16_attention_summary.selected_token,
               exact_summary.selected_token,
               attention_exact_summary.selected_token,
               ffn_exact_summary.selected_token,
               output_exact_summary.selected_token,
               q2_exact_summary.selected_token,
               q3_exact_summary.selected_token,
               q6_exact_summary.selected_token,
               q2_scalar_quant_summary.selected_token,
               q3_scalar_quant_summary.selected_token,
               q23_scalar_quant_summary.selected_token,
               q236_scalar_quant_summary.selected_token,
               q6_scalar_quant_summary.selected_token,
               q2_reference_summary.selected_token,
               q3_reference_summary.selected_token,
               q236_reference_summary.selected_token,
               q6_reference_summary.selected_token,
               reference_key_cache_summary.selected_token,
               reference_value_cache_summary.selected_token,
               reference_kv_cache_summary.selected_token,
               reference_layer0_kqv_out_summary.selected_token,
               reference_layer0_kqv_out_exact_attention_summary.selected_token,
               reference_layer0_kqv_out_reference_q8_summary.selected_token,
               reference_layer0_attn_out_summary.selected_token,
               reference_layer0_attn_out_shared_summary.selected_token,
               reference_layer0_attn_out_q2_scalar_quant_summary.selected_token,
               reference_layer0_attn_out_q3_scalar_quant_summary.selected_token,
               reference_layer0_attn_out_q6_scalar_quant_summary.selected_token,
               reference_layer0_attn_out_q236_scalar_quant_summary.selected_token,
               reference_layer0_attn_out_q236_reference_summary.selected_token,
               reference_layer0_ffn_norm_summary.selected_token,
               reference_layer0_ffn_out_summary.selected_token,
               reference_layer0_l_out_summary.selected_token,
               reference_layer1_kqv_out_summary.selected_token,
               reference_layer1_attn_out_summary.selected_token,
               reference_layer1_attn_norm_summary.selected_token,
               reference_layer1_ffn_inp_summary.selected_token,
               reference_layer1_ffn_norm_summary.selected_token,
               reference_layer1_ffn_out_summary.selected_token,
               reference_layer1_l_out_summary.selected_token,
               reference_token);
  const reference_stage_replay_diff attn_out_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer0_attn_out_backend,
                                        prefix_tokens,
                                        0,
                                        reference_stage_injection::layer0_attn_out,
                                        reference_stage_captures);
  const reference_stage_replay_diff attn_out_q236_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer0_attn_out_q236_reference_backend,
                                        prefix_tokens,
                                        0,
                                        reference_stage_injection::layer0_attn_out,
                                        reference_stage_captures,
                                        attention_projection_override::native,
                                        reference_q236);
  const reference_stage_replay_diff ffn_norm_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer0_ffn_norm_backend,
                                        prefix_tokens,
                                        0,
                                        reference_stage_injection::layer0_ffn_norm,
                                        reference_stage_captures);
  const reference_stage_replay_diff ffn_out_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer0_ffn_out_backend,
                                        prefix_tokens,
                                        0,
                                        reference_stage_injection::layer0_ffn_out,
                                        reference_stage_captures);
  const reference_stage_replay_diff layer1_attn_out_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer1_attn_out_backend,
                                        prefix_tokens,
                                        1,
                                        reference_stage_injection::layer1_attn_out,
                                        reference_stage_captures);
  const reference_stage_replay_diff layer1_attn_norm_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer1_attn_norm_backend,
                                        prefix_tokens,
                                        1,
                                        reference_stage_injection::layer1_attn_norm,
                                        reference_stage_captures);
  const reference_stage_replay_diff layer1_ffn_inp_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer1_ffn_inp_backend,
                                        prefix_tokens,
                                        1,
                                        reference_stage_injection::layer1_ffn_inp,
                                        reference_stage_captures);
  const reference_stage_replay_diff layer1_ffn_norm_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer1_ffn_norm_backend,
                                        prefix_tokens,
                                        1,
                                        reference_stage_injection::layer1_ffn_norm,
                                        reference_stage_captures);
  const reference_stage_replay_diff layer1_ffn_out_replay_diff =
      replay_reference_stage_l_out_diff(reference_layer1_ffn_out_backend,
                                        prefix_tokens,
                                        1,
                                        reference_stage_injection::layer1_ffn_out,
                                        reference_stage_captures);
  std::fprintf(stdout,
               "generation_debug.flash.layer0_replay: attn_out_first_l_out_mismatch=%d "
               "attn_out_max_abs=%g attn_out_q236_first_l_out_mismatch=%d "
               "attn_out_q236_max_abs=%g ffn_norm_first_l_out_mismatch=%d ffn_norm_max_abs=%g "
               "ffn_out_first_l_out_mismatch=%d ffn_out_max_abs=%g\n",
               attn_out_replay_diff.first_target_l_out_mismatch,
               attn_out_replay_diff.max_abs,
               attn_out_q236_replay_diff.first_target_l_out_mismatch,
               attn_out_q236_replay_diff.max_abs,
               ffn_norm_replay_diff.first_target_l_out_mismatch,
               ffn_norm_replay_diff.max_abs,
               ffn_out_replay_diff.first_target_l_out_mismatch,
               ffn_out_replay_diff.max_abs);
  std::fprintf(stdout,
               "generation_debug.flash.layer1_replay: attn_out_first_l_out_mismatch=%d "
               "attn_out_max_abs=%g attn_norm_first_l_out_mismatch=%d "
               "attn_norm_max_abs=%g ffn_inp_first_l_out_mismatch=%d "
               "ffn_inp_max_abs=%g ffn_norm_first_l_out_mismatch=%d ffn_norm_max_abs=%g "
               "ffn_out_first_l_out_mismatch=%d ffn_out_max_abs=%g\n",
               layer1_attn_out_replay_diff.first_target_l_out_mismatch,
               layer1_attn_out_replay_diff.max_abs,
               layer1_attn_norm_replay_diff.first_target_l_out_mismatch,
               layer1_attn_norm_replay_diff.max_abs,
               layer1_ffn_inp_replay_diff.first_target_l_out_mismatch,
               layer1_ffn_inp_replay_diff.max_abs,
               layer1_ffn_norm_replay_diff.first_target_l_out_mismatch,
               layer1_ffn_norm_replay_diff.max_abs,
               layer1_ffn_out_replay_diff.first_target_l_out_mismatch,
               layer1_ffn_out_replay_diff.max_abs);
  dump_prompt0_reference_attn_out_ffn_debug(state, prompt_tokens);
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
  const auto dump_alt_generation = [&](const char * label, const native_layer_runner run_layer_fn) {
    generation_result alt_result = {};
    const emel::error::type alt_err = run_custom_native_generate(
        state.backend, *state.model_data, opts, run_layer_fn, alt_result);
    if (alt_err == emel::error::cast(emel::generator::error::none)) {
      std::fprintf(stdout,
                   "generation_debug.alt.%s: match=%d token_mismatch=%d "
                   "byte_mismatch=%zu tokens=%d output_length=%zu\n",
                   label,
                   generation_results_match(alt_result, reference_result) ? 1 : 0,
                   first_token_mismatch_index(alt_result, reference_result),
                   first_mismatch_offset(alt_result, reference_result),
                   alt_result.tokens_generated,
                   alt_result.output_length);
      return;
    }

    std::fprintf(stdout,
                 "generation_debug.alt.%s: error=%d\n",
                 label,
                 static_cast<int>(alt_err));
  };
  dump_alt_generation("runtime_flash_q_full", run_layer_with_flash_request_q);
  dump_alt_generation("runtime_flash_q_attn_full", run_layer_with_flash_request_q_attn);
  dump_alt_generation("emel_prod_style_full", run_layer_with_scalar_attention_emel_prod_style);
  dump_alt_generation("ggml_nonflash_exact_masked_full",
                      run_layer_with_scalar_attention_ggml_nonflash_exact_masked);
  dump_alt_generation("ggml_nonflash_exact_masked_full_q",
                      run_layer_with_scalar_attention_ggml_nonflash_exact_masked_full_q);
  dump_alt_generation("ggml_nonflash_exact_scores_prod_value_full",
                      run_layer_with_scalar_attention_ggml_nonflash_exact_scores_prod_value);
  dump_alt_generation("ggml_f16_scores_full",
                      run_layer_with_scalar_attention_ggml_f16_scores);
  dump_alt_generation("full_q_full", run_layer_with_scalar_attention_full_q);
  dump_alt_generation("full_q_ggml_f16_full",
                      run_layer_with_scalar_attention_full_q_ggml_f16_value_contraction);
  dump_alt_generation("exact_attention_full",
                      run_layer_with_exact_attention_scalar_attention);
  dump_alt_generation("exact_ffn_full", run_layer_with_exact_ffn_scalar_attention);
  dump_alt_generation("exact_output_full", run_layer_with_exact_output_scalar_attention);
  dump_alt_generation("reference_q236_full",
                      run_layer_with_reference_q236_scalar_attention);
  dump_alt_generation("scalar_q236_full",
                      run_layer_with_scalar_q236_scalar_attention);
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
      ggml_vec_dot_q2_K_q8_K_generic(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    case emel::kernel::event::dtype::q3_k:
      ggml_vec_dot_q3_K_q8_K_generic(
          static_cast<int>(block_count * kernel_quant::QK_K), &out, 0, row_ptr, 0, q8_blocks, 0, 1);
      return out;
    case emel::kernel::event::dtype::q6_k:
      ggml_vec_dot_q6_K_q8_K_generic(
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
  const int32_t batch_capacity = std::max(512, state.model_data->params.n_ctx);
  context_params.n_batch = batch_capacity;
  context_params.n_ubatch = batch_capacity;
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
  const int32_t batch_capacity = std::max(512, state.model_data->params.n_ctx);
  context_params.n_batch = batch_capacity;
  context_params.n_ubatch = batch_capacity;
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

void dump_state_compare(const char * label,
                        std::span<const uint16_t> emel_values,
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
  float max_emel = 0.0f;
  float max_reference = 0.0f;
  for (int32_t idx = 0; idx < static_cast<int32_t>(emel_values.size()); ++idx) {
    const float emel_value = fp16_storage_to_fp32(emel_values[static_cast<size_t>(idx)]);
    const float reference_value = reference_values[static_cast<size_t>(idx)];
    const float diff = std::fabs(emel_value - reference_value);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
      max_emel = emel_value;
      max_reference = reference_value;
    }
  }

  std::fprintf(stdout,
               "%s: max_abs=%g idx=%d emel=%g reference=%g\n",
               label,
               max_abs,
               max_idx,
               max_emel,
               max_reference);
}

void dump_state_compare(const char * label,
                        std::span<const float> emel_values,
                        std::span<const uint16_t> reference_values) {
  dump_state_compare(label, reference_values, emel_values);
}

void dump_state_compare(const char * label,
                        std::span<const uint16_t> emel_values,
                        std::span<const uint16_t> reference_values) {
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
  float max_emel = 0.0f;
  float max_reference = 0.0f;
  for (int32_t idx = 0; idx < static_cast<int32_t>(emel_values.size()); ++idx) {
    const float emel_value = fp16_storage_to_fp32(emel_values[static_cast<size_t>(idx)]);
    const float reference_value = fp16_storage_to_fp32(reference_values[static_cast<size_t>(idx)]);
    const float diff = std::fabs(emel_value - reference_value);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = idx;
      max_emel = emel_value;
      max_reference = reference_value;
    }
  }

  std::fprintf(stdout,
               "%s: max_abs=%g idx=%d emel=%g reference=%g\n",
               label,
               max_abs,
               max_idx,
               max_emel,
               max_reference);
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
  const int64_t ne3 = std::max<int64_t>(capture.shape[3], 1);
  if (token_index < 0 || head_index < 0 || token_index >= ne3 || head_index >= ne2) {
    return {};
  }

  const size_t offset =
      (((static_cast<size_t>(token_index) * static_cast<size_t>(ne2)) +
        static_cast<size_t>(head_index)) *
       static_cast<size_t>(ne1)) *
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
  const std::string kq_key = "kq-" + std::to_string(layer_index);
  const reference_tensor_capture * kq_capture =
      find_reference_capture(graph_capture, kq_key.c_str());
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
  float kq_max_abs = 0.0f;
  int32_t kq_max_head = 0;
  int32_t kq_max_pos = 0;
  float kq_emel_at_max = 0.0f;
  float kq_reference_at_max = 0.0f;
  float full_q_kq_max_abs = 0.0f;
  int32_t full_q_kq_max_head = 0;
  int32_t full_q_kq_max_pos = 0;
  float full_q_kq_emel_at_max = 0.0f;
  float full_q_kq_reference_at_max = 0.0f;
  float ggml_f16_kq_max_abs = 0.0f;
  int32_t ggml_f16_kq_max_head = 0;
  int32_t ggml_f16_kq_max_pos = 0;
  float ggml_f16_kq_emel_at_max = 0.0f;
  float ggml_f16_kq_reference_at_max = 0.0f;
  float full_q_max_abs = 0.0f;
  int32_t full_q_max_head = 0;
  int32_t full_q_max_pos = 0;
  float full_q_emel_at_max = 0.0f;
  float full_q_reference_at_max = 0.0f;
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
        const float cached_value = fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(idx)]);
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
    std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
    std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));
    std::vector<float> full_q_scores(static_cast<size_t>(position_limit), 0.0f);
    const std::span<const float> reference_kq =
        kq_capture != nullptr
            ? reference_softmax_query_head_slice(*kq_capture, head, 0)
            : std::span<const float>{};

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] =
          ggml_fp32_to_fp16(backend.q_attn[q_offset + static_cast<size_t>(dim)]);
    }

    float max_score = -std::numeric_limits<float>::infinity();
    float full_q_max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, position, kv_dim) +
          kv_offset;
      float raw_score = 0.0f;
      float full_q_raw_score = 0.0f;
      float full_q_score = 0.0f;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        raw_score += backend.q_attn[q_offset + static_cast<size_t>(dim)] *
                     fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
        full_q_raw_score += backend.q[q_offset + static_cast<size_t>(dim)] *
                            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
        full_q_score += backend.q[q_offset + static_cast<size_t>(dim)] *
                        fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]);
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(
            fp16_storage_to_fp32(backend.key_cache[cache_offset + static_cast<size_t>(dim)]));
      }
      if (static_cast<size_t>(position) < reference_kq.size()) {
        const float raw_diff =
            std::fabs(raw_score - reference_kq[static_cast<size_t>(position)]);
        if (raw_diff > kq_max_abs) {
          kq_max_abs = raw_diff;
          kq_max_head = head;
          kq_max_pos = position;
          kq_emel_at_max = raw_score;
          kq_reference_at_max = reference_kq[static_cast<size_t>(position)];
        }
        const float full_q_raw_diff =
            std::fabs(full_q_raw_score - reference_kq[static_cast<size_t>(position)]);
        if (full_q_raw_diff > full_q_kq_max_abs) {
          full_q_kq_max_abs = full_q_raw_diff;
          full_q_kq_max_head = head;
          full_q_kq_max_pos = position;
          full_q_kq_emel_at_max = full_q_raw_score;
          full_q_kq_reference_at_max = reference_kq[static_cast<size_t>(position)];
        }
        float ggml_f16_raw_score = 0.0f;
        ggml_vec_dot_f16(head_dim,
                         &ggml_f16_raw_score,
                         0u,
                         k_f16.data(),
                         0u,
                         q_f16.data(),
                         0u,
                         1);
        const float ggml_f16_raw_diff =
            std::fabs(ggml_f16_raw_score - reference_kq[static_cast<size_t>(position)]);
        if (ggml_f16_raw_diff > ggml_f16_kq_max_abs) {
          ggml_f16_kq_max_abs = ggml_f16_raw_diff;
          ggml_f16_kq_max_head = head;
          ggml_f16_kq_max_pos = position;
          ggml_f16_kq_emel_at_max = ggml_f16_raw_score;
          ggml_f16_kq_reference_at_max = reference_kq[static_cast<size_t>(position)];
        }
      }
      const float score = raw_score * inv_scale;
      full_q_score *= inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      full_q_scores[static_cast<size_t>(position)] = full_q_score;
      max_score = std::max(max_score, score);
      full_q_max_score = std::max(full_q_max_score, full_q_score);
    }

    float score_sum = 0.0f;
    float full_q_score_sum = 0.0f;
    for (int32_t position = 0; position < position_limit; ++position) {
      const float prob = std::exp(backend.attn_scores[static_cast<size_t>(position)] - max_score);
      const float full_q_prob =
          std::exp(full_q_scores[static_cast<size_t>(position)] - full_q_max_score);
      backend.attn_probs[static_cast<size_t>(position)] = prob;
      score_sum += prob;
      full_q_score_sum += full_q_prob;
    }

    const std::span<const float> reference_probs =
        softmax_capture != nullptr
            ? reference_softmax_query_head_slice(*softmax_capture, head, 0)
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

        const float full_q_weight =
            full_q_score_sum == 0.0f
                ? 0.0f
                : std::exp(full_q_scores[static_cast<size_t>(position)] - full_q_max_score) /
                      full_q_score_sum;
        const float full_q_diff =
            std::fabs(full_q_weight - reference_probs[static_cast<size_t>(position)]);
        if (full_q_diff > full_q_max_abs) {
          full_q_max_abs = full_q_diff;
          full_q_max_head = head;
          full_q_max_pos = position;
          full_q_emel_at_max = full_q_weight;
          full_q_reference_at_max = reference_probs[static_cast<size_t>(position)];
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
        const float cached_value = fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
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
            fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]));
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
    if (kq_capture != nullptr) {
      std::fprintf(stdout,
                   "%s.kq: max_abs=%g head=%d pos=%d emel=%g reference=%g\n",
                   layer_prefix.c_str(),
                   kq_max_abs,
                   kq_max_head,
                   kq_max_pos,
                   kq_emel_at_max,
                   kq_reference_at_max);
      std::fprintf(stdout,
                   "%s.kq_full_q: max_abs=%g head=%d pos=%d emel=%g reference=%g\n",
                   layer_prefix.c_str(),
                   full_q_kq_max_abs,
                   full_q_kq_max_head,
                   full_q_kq_max_pos,
                   full_q_kq_emel_at_max,
                   full_q_kq_reference_at_max);
      std::fprintf(stdout,
                   "%s.kq_ggml_f16: max_abs=%g head=%d pos=%d emel=%g reference=%g\n",
                   layer_prefix.c_str(),
                   ggml_f16_kq_max_abs,
                   ggml_f16_kq_max_head,
                   ggml_f16_kq_max_pos,
                   ggml_f16_kq_emel_at_max,
                   ggml_f16_kq_reference_at_max);
    } else {
      std::fprintf(stdout, "%s.kq: unavailable\n", layer_prefix.c_str());
    }
    std::fprintf(stdout,
                 "%s.kq_soft_max: max_abs=%g head=%d pos=%d emel=%g reference=%g\n",
                 layer_prefix.c_str(),
                 max_abs,
                 max_head,
                 max_pos,
                 emel_at_max,
                 reference_at_max);
    std::fprintf(stdout,
                 "%s.kq_soft_max_full_q: max_abs=%g head=%d pos=%d emel=%g reference=%g\n",
                 layer_prefix.c_str(),
                 full_q_max_abs,
                 full_q_max_head,
                 full_q_max_pos,
                 full_q_emel_at_max,
                 full_q_reference_at_max);
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
            reference_softmax_query_head_slice(*softmax_capture, head, 0);
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
      if (!reference_ctx.empty() && raw_reference_ctx.size() == reference_ctx.size()) {
        float raw_to_out_max_abs = 0.0f;
        int32_t raw_to_out_idx = 0;
        for (int32_t idx = 0; idx < static_cast<int32_t>(reference_ctx.size()); ++idx) {
          const float diff = std::fabs(raw_reference_ctx[static_cast<size_t>(idx)] -
                                       reference_ctx[static_cast<size_t>(idx)]);
          if (diff > raw_to_out_max_abs) {
            raw_to_out_max_abs = diff;
            raw_to_out_idx = idx;
          }
        }
        std::fprintf(stdout,
                     "%s.kqv_raw_to_kqv_out_reference: max_abs=%g idx=%d raw=%g out=%g\n",
                     layer_prefix.c_str(),
                     raw_to_out_max_abs,
                     raw_to_out_idx,
                     raw_reference_ctx[static_cast<size_t>(raw_to_out_idx)],
                     reference_ctx[static_cast<size_t>(raw_to_out_idx)]);
      }
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
  const int32_t batch_capacity = std::max(512, state.model_data->params.n_ctx);
  context_params.n_batch = batch_capacity;
  context_params.n_ubatch = batch_capacity;
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
  std::array<emel::text::formatter::chat_message, 1> conditioned_messages = {};
  emel::text::conditioner::event::prepare prepare_ev{conditioned_count, conditioned_error};
  prepare_ev.messages = emel::tools::generation_formatter_contract::single_user_messages(
      conditioned_messages, opts.text);
  prepare_ev.add_generation_prompt = true;
  prepare_ev.enable_thinking = false;
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

  int32_t runtime_exact_first_sig_step = -1;
  {
    emel::generator::detail::native_backend runtime_backend = {};
    emel::generator::detail::native_backend exact_nonflash_backend = {};
    if (emel::generator::detail::prepare(runtime_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none) &&
        emel::generator::detail::prepare(exact_nonflash_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none)) {
      const auto runtime_exact_max_abs_diff = [](std::span<const float> lhs,
                                                 std::span<const float> rhs) {
        if (lhs.size() != rhs.size()) {
          return std::numeric_limits<float>::infinity();
        }

        float max_abs = 0.0f;
        for (size_t idx = 0; idx < lhs.size(); ++idx) {
          max_abs = std::max(max_abs, std::fabs(lhs[idx] - rhs[idx]));
        }
        return max_abs;
      };
      const size_t prompt_count = prompt_tokens.size();
      int32_t first_hidden_step = -1;
      int32_t first_hidden_layer = -1;
      float first_hidden_max_abs = 0.0f;
      float first_hidden_attn_max_abs = 0.0f;
      int32_t first_sig_step = -1;
      int32_t first_sig_layer = -1;
      float first_sig_hidden_max_abs = 0.0f;
      float first_sig_attn_max_abs = 0.0f;

      for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
        const int32_t token_id = prefix_tokens[token_index];
        const int32_t position = static_cast<int32_t>(token_index);
        if (!emel::generator::detail::copy_tensor_row(
                *runtime_backend.token_embedding.tensor, token_id, runtime_backend.hidden) ||
            !emel::generator::detail::copy_tensor_row(
                *exact_nonflash_backend.token_embedding.tensor,
                token_id,
                exact_nonflash_backend.hidden)) {
          break;
        }

        for (int32_t layer = 0; layer < runtime_backend.n_layer; ++layer) {
          if (!run_layer_with_flash_request_q_attn(runtime_backend, layer, position) ||
              !run_layer_with_scalar_attention_ggml_nonflash_exact_masked(
                  exact_nonflash_backend, layer, position)) {
            token_index = prefix_tokens.size();
            break;
          }

          const float hidden_max_abs =
              runtime_exact_max_abs_diff(runtime_backend.hidden, exact_nonflash_backend.hidden);
          const float attn_max_abs =
              runtime_exact_max_abs_diff(runtime_backend.attn_ctx, exact_nonflash_backend.attn_ctx);
          const int32_t generated_index =
              static_cast<int32_t>(token_index) - static_cast<int32_t>(prompt_count);

          if (first_hidden_step < 0 && hidden_max_abs > 1.0e-6f) {
            first_hidden_step = generated_index;
            first_hidden_layer = layer;
            first_hidden_max_abs = hidden_max_abs;
            first_hidden_attn_max_abs = attn_max_abs;
          }
          if (first_sig_step < 0 && hidden_max_abs > 1.0e-4f) {
            first_sig_step = generated_index;
            first_sig_layer = layer;
            first_sig_hidden_max_abs = hidden_max_abs;
            first_sig_attn_max_abs = attn_max_abs;
          }
        }

        runtime_backend.kv_cache_tokens = position + 1;
        exact_nonflash_backend.kv_cache_tokens = position + 1;
      }

      std::fprintf(stdout,
                   "generation_debug.runtime_vs_exact_nonflash: first_hidden_gt_1e-6=%d "
                   "layer=%d hidden_max_abs=%g attn_max_abs=%g "
                   "first_hidden_gt_1e-4=%d layer=%d hidden_max_abs=%g attn_max_abs=%g\n",
                   first_hidden_step,
                   first_hidden_layer,
                   first_hidden_max_abs,
                   first_hidden_attn_max_abs,
                   first_sig_step,
                   first_sig_layer,
                   first_sig_hidden_max_abs,
                   first_sig_attn_max_abs);
      runtime_exact_first_sig_step = first_sig_step;
    }
  }
  if (token_mismatch_index > 2) {
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 2);
  }
  if (token_mismatch_index > 3) {
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 3);
  }
  if (token_mismatch_index > 4) {
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 4);
  }
  if (token_mismatch_index > 5) {
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 5);
  }
  if (token_mismatch_index > 6) {
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 6);
  }
  if (token_mismatch_index > 21) {
    dump_generation_selected_step_stage_debug(state, opts, reference_result, 21);
  }
  if (runtime_exact_first_sig_step >= 0) {
    dump_generation_selected_step_stage_debug(
        state, opts, reference_result, runtime_exact_first_sig_step);
  }

  constexpr float target_q2_threshold = 1.0e-4f;
  int32_t target_generated_index = token_mismatch_index - 1;
  const auto local_max_abs_diff = [](std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) {
      return std::numeric_limits<float>::infinity();
    }

    float max_abs = 0.0f;
    for (size_t idx = 0; idx < lhs.size(); ++idx) {
      max_abs = std::max(max_abs, std::fabs(lhs[idx] - rhs[idx]));
    }
    return max_abs;
  };
  const exact_matmul_mode reference_q2_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
      .use_reference_q8 = true,
  };
  if (target_generated_index > 0) {
    emel::generator::detail::native_backend scan_actor_backend = {};
    emel::generator::detail::native_backend scan_q2_reference_backend = {};
    if (emel::generator::detail::prepare(scan_actor_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none) &&
        emel::generator::detail::prepare(scan_q2_reference_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none)) {
      const size_t prompt_count = prompt_tokens.size();
      for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
        const int32_t scan_token_id = prefix_tokens[token_index];
        const int32_t scan_position = static_cast<int32_t>(token_index);
        if (!emel::generator::detail::copy_tensor_row(
                *scan_actor_backend.token_embedding.tensor, scan_token_id, scan_actor_backend.hidden) ||
            !emel::generator::detail::copy_tensor_row(*scan_q2_reference_backend.token_embedding.tensor,
                                                      scan_token_id,
                                                      scan_q2_reference_backend.hidden)) {
          break;
        }

        const bool is_prompt_token = token_index < prompt_count;
        const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
        bool found_target = false;
        for (int32_t layer = 0; layer < scan_actor_backend.n_layer; ++layer) {
          if (!run_layer_with_scalar_attention(scan_actor_backend, layer, scan_position) ||
              !run_layer_with_matmul_mode_scalar_attention(
                  scan_q2_reference_backend, layer, scan_position, reference_q2_only)) {
            found_target = false;
            break;
          }

          if (!is_prompt_token && layer == 1 &&
              local_max_abs_diff(scan_actor_backend.hidden, scan_q2_reference_backend.hidden) >
                  target_q2_threshold) {
            target_generated_index = generated_index;
            found_target = true;
            break;
          }
        }

        scan_actor_backend.kv_cache_tokens = scan_position + 1;
        scan_q2_reference_backend.kv_cache_tokens = scan_position + 1;
        if (found_target) {
          break;
        }
      }
    }
  }

  const int32_t target_prefix_token_count =
      static_cast<int32_t>(prompt_tokens.size()) + target_generated_index + 1;
  if (target_prefix_token_count <= 0 ||
      target_prefix_token_count > static_cast<int32_t>(prefix_tokens.size())) {
    std::fprintf(stdout, "generation_debug.state: invalid target prefix\n");
    return;
  }
  prefix_tokens.resize(static_cast<size_t>(target_prefix_token_count));

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

  const std::span<const float> reference_layer1_l_out =
      find_reference_tensor(graph_capture, "l_out-1");
  const auto dump_attention_mode = [&](const char * label, const auto & run_prefill) {
    if (reference_layer1_l_out.empty()) {
      std::fprintf(stdout, "generation_debug.mode_summary.%s: reference unavailable\n", label);
      return;
    }

    emel::generator::detail::native_backend mode_backend = {};
    if (emel::generator::detail::prepare(mode_backend, *state.model_data) !=
            emel::error::cast(emel::model::loader::error::none) ||
        !run_prefill(mode_backend, prefix_tokens)) {
      std::fprintf(stdout, "generation_debug.mode_summary.%s: replay failed\n", label);
      return;
    }

    std::fprintf(stdout,
                 "generation_debug.mode_summary.%s.layer1_l_out: max_abs=%g\n",
                 label,
                 local_max_abs_diff(mode_backend.hidden, reference_layer1_l_out));
  };

  dump_attention_mode("scalar", run_prefill_with_scalar_attention);
  dump_attention_mode("runtime_flash_q", run_prefill_with_flash_request_q);
  dump_attention_mode("runtime_flash_q_attn", run_prefill_with_flash_request_q_attn);
  dump_attention_mode("emel_prod_style", run_prefill_with_scalar_attention_emel_prod_style);
  dump_attention_mode("no_weight_rounding", run_prefill_with_scalar_attention_no_weight_rounding);
  dump_attention_mode("rounded_weight_scalar", run_prefill_with_scalar_attention_rounded_weight_scalar);
  dump_attention_mode("ggml_f16_value_contraction",
                      run_prefill_with_scalar_attention_ggml_f16_value_contraction);
  dump_attention_mode("online_f32", run_prefill_with_scalar_attention_online_f32);
  dump_attention_mode("double_softmax_sum",
                      run_prefill_with_scalar_attention_double_softmax_sum);
  dump_attention_mode("ggml_softmax",
                      run_prefill_with_scalar_attention_ggml_softmax);
  dump_attention_mode("ggml_online_f16", run_prefill_with_scalar_attention_ggml_online_f16);
  dump_attention_mode("ggml_nonflash_f16", run_prefill_with_scalar_attention_ggml_nonflash_f16);
  dump_attention_mode("ggml_f16_scores", run_prefill_with_scalar_attention_ggml_f16_scores);
  dump_attention_mode("full_q", run_prefill_with_scalar_attention_full_q);
  dump_attention_mode("full_q_no_weight_rounding",
                      run_prefill_with_scalar_attention_full_q_no_weight_rounding);
  dump_attention_mode("full_q_rounded_weight",
                      run_prefill_with_scalar_attention_full_q_rounded_weight);
  dump_attention_mode("full_q_ggml_f16_value_contraction",
                      run_prefill_with_scalar_attention_full_q_ggml_f16_value_contraction);

  const auto dump_full_generation_mode =
      [&](const char * label, const native_layer_runner run_layer_fn) {
        generation_result alt_result = {};
        const emel::error::type alt_err = run_custom_native_generate(
            state.backend, *state.model_data, opts, run_layer_fn, alt_result);
        if (alt_err == emel::error::cast(emel::generator::error::none)) {
          std::fprintf(stdout,
                       "generation_debug.mode_full.%s: match=%d token_mismatch=%d "
                       "byte_mismatch=%zu tokens=%d output_length=%zu\n",
                       label,
                       generation_results_match(alt_result, reference_result) ? 1 : 0,
                       first_token_mismatch_index(alt_result, reference_result),
                       first_mismatch_offset(alt_result, reference_result),
                       alt_result.tokens_generated,
                       alt_result.output_length);
          return;
        }

        std::fprintf(stdout,
                     "generation_debug.mode_full.%s: error=%d\n",
                     label,
                     static_cast<int>(alt_err));
      };

  dump_full_generation_mode("rounded_weight_scalar",
                            run_layer_with_scalar_attention_rounded_weight_scalar);
  dump_full_generation_mode("runtime_flash_q_attn", run_layer_with_flash_request_q_attn);
  dump_full_generation_mode("ggml_flash_ext_masked",
                            run_layer_with_scalar_attention_ggml_flash_ext_masked);
  dump_full_generation_mode("emel_prod_style", run_layer_with_scalar_attention_emel_prod_style);
  dump_full_generation_mode("emel_prod_style_float_value",
                            run_layer_with_scalar_attention_emel_prod_style_float_value);
  dump_full_generation_mode("emel_prod_style_float_value_reference_q2",
                            run_layer_with_scalar_attention_emel_prod_style_float_value_reference_q2);
  dump_full_generation_mode("ggml_f16_scores", run_layer_with_scalar_attention_ggml_f16_scores);
  dump_full_generation_mode("ggml_f16_scores_ggml_softmax",
                            run_layer_with_scalar_attention_ggml_f16_scores_ggml_softmax);
  dump_full_generation_mode("ggml_f16_value_contraction",
                            run_layer_with_scalar_attention_ggml_f16_value_contraction);
  dump_full_generation_mode("online_f32", run_layer_with_scalar_attention_online_f32);
  dump_full_generation_mode("ggml_online_f16", run_layer_with_scalar_attention_ggml_online_f16);
  dump_full_generation_mode("ggml_nonflash_f16", run_layer_with_scalar_attention_ggml_nonflash_f16);
  dump_full_generation_mode("ggml_nonflash_f16_ggml_softmax",
                            run_layer_with_scalar_attention_ggml_nonflash_f16_ggml_softmax);
  dump_full_generation_mode("ggml_nonflash_exact_masked",
                            run_layer_with_scalar_attention_ggml_nonflash_exact_masked);
  dump_full_generation_mode("ggml_nonflash_exact_masked_full_q",
                            run_layer_with_scalar_attention_ggml_nonflash_exact_masked_full_q);
  dump_full_generation_mode("ggml_nonflash_exact_scores_prod_value",
                            run_layer_with_scalar_attention_ggml_nonflash_exact_scores_prod_value);
  dump_full_generation_mode("exact_attention", run_layer_with_exact_attention_scalar_attention);
  dump_full_generation_mode("exact_q2", run_layer_with_exact_q2_scalar_attention);
  dump_full_generation_mode("exact_q3", run_layer_with_exact_q3_scalar_attention);
  dump_full_generation_mode("exact_q6", run_layer_with_exact_q6_scalar_attention);
  dump_full_generation_mode("scalar_q2", run_layer_with_scalar_q2_scalar_attention);
  dump_full_generation_mode("scalar_q3", run_layer_with_scalar_q3_scalar_attention);
  dump_full_generation_mode("scalar_q6", run_layer_with_scalar_q6_scalar_attention);
  dump_full_generation_mode("scalar_q236", run_layer_with_scalar_q236_scalar_attention);
  dump_full_generation_mode("reference_q2", run_layer_with_reference_q2_scalar_attention);
  dump_full_generation_mode("reference_q3", run_layer_with_reference_q3_scalar_attention);
  dump_full_generation_mode("reference_q6", run_layer_with_reference_q6_scalar_attention);
  dump_full_generation_mode("reference_q236", run_layer_with_reference_q236_scalar_attention);

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
  std::vector<std::vector<float>> reference_key_cache_rows(
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
    if (!capture_reference_key_cache_rows(
            reference_ctx.get(),
            layer,
            reference_key_cache_rows[static_cast<size_t>(layer)])) {
      std::fprintf(stdout,
                   "generation_debug.state.layer%d.reference_key_cache: unavailable\n",
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
    {
      const std::span<const float> reference_q = reference_last_token_row(
          find_reference_tensor(graph_capture, ("Qcur-" + layer_suffix).c_str()),
          static_cast<size_t>(backend.n_head * backend.head_dim));
      if (reference_q.size() == backend.q_attn.size()) {
        std::vector<float> reference_q_attn(reference_q.begin(), reference_q.end());
        emel::generator::detail::store_fp16_rounded_cache(reference_q, reference_q_attn.data());
        dump_state_compare((layer_prefix + ".q_attn").c_str(), backend.q_attn, reference_q_attn);
      } else {
        std::fprintf(stdout, "%s.q_attn: unavailable\n", layer_prefix.c_str());
      }
    }
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

    {
      auto flash_request = emel::generator::detail::make_flash_attn_request(backend, layer, position);
      std::vector<float> shared_flash_ctx(backend.attn_ctx.size(), 0.0f);
      flash_request.dst = emel::generator::detail::make_dst_view_3d(
          shared_flash_ctx.data(),
          flash_request.dst.ne[0],
          flash_request.dst.ne[1],
          flash_request.dst.ne[2]);
      emel::kernel::detail::flash_attn_workspace shared_flash_workspace = {};
      if (emel::kernel::detail::run_flash_attn_ext_with_workspace(
              flash_request, shared_flash_workspace)) {
        dump_state_compare((layer_prefix + ".kqv_out_shared_flash").c_str(),
                           shared_flash_ctx,
                           find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
      } else {
        std::fprintf(stdout, "%s.kqv_out_shared_flash: compute failed\n", layer_prefix.c_str());
      }

      const size_t layer_cache_offset = emel::generator::detail::layer_cache_offset(
          backend, layer, 0, kv_dim);
      const std::span<const float> reference_q =
          reference_last_token_row(find_reference_tensor(graph_capture, ("Qcur-" + layer_suffix).c_str()),
                                   static_cast<size_t>(backend.n_head * backend.head_dim));
      const auto & reference_key_cache =
          reference_key_cache_rows[static_cast<size_t>(layer)];
      const auto & reference_value_cache =
          reference_value_cache_rows[static_cast<size_t>(layer)];
      const std::span<const float> reference_key_prefix =
          reference_key_cache.size() >= static_cast<size_t>((position + 1) * kv_dim)
              ? std::span<const float>(reference_key_cache.data(),
                                       static_cast<size_t>((position + 1) * kv_dim))
              : std::span<const float>{};
      const std::span<const float> reference_value_prefix =
          reference_value_cache.size() >= static_cast<size_t>((position + 1) * kv_dim)
              ? std::span<const float>(reference_value_cache.data(),
                                       static_cast<size_t>((position + 1) * kv_dim))
              : std::span<const float>{};
      const reference_tensor_capture * softmax_capture_current =
          find_reference_capture(graph_capture, ("kq_soft_max-" + layer_suffix).c_str());
      const int32_t total_kv_tokens =
          softmax_capture_current != nullptr && softmax_capture_current->shape[0] > 0
              ? static_cast<int32_t>(softmax_capture_current->shape[0])
              : backend.n_ctx;
      std::vector<float> reference_key_padded(
          static_cast<size_t>(total_kv_tokens) * static_cast<size_t>(kv_dim), 0.0f);
      std::vector<float> reference_value_padded(
          static_cast<size_t>(total_kv_tokens) * static_cast<size_t>(kv_dim), 0.0f);
      if (!reference_key_prefix.empty()) {
        std::copy(reference_key_prefix.begin(),
                  reference_key_prefix.end(),
                  reference_key_padded.begin());
      }
      if (!reference_value_prefix.empty()) {
        std::copy(reference_value_prefix.begin(),
                  reference_value_prefix.end(),
                  reference_value_padded.begin());
      }

      if (!reference_key_prefix.empty()) {
        float key_cache_max_abs = 0.0f;
        int32_t key_cache_head = 0;
        int32_t key_cache_pos = 0;
        int32_t key_cache_dim = 0;
        float key_cache_emel = 0.0f;
        float key_cache_reference = 0.0f;
        for (int32_t cache_pos = 0; cache_pos <= position; ++cache_pos) {
          const size_t cache_row_offset =
              emel::generator::detail::layer_cache_offset(backend, layer, cache_pos, kv_dim);
          const std::span<const uint16_t> emel_cache_row{
              backend.key_cache.data() + cache_row_offset,
              static_cast<size_t>(kv_dim),
          };
          const std::span<const float> reference_cache_row = reference_key_prefix.subspan(
              static_cast<size_t>(cache_pos) * static_cast<size_t>(kv_dim),
              static_cast<size_t>(kv_dim));
          for (int32_t head = 0; head < backend.n_head_kv; ++head) {
            for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
              const int32_t idx = head * backend.head_dim_kv + dim;
              const float diff = std::fabs(read_debug_value(emel_cache_row, static_cast<size_t>(idx)) -
                                           reference_cache_row[static_cast<size_t>(idx)]);
              if (diff > key_cache_max_abs) {
                key_cache_max_abs = diff;
                key_cache_head = head;
                key_cache_pos = cache_pos;
                key_cache_dim = dim;
                key_cache_emel = read_debug_value(emel_cache_row, static_cast<size_t>(idx));
                key_cache_reference = reference_cache_row[static_cast<size_t>(idx)];
              }
            }
          }
        }
        std::fprintf(stdout,
                     "%s.key_cache: max_abs=%g head=%d pos=%d dim=%d emel=%g reference=%g\n",
                     layer_prefix.c_str(),
                     key_cache_max_abs,
                     key_cache_head,
                     key_cache_pos,
                     key_cache_dim,
                     key_cache_emel,
                     key_cache_reference);
      }

      const auto dump_ggml_flash_case =
          [&](const std::string & label,
              std::span<const float> q_rows,
              const auto k_rows,
              const auto v_rows) {
            std::vector<float> ggml_flash_ctx;
            if (run_ggml_flash_attn_ext_case(q_rows,
                                             k_rows,
                                             v_rows,
                                             static_cast<int64_t>(backend.head_dim),
                                             static_cast<int64_t>(position + 1),
                                             static_cast<int64_t>(backend.n_head),
                                             static_cast<int64_t>(backend.n_head_kv),
                                             ::emel::kernel::detail::flash_attn_scale(flash_request),
                                             ggml_flash_ctx)) {
              dump_state_compare(
                  label.c_str(),
                  ggml_flash_ctx,
                  find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
            } else {
              std::fprintf(stdout, "%s: compute failed\n", label.c_str());
            }
          };

      const auto dump_ggml_flash_masked_case =
          [&](const std::string & label,
              std::span<const float> q_rows,
              const auto k_rows,
              const auto v_rows) {
            std::vector<float> ggml_flash_ctx;
            if (run_ggml_flash_attn_ext_masked_case(q_rows,
                                                    k_rows,
                                                    v_rows,
                                                    static_cast<int64_t>(backend.head_dim),
                                                    static_cast<int64_t>(total_kv_tokens),
                                                    static_cast<int64_t>(position + 1),
                                                    static_cast<int64_t>(backend.n_head),
                                                    static_cast<int64_t>(backend.n_head_kv),
                                                    ::emel::kernel::detail::flash_attn_scale(
                                                        flash_request),
                                                    ggml_flash_ctx)) {
              dump_state_compare(
                  label.c_str(),
                  ggml_flash_ctx,
                  find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
            } else {
              std::fprintf(stdout, "%s: compute failed\n", label.c_str());
            }
          };

      const auto dump_ggml_nonflash_case =
          [&](const std::string & label,
              std::span<const float> q_rows,
              const auto k_rows,
              const auto v_rows) {
            std::vector<float> ggml_nonflash_ctx;
            if (run_ggml_nonflash_attn_case(q_rows,
                                            k_rows,
                                            v_rows,
                                            static_cast<int64_t>(backend.head_dim),
                                            static_cast<int64_t>(total_kv_tokens),
                                            static_cast<int64_t>(position + 1),
                                            static_cast<int64_t>(backend.n_head),
                                            static_cast<int64_t>(backend.n_head_kv),
                                            ::emel::kernel::detail::flash_attn_scale(flash_request),
                                            ggml_nonflash_ctx)) {
              dump_state_compare(
                  label.c_str(),
                  ggml_nonflash_ctx,
                  find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
            } else {
              std::fprintf(stdout, "%s: compute failed\n", label.c_str());
            }
          };

      const auto dump_emel_nonflash_case =
          [&](const std::string & label,
              std::span<const float> q_rows,
              const auto k_rows,
              const auto v_rows) {
            std::vector<float> emel_nonflash_ctx;
            if (run_emel_nonflash_f16_ggml_softmax_case(
                    q_rows,
                    k_rows,
                    v_rows,
                    static_cast<int64_t>(backend.head_dim),
                    static_cast<int64_t>(total_kv_tokens),
                    static_cast<int64_t>(position + 1),
                    static_cast<int64_t>(backend.n_head),
                    static_cast<int64_t>(backend.n_head_kv),
                    ::emel::kernel::detail::flash_attn_scale(flash_request),
                    emel_nonflash_ctx)) {
              dump_state_compare(
                  label.c_str(),
                  emel_nonflash_ctx,
                  find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
            } else {
              std::fprintf(stdout, "%s: compute failed\n", label.c_str());
            }
          };

      const auto dump_emel_prod_style_case =
          [&](const std::string & label,
              std::span<const float> q_rows,
              const auto k_rows,
              const auto v_rows) {
            std::vector<float> emel_prod_ctx;
            if (run_emel_prod_style_attn_case(q_rows,
                                              k_rows,
                                              v_rows,
                                              static_cast<int64_t>(backend.head_dim),
                                              static_cast<int64_t>(total_kv_tokens),
                                              static_cast<int64_t>(position + 1),
                                              static_cast<int64_t>(backend.n_head),
                                              static_cast<int64_t>(backend.n_head_kv),
                                              ::emel::kernel::detail::flash_attn_scale(flash_request),
                                              emel_prod_ctx)) {
              dump_state_compare(
                  label.c_str(),
                  emel_prod_ctx,
                  find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
            } else {
              std::fprintf(stdout, "%s: compute failed\n", label.c_str());
            }
          };

      dump_ggml_flash_masked_case(layer_prefix + ".kqv_out_ggml_flash_masked_reference_qkv",
                                  reference_q,
                                  reference_key_padded,
                                  reference_value_padded);
      dump_ggml_flash_masked_case(layer_prefix + ".kqv_out_ggml_flash_masked_reference_q_emel_kv",
                                  reference_q,
                                  std::span<const uint16_t>(
                                      backend.key_cache.data() + layer_cache_offset,
                                      static_cast<size_t>(total_kv_tokens * kv_dim)),
                                  std::span<const uint16_t>(
                                      backend.value_cache.data() + layer_cache_offset,
                                      static_cast<size_t>(total_kv_tokens * kv_dim)));
      dump_ggml_flash_masked_case(layer_prefix + ".kqv_out_ggml_flash_masked_emel_q_reference_kv",
                                  std::span<const float>(
                                      backend.q.data(),
                                      static_cast<size_t>(backend.n_head * backend.head_dim)),
                                  reference_key_padded,
                                  reference_value_padded);
      dump_ggml_flash_case(layer_prefix + ".kqv_out_ggml_flash_reference_qkv",
                           reference_q,
                           reference_key_prefix,
                           reference_value_prefix);
      dump_ggml_flash_case(layer_prefix + ".kqv_out_ggml_flash_reference_q_emel_kv",
                           reference_q,
                           std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                  static_cast<size_t>((position + 1) * kv_dim)),
                           std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                  static_cast<size_t>((position + 1) * kv_dim)));
      dump_ggml_flash_case(layer_prefix + ".kqv_out_ggml_flash_emel_q_reference_kv",
                           std::span<const float>(
                               backend.q.data(),
                               static_cast<size_t>(backend.n_head * backend.head_dim)),
                           reference_key_prefix,
                           reference_value_prefix);
      dump_ggml_nonflash_case(layer_prefix + ".kqv_out_ggml_nonflash_reference_qkv",
                              reference_q,
                              reference_key_padded,
                              reference_value_padded);
      dump_emel_nonflash_case(layer_prefix + ".kqv_out_emel_nonflash_reference_qkv",
                              reference_q,
                              reference_key_padded,
                              reference_value_padded);
      dump_emel_prod_style_case(layer_prefix + ".kqv_out_emel_prod_reference_qkv",
                                reference_q,
                                reference_key_padded,
                                reference_value_padded);
      dump_ggml_nonflash_case(layer_prefix + ".kqv_out_ggml_nonflash_reference_q_emel_kv",
                              reference_q,
                              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
                              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)));
      dump_emel_nonflash_case(layer_prefix + ".kqv_out_emel_nonflash_reference_q_emel_kv",
                              reference_q,
                              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
                              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)));
      dump_ggml_nonflash_case(layer_prefix + ".kqv_out_ggml_nonflash_emel_q_reference_kv",
                              std::span<const float>(
                                  backend.q.data(),
                                  static_cast<size_t>(backend.n_head * backend.head_dim)),
                              reference_key_padded,
                              reference_value_padded);
      dump_emel_nonflash_case(layer_prefix + ".kqv_out_emel_nonflash_emel_q_reference_kv",
                              std::span<const float>(
                                  backend.q.data(),
                                  static_cast<size_t>(backend.n_head * backend.head_dim)),
                              reference_key_padded,
                              reference_value_padded);
      dump_emel_nonflash_case(layer_prefix + ".kqv_out_emel_nonflash_emel_q",
                              std::span<const float>(
                                  backend.q.data(),
                                  static_cast<size_t>(backend.n_head * backend.head_dim)),
                              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
                              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)));
      dump_emel_prod_style_case(layer_prefix + ".kqv_out_emel_prod_emel_q",
                                std::span<const float>(
                                    backend.q.data(),
                                    static_cast<size_t>(backend.n_head * backend.head_dim)),
                                std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                       static_cast<size_t>(total_kv_tokens * kv_dim)),
                                std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                       static_cast<size_t>(total_kv_tokens * kv_dim)));
      dump_ggml_nonflash_case(layer_prefix + ".kqv_out_ggml_nonflash_emel_q_attn",
                              std::span<const float>(
                                  backend.q_attn.data(),
                                  static_cast<size_t>(backend.n_head * backend.head_dim)),
                              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
                              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                     static_cast<size_t>(total_kv_tokens * kv_dim)));
      dump_emel_prod_style_case(layer_prefix + ".kqv_out_emel_prod_emel_q_attn",
                                std::span<const float>(
                                    backend.q_attn.data(),
                                    static_cast<size_t>(backend.n_head * backend.head_dim)),
                                std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                                       static_cast<size_t>(total_kv_tokens * kv_dim)),
                                std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                                       static_cast<size_t>(total_kv_tokens * kv_dim)));

      std::vector<float> ggml_flash_ctx;
      if (run_ggml_flash_attn_ext_case(
              std::span<const float>(backend.q.data(),
                                     static_cast<size_t>(backend.n_head * backend.head_dim)),
              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                     static_cast<size_t>((position + 1) * kv_dim)),
              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                     static_cast<size_t>((position + 1) * kv_dim)),
              static_cast<int64_t>(backend.head_dim),
              static_cast<int64_t>(position + 1),
              static_cast<int64_t>(backend.n_head),
              static_cast<int64_t>(backend.n_head_kv),
              ::emel::kernel::detail::flash_attn_scale(flash_request),
              ggml_flash_ctx)) {
        dump_state_compare((layer_prefix + ".kqv_out_ggml_flash_emel_q").c_str(),
                           ggml_flash_ctx,
                           find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
      } else {
        std::fprintf(stdout,
                     "%s.kqv_out_ggml_flash_emel_q: compute failed\n",
                     layer_prefix.c_str());
      }

      std::vector<float> ggml_flash_masked_ctx;
      if (run_ggml_flash_attn_ext_masked_case(
              std::span<const float>(backend.q.data(),
                                     static_cast<size_t>(backend.n_head * backend.head_dim)),
              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
              static_cast<int64_t>(backend.head_dim),
              static_cast<int64_t>(total_kv_tokens),
              static_cast<int64_t>(position + 1),
              static_cast<int64_t>(backend.n_head),
              static_cast<int64_t>(backend.n_head_kv),
              ::emel::kernel::detail::flash_attn_scale(flash_request),
              ggml_flash_masked_ctx)) {
        dump_state_compare((layer_prefix + ".kqv_out_ggml_flash_masked_emel_q").c_str(),
                           ggml_flash_masked_ctx,
                           find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
      } else {
        std::fprintf(stdout,
                     "%s.kqv_out_ggml_flash_masked_emel_q: compute failed\n",
                     layer_prefix.c_str());
      }

      std::vector<float> ggml_nonflash_ctx;
      if (run_ggml_nonflash_attn_case(
              std::span<const float>(backend.q.data(),
                                     static_cast<size_t>(backend.n_head * backend.head_dim)),
              std::span<const uint16_t>(backend.key_cache.data() + layer_cache_offset,
                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
              std::span<const uint16_t>(backend.value_cache.data() + layer_cache_offset,
                                     static_cast<size_t>(total_kv_tokens * kv_dim)),
              static_cast<int64_t>(backend.head_dim),
              static_cast<int64_t>(total_kv_tokens),
              static_cast<int64_t>(position + 1),
              static_cast<int64_t>(backend.n_head),
              static_cast<int64_t>(backend.n_head_kv),
              ::emel::kernel::detail::flash_attn_scale(flash_request),
              ggml_nonflash_ctx)) {
        dump_state_compare((layer_prefix + ".kqv_out_ggml_nonflash_emel_q").c_str(),
                           ggml_nonflash_ctx,
                           find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
      } else {
        std::fprintf(stdout,
                     "%s.kqv_out_ggml_nonflash_emel_q: compute failed\n",
                     layer_prefix.c_str());
      }

#if defined(__aarch64__) || defined(__ARM_NEON)
      std::vector<float> neon_flash_ctx(backend.attn_ctx.size(), 0.0f);
      auto neon_flash_request = flash_request;
      neon_flash_request.dst = emel::generator::detail::make_dst_view_3d(
          neon_flash_ctx.data(),
          neon_flash_request.dst.ne[0],
          neon_flash_request.dst.ne[1],
          neon_flash_request.dst.ne[2]);
      emel::kernel::detail::flash_attn_workspace neon_flash_workspace = {};
      if (emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
              neon_flash_request, true, neon_flash_workspace)) {
        dump_state_compare((layer_prefix + ".kqv_out_neon_flash").c_str(),
                           neon_flash_ctx,
                           find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()));
      } else {
        std::fprintf(stdout, "%s.kqv_out_neon_flash: compute failed\n", layer_prefix.c_str());
      }
#endif
    }

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
    if (layer == 0) {
      const std::span<const float> reference_ffn_inp =
          find_reference_tensor(graph_capture, ("ffn_inp-" + layer_suffix).c_str());
      const std::span<const float> reference_ffn_norm =
          find_reference_tensor(graph_capture, ("ffn_norm-" + layer_suffix).c_str());
      if (!reference_ffn_inp.empty() && !reference_ffn_norm.empty()) {
        std::vector<float> emel_reference_ffn_norm(reference_ffn_norm.size(), 0.0f);
        if (emel::generator::detail::rms_norm(reference_ffn_inp,
                                              block.feed_forward_norm,
                                              backend.rms_epsilon,
                                              emel_reference_ffn_norm)) {
          dump_state_compare((layer_prefix + ".ffn_norm_from_reference_inp").c_str(),
                             emel_reference_ffn_norm,
                             reference_ffn_norm);

          const std::span<const float> reference_ffn_gate =
              find_reference_tensor(graph_capture, ("ffn_gate-" + layer_suffix).c_str());
          const std::span<const float> reference_ffn_up =
              find_reference_tensor(graph_capture, ("ffn_up-" + layer_suffix).c_str());
          if (!reference_ffn_gate.empty() && !reference_ffn_up.empty()) {
            std::vector<float> emel_reference_ffn_gate(reference_ffn_gate.size(), 0.0f);
            std::vector<float> emel_reference_ffn_up(reference_ffn_up.size(), 0.0f);
            if (emel::generator::detail::matmul_vector(
                    backend,
                    block.feed_forward_gate,
                    emel_reference_ffn_norm,
                    emel_reference_ffn_gate) &&
                emel::generator::detail::matmul_vector(
                    backend,
                    block.feed_forward_up,
                    emel_reference_ffn_norm,
                    emel_reference_ffn_up)) {
              dump_state_compare((layer_prefix + ".ffn_gate_from_reference_norm").c_str(),
                                 emel_reference_ffn_gate,
                                 reference_ffn_gate);
              dump_state_compare((layer_prefix + ".ffn_up_from_reference_norm").c_str(),
                                 emel_reference_ffn_up,
                                 reference_ffn_up);
            }
          }
        }
      }
    }
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

float max_abs_diff(std::span<const uint16_t> emel_values, std::span<const float> reference_values) {
  if (emel_values.size() != reference_values.size()) {
    return -1.0f;
  }

  float max_abs = 0.0f;
  for (size_t idx = 0; idx < emel_values.size(); ++idx) {
    max_abs = std::max(
        max_abs,
        std::fabs(fp16_storage_to_fp32(emel_values[idx]) - reference_values[idx]));
  }
  return max_abs;
}

float max_abs_diff(std::span<const float> emel_values, std::span<const uint16_t> reference_values) {
  return max_abs_diff(reference_values, emel_values);
}

float max_abs_diff(std::span<const uint16_t> emel_values,
                   std::span<const uint16_t> reference_values) {
  if (emel_values.size() != reference_values.size()) {
    return -1.0f;
  }

  float max_abs = 0.0f;
  for (size_t idx = 0; idx < emel_values.size(); ++idx) {
    max_abs = std::max(max_abs,
                       std::fabs(fp16_storage_to_fp32(emel_values[idx]) -
                                 fp16_storage_to_fp32(reference_values[idx])));
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
  const int64_t ne3 = std::max<int64_t>(capture.shape[3], 1);
  if (token_index < 0 || token_index >= ne3) {
    return {};
  }

  const size_t row_width = static_cast<size_t>(ne0 * ne1 * ne2);
  const size_t offset = static_cast<size_t>(token_index) * row_width;
  if (offset + row_width > capture.values.size()) {
    return {};
  }
  return std::span<const float>(capture.values).subspan(offset, row_width);
}

std::vector<float> reference_token_tensor_values(const reference_tensor_capture & capture,
                                                 const int32_t token_index) {
  const int64_t ne0 = std::max<int64_t>(capture.shape[0], 1);
  const int64_t ne1 = std::max<int64_t>(capture.shape[1], 1);
  const int64_t ne2 = std::max<int64_t>(capture.shape[2], 1);
  const int64_t ne3 = std::max<int64_t>(capture.shape[3], 1);
  if (token_index < 0) {
    return {};
  }

  if (ne3 > 1 && token_index < ne3) {
    const size_t row_width = static_cast<size_t>(ne0 * ne1 * ne2);
    const size_t offset = static_cast<size_t>(token_index) * row_width;
    if (offset + row_width > capture.values.size()) {
      return {};
    }
    return std::vector<float>(capture.values.begin() + static_cast<std::ptrdiff_t>(offset),
                              capture.values.begin() +
                                  static_cast<std::ptrdiff_t>(offset + row_width));
  }

  if (ne1 <= 1 || token_index >= ne1) {
    return {};
  }

  const size_t slice_width = static_cast<size_t>(ne0 * ne2 * ne3);
  std::vector<float> out(slice_width, 0.0f);
  size_t out_offset = 0;
  for (int64_t i3 = 0; i3 < ne3; ++i3) {
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
      const size_t capture_offset =
          ((((static_cast<size_t>(i3) * static_cast<size_t>(ne2)) + static_cast<size_t>(i2)) *
                static_cast<size_t>(ne1)) +
           static_cast<size_t>(token_index)) *
          static_cast<size_t>(ne0);
      if (capture_offset + static_cast<size_t>(ne0) > capture.values.size() ||
          out_offset + static_cast<size_t>(ne0) > out.size()) {
        return {};
      }
      std::copy_n(capture.values.begin() + static_cast<std::ptrdiff_t>(capture_offset),
                  static_cast<size_t>(ne0),
                  out.begin() + static_cast<std::ptrdiff_t>(out_offset));
      out_offset += static_cast<size_t>(ne0);
    }
  }

  return out;
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
  const int32_t timeline_start_generated = 0;
  const int32_t prefix_generated_tokens = token_mismatch_index;
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
  emel::generator::detail::native_backend exact_q3_backend = {};
  emel::generator::detail::native_backend reference_q3_backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(exact_q3_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(reference_q3_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.timeline: backend prepare failed\n");
    return;
  }

  const exact_matmul_mode exact_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
  };
  const exact_matmul_mode reference_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_reference_q8 = true,
  };

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
  constexpr std::array<float, 3> cache_thresholds{1.0e-5f, 1.0e-4f, 1.0e-3f};
  float layer1_key_cache_max_abs = 0.0f;
  float layer1_value_cache_max_abs = 0.0f;
  int32_t layer1_key_cache_max_generated = -1;
  int32_t layer1_value_cache_max_generated = -1;
  std::array<int32_t, cache_thresholds.size()> layer1_key_cache_first_generated = {};
  std::array<int32_t, cache_thresholds.size()> layer1_value_cache_first_generated = {};
  layer1_key_cache_first_generated.fill(-1);
  layer1_value_cache_first_generated.fill(-1);

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
      if (!is_prompt_token && generated_index < timeline_start_generated) {
        continue;
      }
      const std::string token_prefix = is_prompt_token
                                           ? "generation_debug.timeline.prompt" +
                                                 std::to_string(token_index)
                                           : "generation_debug.timeline.gen" +
                                                 std::to_string(generated_index);
      const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer, 0, kv_dim);
      const std::span<const uint16_t> cache_row0{
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
            const std::span<const uint16_t> emel_cache_row{
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
            const std::span<const uint16_t> emel_cache_row{
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
          if (token_index == 0u) {
            const std::span<const float> reference_cache_row =
                std::span<const float>(reference_layer0_value_cache)
                    .subspan(static_cast<size_t>(position) * static_cast<size_t>(kv_dim),
                             static_cast<size_t>(kv_dim));
            std::vector<float> ggml_rounded_cache_row(static_cast<size_t>(kv_dim));
            for (int32_t idx = 0; idx < kv_dim; ++idx) {
              ggml_rounded_cache_row[static_cast<size_t>(idx)] =
                  round_scalar_to_fp16(backend.v[static_cast<size_t>(idx)]);
            }
            dump_state_compare((token_prefix + ".layer0_value_cache_ggml_round").c_str(),
                               ggml_rounded_cache_row,
                               reference_cache_row);
            dump_state_compare((token_prefix + ".layer0_value_cache_vs_ggml_round").c_str(),
                               std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                                      static_cast<size_t>(kv_dim)),
                               ggml_rounded_cache_row);
            if (!emel::generator::detail::copy_tensor_row(
                    *exact_q3_backend.token_embedding.tensor,
                    token_id,
                    exact_q3_backend.hidden) ||
                !run_layer_with_matmul_mode_scalar_attention(
                    exact_q3_backend, layer, position, exact_q3_only)) {
              std::fprintf(stdout,
                           "%s.layer0_value_cache_exact_q3: replay failed\n",
                           token_prefix.c_str());
            } else {
              const size_t exact_cache_offset = emel::generator::detail::layer_cache_offset(
                  exact_q3_backend, layer, position, kv_dim);
              const std::span<const uint16_t> exact_cache_row{
                  exact_q3_backend.value_cache.data() + exact_cache_offset,
                  static_cast<size_t>(kv_dim),
              };
              std::fprintf(stdout,
                           "%s.layer0_value_cache_exact_q3: max_abs=%g\n",
                           token_prefix.c_str(),
                           max_abs_diff(exact_cache_row, reference_cache_row));
            }
            if (!emel::generator::detail::copy_tensor_row(
                    *reference_q3_backend.token_embedding.tensor,
                    token_id,
                    reference_q3_backend.hidden) ||
                !run_layer_with_matmul_mode_scalar_attention(
                    reference_q3_backend, layer, position, reference_q3_only)) {
              std::fprintf(stdout,
                           "%s.layer0_value_cache_reference_q3: replay failed\n",
                           token_prefix.c_str());
            } else {
              const size_t reference_q3_cache_offset = emel::generator::detail::layer_cache_offset(
                  reference_q3_backend, layer, position, kv_dim);
              const std::span<const uint16_t> reference_q3_cache_row{
                  reference_q3_backend.value_cache.data() + reference_q3_cache_offset,
                  static_cast<size_t>(kv_dim),
              };
              std::fprintf(stdout,
                           "%s.layer0_value_cache_reference_q3: max_abs=%g\n",
                           token_prefix.c_str(),
                           max_abs_diff(reference_q3_cache_row, reference_cache_row));
            }
          }
        }
        if (token_index == 0u) {
          layer0_value_cache0_baseline = decode_fp16_storage(cache_row0);
        } else if (!layer0_value_cache0_baseline.empty()) {
          std::fprintf(stdout,
                       "%s.layer0_value_cache0_drift: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(cache_row0, layer0_value_cache0_baseline));
        }
        if (layer0_v_capture != nullptr) {
          const std::vector<float> reference_v =
              reference_token_tensor_values(*layer0_v_capture, position);
          std::fprintf(stdout,
                       "%s.layer0_v: max_abs=%g\n",
                       token_prefix.c_str(),
                       reference_v.empty() ? -1.0f : max_abs_diff(backend.v, reference_v));
        }
        if (layer0_q_capture != nullptr) {
          const std::vector<float> reference_q =
              reference_token_tensor_values(*layer0_q_capture, position);
          std::fprintf(stdout,
                       "%s.layer0_q: max_abs=%g\n",
                       token_prefix.c_str(),
                       reference_q.empty() ? -1.0f : max_abs_diff(backend.q, reference_q));
        }
        if (layer0_k_capture != nullptr) {
          const std::vector<float> reference_k =
              reference_token_tensor_values(*layer0_k_capture, position);
          std::fprintf(stdout,
                       "%s.layer0_k: max_abs=%g\n",
                       token_prefix.c_str(),
                       reference_k.empty() ? -1.0f : max_abs_diff(backend.k, reference_k));
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
        const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
        std::vector<float> reference_layer1_key_cache;
        std::vector<float> reference_layer1_value_cache;
        if (capture_reference_key_cache_rows(reference_ctx.get(), 1, reference_layer1_key_cache) &&
            reference_layer1_key_cache.size() == static_cast<size_t>((position + 1) * kv_dim)) {
          float max_cache_abs = 0.0f;
          for (int32_t cache_pos = 0; cache_pos <= position; ++cache_pos) {
            const size_t cache_offset_compare = emel::generator::detail::layer_cache_offset(
                backend, layer, cache_pos, kv_dim);
            const std::span<const uint16_t> emel_cache_row{
                backend.key_cache.data() + cache_offset_compare,
                static_cast<size_t>(kv_dim),
            };
            const std::span<const float> reference_cache_row =
                std::span<const float>(reference_layer1_key_cache)
                    .subspan(static_cast<size_t>(cache_pos) * static_cast<size_t>(kv_dim),
                             static_cast<size_t>(kv_dim));
            max_cache_abs = std::max(max_cache_abs, max_abs_diff(emel_cache_row, reference_cache_row));
          }
          std::fprintf(stdout,
                       "%s.layer1_key_cache: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_cache_abs);
          if (!is_prompt_token) {
            if (max_cache_abs > layer1_key_cache_max_abs) {
              layer1_key_cache_max_abs = max_cache_abs;
              layer1_key_cache_max_generated = generated_index;
            }
            for (size_t threshold_index = 0; threshold_index < cache_thresholds.size();
                 ++threshold_index) {
              if (layer1_key_cache_first_generated[threshold_index] < 0 &&
                  max_cache_abs > cache_thresholds[threshold_index]) {
                layer1_key_cache_first_generated[threshold_index] = generated_index;
              }
            }
          }
        }
        if (capture_reference_value_cache_rows(reference_ctx.get(), 1, reference_layer1_value_cache) &&
            reference_layer1_value_cache.size() == static_cast<size_t>((position + 1) * kv_dim)) {
          float max_cache_abs = 0.0f;
          for (int32_t cache_pos = 0; cache_pos <= position; ++cache_pos) {
            const size_t cache_offset_compare = emel::generator::detail::layer_cache_offset(
                backend, layer, cache_pos, kv_dim);
            const std::span<const uint16_t> emel_cache_row{
                backend.value_cache.data() + cache_offset_compare,
                static_cast<size_t>(kv_dim),
            };
            const std::span<const float> reference_cache_row =
                std::span<const float>(reference_layer1_value_cache)
                    .subspan(static_cast<size_t>(cache_pos) * static_cast<size_t>(kv_dim),
                             static_cast<size_t>(kv_dim));
            max_cache_abs = std::max(max_cache_abs, max_abs_diff(emel_cache_row, reference_cache_row));
          }
          std::fprintf(stdout,
                       "%s.layer1_value_cache: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_cache_abs);
          if (!is_prompt_token) {
            if (max_cache_abs > layer1_value_cache_max_abs) {
              layer1_value_cache_max_abs = max_cache_abs;
              layer1_value_cache_max_generated = generated_index;
            }
            for (size_t threshold_index = 0; threshold_index < cache_thresholds.size();
                 ++threshold_index) {
              if (layer1_value_cache_first_generated[threshold_index] < 0 &&
                  max_cache_abs > cache_thresholds[threshold_index]) {
                layer1_value_cache_first_generated[threshold_index] = generated_index;
              }
            }
          }
        }
        if (token_index == 0u) {
          layer1_value_cache0_baseline = decode_fp16_storage(cache_row0);
        } else if (!layer1_value_cache0_baseline.empty()) {
          std::fprintf(stdout,
                       "%s.layer1_value_cache0_drift: max_abs=%g\n",
                       token_prefix.c_str(),
                       max_abs_diff(cache_row0, layer1_value_cache0_baseline));
        }
        if (layer1_v_capture != nullptr) {
          const std::vector<float> reference_v =
              reference_token_tensor_values(*layer1_v_capture, position);
          std::fprintf(stdout,
                       "%s.layer1_v: max_abs=%g\n",
                       token_prefix.c_str(),
                       reference_v.empty() ? -1.0f : max_abs_diff(backend.v, reference_v));
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

  std::fprintf(stdout,
               "generation_debug.timeline.summary.layer1_key_cache: max_abs=%g gen=%d\n",
               layer1_key_cache_max_abs,
               layer1_key_cache_max_generated);
  std::fprintf(stdout,
               "generation_debug.timeline.summary.layer1_value_cache: max_abs=%g gen=%d\n",
               layer1_value_cache_max_abs,
               layer1_value_cache_max_generated);
  for (size_t threshold_index = 0; threshold_index < cache_thresholds.size(); ++threshold_index) {
    std::fprintf(stdout,
                 "generation_debug.timeline.summary.layer1_threshold=%g key_first_gen=%d "
                 "value_first_gen=%d\n",
                 cache_thresholds[threshold_index],
                 layer1_key_cache_first_generated[threshold_index],
                 layer1_value_cache_first_generated[threshold_index]);
  }
}

void dump_generation_q23_timeline_debug(const generation_load_state & state,
                                        const emel::paritychecker::parity_options & opts,
                                        const generation_result & emel_result,
                                        const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  const int32_t timeline_window = 12;
  const int32_t timeline_start_generated = std::max<int32_t>(0, token_mismatch_index - timeline_window);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    std::fprintf(stdout, "generation_debug.q23_timeline: tokenize failed\n");
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

  emel::generator::detail::native_backend actor_backend = {};
  emel::generator::detail::native_backend q23_backend = {};
  if (emel::generator::detail::prepare(actor_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q23_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.q23_timeline: backend prepare failed\n");
    return;
  }

  const exact_matmul_mode scalar_quant_q23{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)),
      .use_scalar_quantized = true,
  };
  const size_t prompt_count = prompt_tokens.size();

  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (!emel::generator::detail::copy_tensor_row(
            *actor_backend.token_embedding.tensor, token_id, actor_backend.hidden) ||
        !emel::generator::detail::copy_tensor_row(
            *q23_backend.token_embedding.tensor, token_id, q23_backend.hidden)) {
      std::fprintf(stdout, "generation_debug.q23_timeline: token embedding replay failed\n");
      return;
    }

    const bool is_prompt_token = token_index < prompt_count;
    const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
    const bool emit_token_debug = is_prompt_token || generated_index >= timeline_start_generated;
    const std::string token_prefix = is_prompt_token
                                         ? "generation_debug.q23_timeline.prompt" +
                                               std::to_string(token_index)
                                         : "generation_debug.q23_timeline.gen" +
                                               std::to_string(generated_index);

    for (int32_t layer = 0; layer < actor_backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention(actor_backend, layer, position) ||
          !run_layer_with_matmul_mode_scalar_attention(
              q23_backend, layer, position, scalar_quant_q23)) {
        std::fprintf(stdout, "generation_debug.q23_timeline: layer replay failed\n");
        return;
      }

      if (!emit_token_debug) {
        continue;
      }
      std::fprintf(stdout,
                   "%s.layer%d_l_out: max_abs=%g\n",
                   token_prefix.c_str(),
                   layer,
                   max_abs_diff(actor_backend.hidden, q23_backend.hidden));
    }

    actor_backend.kv_cache_tokens = position + 1;
    q23_backend.kv_cache_tokens = position + 1;
  }
}

void dump_generation_reference_q_timeline_debug(const generation_load_state & state,
                                                const emel::paritychecker::parity_options & opts,
                                                const generation_result & emel_result,
                                                const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  const int32_t timeline_window = 12;
  const int32_t timeline_start_generated = std::max<int32_t>(0, token_mismatch_index - timeline_window);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    std::fprintf(stdout, "generation_debug.qref_timeline: tokenize failed\n");
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

  emel::generator::detail::native_backend actor_backend = {};
  emel::generator::detail::native_backend q2_reference_backend = {};
  emel::generator::detail::native_backend q3_reference_backend = {};
  if (emel::generator::detail::prepare(actor_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.qref_timeline: backend prepare failed\n");
    return;
  }

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
  const size_t prompt_count = prompt_tokens.size();
  constexpr std::array<float, 3> qref_thresholds{1.0e-5f, 1.0e-4f, 1.0e-3f};
  std::array<float, 2> q2_max_abs{0.0f, 0.0f};
  std::array<float, 2> q3_max_abs{0.0f, 0.0f};
  std::array<int32_t, 2> q2_max_generated{-1, -1};
  std::array<int32_t, 2> q3_max_generated{-1, -1};
  std::array<std::array<int32_t, qref_thresholds.size()>, 2> q2_first_generated{};
  std::array<std::array<int32_t, qref_thresholds.size()>, 2> q3_first_generated{};
  for (auto & per_layer : q2_first_generated) {
    per_layer.fill(-1);
  }
  for (auto & per_layer : q3_first_generated) {
    per_layer.fill(-1);
  }

  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (!emel::generator::detail::copy_tensor_row(
            *actor_backend.token_embedding.tensor, token_id, actor_backend.hidden) ||
        !emel::generator::detail::copy_tensor_row(
            *q2_reference_backend.token_embedding.tensor, token_id, q2_reference_backend.hidden) ||
        !emel::generator::detail::copy_tensor_row(
            *q3_reference_backend.token_embedding.tensor, token_id, q3_reference_backend.hidden)) {
      std::fprintf(stdout, "generation_debug.qref_timeline: token embedding replay failed\n");
      return;
    }

    const bool is_prompt_token = token_index < prompt_count;
    const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
    const bool emit_token_debug = is_prompt_token || generated_index >= timeline_start_generated;
    const std::string token_prefix = is_prompt_token
                                         ? "generation_debug.qref_timeline.prompt" +
                                               std::to_string(token_index)
                                         : "generation_debug.qref_timeline.gen" +
                                               std::to_string(generated_index);

    for (int32_t layer = 0; layer < actor_backend.n_layer; ++layer) {
      if (!run_layer_with_scalar_attention(actor_backend, layer, position) ||
          !run_layer_with_matmul_mode_scalar_attention(
              q2_reference_backend, layer, position, reference_q2_only) ||
          !run_layer_with_matmul_mode_scalar_attention(
              q3_reference_backend, layer, position, reference_q3_only)) {
        std::fprintf(stdout, "generation_debug.qref_timeline: layer replay failed\n");
        return;
      }

      const float q2_diff = max_abs_diff(actor_backend.hidden, q2_reference_backend.hidden);
      const float q3_diff = max_abs_diff(actor_backend.hidden, q3_reference_backend.hidden);
      if (!is_prompt_token && layer < static_cast<int32_t>(q2_max_abs.size())) {
        if (q2_diff > q2_max_abs[static_cast<size_t>(layer)]) {
          q2_max_abs[static_cast<size_t>(layer)] = q2_diff;
          q2_max_generated[static_cast<size_t>(layer)] = generated_index;
        }
        if (q3_diff > q3_max_abs[static_cast<size_t>(layer)]) {
          q3_max_abs[static_cast<size_t>(layer)] = q3_diff;
          q3_max_generated[static_cast<size_t>(layer)] = generated_index;
        }
        for (size_t threshold_index = 0; threshold_index < qref_thresholds.size(); ++threshold_index) {
          if (q2_first_generated[static_cast<size_t>(layer)][threshold_index] < 0 &&
              q2_diff > qref_thresholds[threshold_index]) {
            q2_first_generated[static_cast<size_t>(layer)][threshold_index] = generated_index;
          }
          if (q3_first_generated[static_cast<size_t>(layer)][threshold_index] < 0 &&
              q3_diff > qref_thresholds[threshold_index]) {
            q3_first_generated[static_cast<size_t>(layer)][threshold_index] = generated_index;
          }
        }
      }

      if (!emit_token_debug) {
        continue;
      }
      std::fprintf(stdout,
                   "%s.layer%d_q2_reference_l_out: max_abs=%g\n",
                   token_prefix.c_str(),
                   layer,
                   q2_diff);
      std::fprintf(stdout,
                   "%s.layer%d_q3_reference_l_out: max_abs=%g\n",
                   token_prefix.c_str(),
                   layer,
                   q3_diff);
    }

    actor_backend.kv_cache_tokens = position + 1;
    q2_reference_backend.kv_cache_tokens = position + 1;
    q3_reference_backend.kv_cache_tokens = position + 1;
  }

  for (size_t layer = 0; layer < q2_max_abs.size(); ++layer) {
    std::fprintf(stdout,
                 "generation_debug.qref_summary.layer%zu.q2_max=%g gen=%d q3_max=%g gen=%d\n",
                 layer,
                 q2_max_abs[layer],
                 q2_max_generated[layer],
                 q3_max_abs[layer],
                 q3_max_generated[layer]);
    for (size_t threshold_index = 0; threshold_index < qref_thresholds.size(); ++threshold_index) {
      std::fprintf(stdout,
                   "generation_debug.qref_summary.layer%zu.threshold=%g q2_first_gen=%d "
                   "q3_first_gen=%d\n",
                   layer,
                   qref_thresholds[threshold_index],
                   q2_first_generated[layer][threshold_index],
                   q3_first_generated[layer][threshold_index]);
    }
  }
}

void dump_generation_q23_stage_debug(const generation_load_state & state,
                                     const emel::paritychecker::parity_options & opts,
                                     const generation_result & emel_result,
                                     const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.q23_stage: tokenize failed\n");
    return;
  }
  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx == nullptr) {
    std::fprintf(stdout, "generation_debug.q23_stage: reference context failed\n");
    return;
  }

  emel::generator::detail::native_backend actor_backend = {};
  emel::generator::detail::native_backend q23_backend = {};
  emel::generator::detail::native_backend q3_reference_backend = {};
  if (emel::generator::detail::prepare(actor_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q23_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.q23_stage: backend prepare failed\n");
    return;
  }

  const exact_matmul_mode scalar_quant_q23{
      .attention = true,
      .ffn = true,
      .output = true,
      .dtype_mask =
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q2_k)) |
          (1u << static_cast<uint8_t>(emel::kernel::event::dtype::q3_k)),
      .use_scalar_quantized = true,
  };
  const exact_matmul_mode reference_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_reference_q8 = true,
  };
  const int32_t token_id = static_cast<int32_t>(prompt_tokens.front());
  const int32_t position = 0;
  const int32_t layer_index = 0;
  llama_token reference_token = static_cast<llama_token>(token_id);
  llama_batch reference_batch = llama_batch_get_one(&reference_token, 1);
  if (llama_decode(reference_ctx.get(), reference_batch) != 0) {
    std::fprintf(stdout, "generation_debug.q23_stage: reference decode failed\n");
    return;
  }

  if (!emel::generator::detail::copy_tensor_row(
          *actor_backend.token_embedding.tensor, token_id, actor_backend.hidden) ||
      !emel::generator::detail::copy_tensor_row(
          *q23_backend.token_embedding.tensor, token_id, q23_backend.hidden) ||
      !emel::generator::detail::copy_tensor_row(
          *q3_reference_backend.token_embedding.tensor, token_id, q3_reference_backend.hidden)) {
    std::fprintf(stdout, "generation_debug.q23_stage: token embedding replay failed\n");
    return;
  }

  auto dump_stage = [&](const char * label,
                        const auto actor_values,
                        const auto q23_values) {
    std::fprintf(stdout,
                 "generation_debug.q23_stage.%s: max_abs=%g\n",
                 label,
                 max_abs_diff(actor_values, q23_values));
  };

  auto & actor_block = actor_backend.blocks[static_cast<size_t>(layer_index)];
  auto & q23_block = q23_backend.blocks[static_cast<size_t>(layer_index)];
  if (!emel::generator::detail::rms_norm(
          actor_backend.hidden,
          actor_block.attention_norm,
          actor_backend.rms_epsilon,
          actor_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q23_backend.hidden,
          q23_block.attention_norm,
          q23_backend.rms_epsilon,
          q23_backend.norm)) {
    std::fprintf(stdout, "generation_debug.q23_stage: attn rms_norm failed\n");
    return;
  }
  dump_stage("prompt0.layer0.attn_norm", actor_backend.norm, q23_backend.norm);

  if (!emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_q, actor_backend.norm, actor_backend.q) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.attention_q,
                          q23_backend.norm,
                          q23_backend.q,
                          scalar_quant_q23.attention,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized) ||
      !emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_k, actor_backend.norm, actor_backend.k) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.attention_k,
                          q23_backend.norm,
                          q23_backend.k,
                          scalar_quant_q23.attention,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized) ||
      !emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_v, actor_backend.norm, actor_backend.v) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.attention_v,
                          q23_backend.norm,
                          q23_backend.v,
                          scalar_quant_q23.attention,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized)) {
    std::fprintf(stdout, "generation_debug.q23_stage: qkv matmul failed\n");
    return;
  }
  dump_q8_quantize_compare("generation_debug.q23_stage.prompt0.layer0.attn_norm_q8",
                           actor_backend.norm);
  dump_stage("prompt0.layer0.q_pre_rope", actor_backend.q, q23_backend.q);
  dump_stage("prompt0.layer0.k_pre_rope", actor_backend.k, q23_backend.k);
  dump_stage("prompt0.layer0.v_pre_cache", actor_backend.v, q23_backend.v);
  std::vector<float> reference_q(actor_backend.q.size());
  std::vector<float> reference_k(actor_backend.k.size());
  std::vector<float> reference_v(actor_backend.v.size());
  if (matmul_vector_reference_q8(actor_block.attention_q, actor_backend.norm, reference_q) &&
      matmul_vector_reference_q8(actor_block.attention_k, actor_backend.norm, reference_k) &&
      matmul_vector_reference_q8(actor_block.attention_v, actor_backend.norm, reference_v)) {
    dump_stage("prompt0.layer0.q_pre_rope_actor_vs_reference_q8",
               actor_backend.q,
               reference_q);
    dump_stage("prompt0.layer0.q_pre_rope_q23_vs_reference_q8",
               q23_backend.q,
               reference_q);
    dump_stage("prompt0.layer0.k_pre_rope_actor_vs_reference_q8",
               actor_backend.k,
               reference_k);
    dump_stage("prompt0.layer0.k_pre_rope_q23_vs_reference_q8",
               q23_backend.k,
               reference_k);
    dump_stage("prompt0.layer0.v_pre_cache_actor_vs_reference_q8",
               actor_backend.v,
               reference_v);
    dump_stage("prompt0.layer0.v_pre_cache_q23_vs_reference_q8",
               q23_backend.v,
               reference_v);
  }

  emel::generator::detail::apply_rope(
      actor_backend.q,
      actor_backend.n_head,
      actor_backend.head_dim,
      actor_backend.n_rot,
      position,
      actor_backend.rope_freq_base);
  emel::generator::detail::apply_rope(
      actor_backend.k,
      actor_backend.n_head_kv,
      actor_backend.head_dim_kv,
      actor_backend.n_rot,
      position,
      actor_backend.rope_freq_base);
  emel::generator::detail::apply_rope(
      q23_backend.q,
      q23_backend.n_head,
      q23_backend.head_dim,
      q23_backend.n_rot,
      position,
      q23_backend.rope_freq_base);
  emel::generator::detail::apply_rope(
      q23_backend.k,
      q23_backend.n_head_kv,
      q23_backend.head_dim_kv,
      q23_backend.n_rot,
      position,
      q23_backend.rope_freq_base);
  dump_stage("prompt0.layer0.q", actor_backend.q, q23_backend.q);
  dump_stage("prompt0.layer0.k", actor_backend.k, q23_backend.k);

  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(actor_backend.q.data(), static_cast<size_t>(actor_backend.n_embd)),
      actor_backend.q_attn.data());
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(q23_backend.q.data(), static_cast<size_t>(q23_backend.n_embd)),
      q23_backend.q_attn.data());
  dump_stage("prompt0.layer0.q_attn", actor_backend.q_attn, q23_backend.q_attn);

  const int32_t kv_dim = actor_backend.n_head_kv * actor_backend.head_dim_kv;
  const size_t actor_cache_offset =
      emel::generator::detail::layer_cache_offset(actor_backend, layer_index, position, kv_dim);
  const size_t q23_cache_offset =
      emel::generator::detail::layer_cache_offset(q23_backend, layer_index, position, kv_dim);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(actor_backend.k.data(), static_cast<size_t>(kv_dim)),
      actor_backend.key_cache.data() + actor_cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(actor_backend.v.data(), static_cast<size_t>(kv_dim)),
      actor_backend.value_cache.data() + actor_cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(q23_backend.k.data(), static_cast<size_t>(kv_dim)),
      q23_backend.key_cache.data() + q23_cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(q23_backend.v.data(), static_cast<size_t>(kv_dim)),
      q23_backend.value_cache.data() + q23_cache_offset);
  dump_stage("prompt0.layer0.key_cache",
             std::span<const uint16_t>(actor_backend.key_cache.data() + actor_cache_offset,
                                    static_cast<size_t>(kv_dim)),
             std::span<const uint16_t>(q23_backend.key_cache.data() + q23_cache_offset,
                                    static_cast<size_t>(kv_dim)));
  dump_stage("prompt0.layer0.value_cache",
             std::span<const uint16_t>(actor_backend.value_cache.data() + actor_cache_offset,
                                    static_cast<size_t>(kv_dim)),
             std::span<const uint16_t>(q23_backend.value_cache.data() + q23_cache_offset,
                                    static_cast<size_t>(kv_dim)));
  std::vector<float> reference_value_cache_rows;
  if (capture_reference_value_cache_rows(reference_ctx.get(), layer_index, reference_value_cache_rows) &&
      reference_value_cache_rows.size() >= static_cast<size_t>(kv_dim)) {
    const std::span<const float> reference_value_cache_row{
        reference_value_cache_rows.data(),
        static_cast<size_t>(kv_dim),
    };
    dump_state_compare(
        "generation_debug.q23_stage.prompt0.layer0.value_cache_vs_reference",
        std::span<const uint16_t>(actor_backend.value_cache.data() + actor_cache_offset,
                               static_cast<size_t>(kv_dim)),
        reference_value_cache_row);
    dump_state_compare(
        "generation_debug.q23_stage.prompt0.layer0.value_cache_q23_vs_reference",
        std::span<const uint16_t>(q23_backend.value_cache.data() + q23_cache_offset,
                               static_cast<size_t>(kv_dim)),
        reference_value_cache_row);
    if (run_layer_with_matmul_mode_scalar_attention(
            q3_reference_backend, layer_index, position, reference_q3_only)) {
      const size_t q3_reference_cache_offset = emel::generator::detail::layer_cache_offset(
          q3_reference_backend, layer_index, position, kv_dim);
      dump_state_compare(
          "generation_debug.q23_stage.prompt0.layer0.value_cache_q3_reference_vs_reference",
          std::span<const uint16_t>(q3_reference_backend.value_cache.data() + q3_reference_cache_offset,
                                 static_cast<size_t>(kv_dim)),
          reference_value_cache_row);
      dump_state_compare(
          "generation_debug.q23_stage.prompt0.layer0.value_cache_q3_reference_vs_actor",
          std::span<const uint16_t>(q3_reference_backend.value_cache.data() + q3_reference_cache_offset,
                                 static_cast<size_t>(kv_dim)),
          std::span<const uint16_t>(actor_backend.value_cache.data() + actor_cache_offset,
                                 static_cast<size_t>(kv_dim)));
    } else {
      std::fprintf(stdout,
                   "generation_debug.q23_stage.prompt0.layer0.value_cache_q3_reference_vs_reference: replay failed\n");
    }
  } else {
    std::fprintf(stdout,
                 "generation_debug.q23_stage.prompt0.layer0.value_cache_vs_reference: unavailable\n");
  }

  if (!emel::generator::detail::compute_attention(
          actor_backend, layer_index, position + 1, actor_backend.q_attn) ||
      !emel::generator::detail::compute_attention(
          q23_backend, layer_index, position + 1, q23_backend.q_attn)) {
    std::fprintf(stdout, "generation_debug.q23_stage: attention failed\n");
    return;
  }
  dump_stage("prompt0.layer0.kqv_out", actor_backend.attn_ctx, q23_backend.attn_ctx);

  if (!emel::generator::detail::matmul_vector(
          actor_backend,
          actor_block.attention_output,
          actor_backend.attn_ctx,
          actor_backend.projected) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.attention_output,
                          q23_backend.attn_ctx,
                          q23_backend.projected,
                          scalar_quant_q23.attention,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized)) {
    std::fprintf(stdout, "generation_debug.q23_stage: attention output matmul failed\n");
    return;
  }
  dump_stage("prompt0.layer0.attn_out", actor_backend.projected, q23_backend.projected);

  for (int32_t idx = 0; idx < actor_backend.n_embd; ++idx) {
    actor_backend.hidden[static_cast<size_t>(idx)] += actor_backend.projected[static_cast<size_t>(idx)];
    q23_backend.hidden[static_cast<size_t>(idx)] += q23_backend.projected[static_cast<size_t>(idx)];
  }
  dump_stage("prompt0.layer0.attn_residual", actor_backend.hidden, q23_backend.hidden);

  if (!emel::generator::detail::rms_norm(
          actor_backend.hidden,
          actor_block.feed_forward_norm,
          actor_backend.rms_epsilon,
          actor_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q23_backend.hidden,
          q23_block.feed_forward_norm,
          q23_backend.rms_epsilon,
          q23_backend.norm)) {
    std::fprintf(stdout, "generation_debug.q23_stage: ffn rms_norm failed\n");
    return;
  }
  dump_stage("prompt0.layer0.ffn_norm", actor_backend.norm, q23_backend.norm);

  if (!emel::generator::detail::matmul_vector(
          actor_backend, actor_block.feed_forward_gate, actor_backend.norm, actor_backend.gate) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.feed_forward_gate,
                          q23_backend.norm,
                          q23_backend.gate,
                          scalar_quant_q23.ffn,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized) ||
      !emel::generator::detail::matmul_vector(
          actor_backend, actor_block.feed_forward_up, actor_backend.norm, actor_backend.up) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.feed_forward_up,
                          q23_backend.norm,
                          q23_backend.up,
                          scalar_quant_q23.ffn,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized)) {
    std::fprintf(stdout, "generation_debug.q23_stage: ffn gate/up matmul failed\n");
    return;
  }
  dump_stage("prompt0.layer0.ffn_gate", actor_backend.gate, q23_backend.gate);
  dump_stage("prompt0.layer0.ffn_up", actor_backend.up, q23_backend.up);

  for (size_t idx = 0; idx < actor_backend.gate.size(); ++idx) {
    actor_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(actor_backend.gate[idx]) * actor_backend.up[idx];
    q23_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(q23_backend.gate[idx]) * q23_backend.up[idx];
  }
  dump_stage("prompt0.layer0.ffn_swiglu", actor_backend.ffn_hidden, q23_backend.ffn_hidden);

  if (!emel::generator::detail::matmul_vector(
          actor_backend,
          actor_block.feed_forward_down,
          actor_backend.ffn_hidden,
          actor_backend.projected) ||
      !matmul_vector_mode(q23_backend,
                          q23_block.feed_forward_down,
                          q23_backend.ffn_hidden,
                          q23_backend.projected,
                          scalar_quant_q23.ffn,
                          scalar_quant_q23.only_dtype,
                          scalar_quant_q23.dtype_mask,
                          scalar_quant_q23.use_reference_q8,
                          scalar_quant_q23.use_scalar_quantized)) {
    std::fprintf(stdout, "generation_debug.q23_stage: ffn down matmul failed\n");
    return;
  }
  dump_stage("prompt0.layer0.ffn_out", actor_backend.projected, q23_backend.projected);

  for (int32_t idx = 0; idx < actor_backend.n_embd; ++idx) {
    actor_backend.hidden[static_cast<size_t>(idx)] += actor_backend.projected[static_cast<size_t>(idx)];
    q23_backend.hidden[static_cast<size_t>(idx)] += q23_backend.projected[static_cast<size_t>(idx)];
  }
  dump_stage("prompt0.layer0.l_out", actor_backend.hidden, q23_backend.hidden);
}

void dump_generation_reference_q_stage_debug(const generation_load_state & state,
                                             const emel::paritychecker::parity_options & opts,
                                             const generation_result & emel_result,
                                             const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.qref_stage: tokenize failed\n");
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
    std::fprintf(stdout, "generation_debug.qref_stage: empty prefix\n");
    return;
  }

  emel::generator::detail::native_backend actor_backend = {};
  emel::generator::detail::native_backend q2_exact_backend = {};
  emel::generator::detail::native_backend q2_reference_backend = {};
  emel::generator::detail::native_backend q3_reference_backend = {};
  emel::generator::detail::native_backend q3_scalar_backend = {};
  if (emel::generator::detail::prepare(actor_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_exact_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q2_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_reference_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(q3_scalar_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.qref_stage: backend prepare failed\n");
    return;
  }

  const exact_matmul_mode exact_q2_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q2_k),
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
  const exact_matmul_mode scalar_quant_q3_only{
      .attention = true,
      .ffn = true,
      .output = true,
      .only_dtype = static_cast<uint8_t>(emel::kernel::event::dtype::q3_k),
      .use_scalar_quantized = true,
  };

  constexpr float target_q2_threshold = 1.0e-4f;
  int32_t target_generated_index = token_mismatch_index - 1;
  if (target_generated_index > 0) {
    emel::generator::detail::native_backend scan_actor_backend = {};
    emel::generator::detail::native_backend scan_q2_reference_backend = {};
    emel::generator::detail::native_backend scan_q3_reference_backend = {};
    if (emel::generator::detail::prepare(scan_actor_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none) &&
        emel::generator::detail::prepare(scan_q2_reference_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none) &&
        emel::generator::detail::prepare(scan_q3_reference_backend, *state.model_data) ==
            emel::error::cast(emel::model::loader::error::none)) {
      const size_t prompt_count = prompt_tokens.size();
      for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
        const int32_t scan_token_id = prefix_tokens[token_index];
        const int32_t scan_position = static_cast<int32_t>(token_index);
        if (!emel::generator::detail::copy_tensor_row(
                *scan_actor_backend.token_embedding.tensor, scan_token_id, scan_actor_backend.hidden) ||
            !emel::generator::detail::copy_tensor_row(*scan_q2_reference_backend.token_embedding.tensor,
                                                      scan_token_id,
                                                      scan_q2_reference_backend.hidden) ||
            !emel::generator::detail::copy_tensor_row(*scan_q3_reference_backend.token_embedding.tensor,
                                                      scan_token_id,
                                                      scan_q3_reference_backend.hidden)) {
          break;
        }

        const bool is_prompt_token = token_index < prompt_count;
        const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
        bool found_target = false;
        for (int32_t layer = 0; layer < scan_actor_backend.n_layer; ++layer) {
          if (!run_layer_with_scalar_attention(scan_actor_backend, layer, scan_position) ||
              !run_layer_with_matmul_mode_scalar_attention(
                  scan_q2_reference_backend, layer, scan_position, reference_q2_only) ||
              !run_layer_with_matmul_mode_scalar_attention(
                  scan_q3_reference_backend, layer, scan_position, reference_q3_only)) {
            found_target = false;
            break;
          }

          if (!is_prompt_token && layer == 1 &&
              max_abs_diff(scan_actor_backend.hidden, scan_q2_reference_backend.hidden) >
                  target_q2_threshold) {
            target_generated_index = generated_index;
            found_target = true;
            break;
          }
        }

        scan_actor_backend.kv_cache_tokens = scan_position + 1;
        scan_q2_reference_backend.kv_cache_tokens = scan_position + 1;
        scan_q3_reference_backend.kv_cache_tokens = scan_position + 1;
        if (found_target) {
          break;
        }
      }
    }
  }

  const int32_t target_prefix_token_count =
      static_cast<int32_t>(prompt_tokens.size()) + target_generated_index + 1;
  if (target_prefix_token_count <= 0 ||
      target_prefix_token_count > static_cast<int32_t>(prefix_tokens.size())) {
    std::fprintf(stdout, "generation_debug.qref_stage: invalid target prefix\n");
    return;
  }

  if (target_prefix_token_count > 1) {
    const std::span<const int32_t> prior_tokens{
        prefix_tokens.data(), static_cast<size_t>(target_prefix_token_count - 1)};
    if (!run_prefill_with_scalar_attention(actor_backend, prior_tokens) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q2_exact_backend, prior_tokens, exact_q2_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q2_reference_backend, prior_tokens, reference_q2_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q3_reference_backend, prior_tokens, reference_q3_only) ||
        !run_prefill_with_scalar_attention_matmul_mode(
            q3_scalar_backend, prior_tokens, scalar_quant_q3_only)) {
      std::fprintf(stdout, "generation_debug.qref_stage: prior prefix replay failed\n");
      return;
    }
  } else {
    actor_backend.kv_cache_tokens = 0;
    q2_exact_backend.kv_cache_tokens = 0;
    q2_reference_backend.kv_cache_tokens = 0;
    q3_reference_backend.kv_cache_tokens = 0;
    q3_scalar_backend.kv_cache_tokens = 0;
  }

  const int32_t token_id =
      prefix_tokens[static_cast<size_t>(target_prefix_token_count - 1)];
  const int32_t position = target_prefix_token_count - 1;
  const int32_t generated_index = target_generated_index;
  const int32_t layer_index = 1;
  if (actor_backend.n_layer <= layer_index) {
    std::fprintf(stdout, "generation_debug.qref_stage: layer%d unavailable\n", layer_index);
    return;
  }

  if (!emel::generator::detail::copy_tensor_row(
          *actor_backend.token_embedding.tensor, token_id, actor_backend.hidden) ||
      !emel::generator::detail::copy_tensor_row(
          *q2_exact_backend.token_embedding.tensor, token_id, q2_exact_backend.hidden) ||
      !emel::generator::detail::copy_tensor_row(
          *q2_reference_backend.token_embedding.tensor, token_id, q2_reference_backend.hidden) ||
      !emel::generator::detail::copy_tensor_row(
          *q3_reference_backend.token_embedding.tensor, token_id, q3_reference_backend.hidden) ||
      !emel::generator::detail::copy_tensor_row(
          *q3_scalar_backend.token_embedding.tensor, token_id, q3_scalar_backend.hidden)) {
    std::fprintf(stdout, "generation_debug.qref_stage: token embedding replay failed\n");
    return;
  }

  auto dump_stage = [&](const char * label,
                        const auto actor_values,
                        const auto q2_exact_values,
                        const auto q2_values,
                        const auto q3_values,
                        const auto q3_scalar_values) {
    std::fprintf(stdout,
                 "generation_debug.qref_stage.%s.q2_exact: max_abs=%g\n",
                 label,
                 max_abs_diff(actor_values, q2_exact_values));
    std::fprintf(stdout,
                 "generation_debug.qref_stage.%s.q2_reference: max_abs=%g\n",
                 label,
                 max_abs_diff(actor_values, q2_values));
    std::fprintf(stdout,
                 "generation_debug.qref_stage.%s.q3_reference: max_abs=%g\n",
                 label,
                 max_abs_diff(actor_values, q3_values));
    std::fprintf(stdout,
                 "generation_debug.qref_stage.%s.q3_scalar: max_abs=%g\n",
                 label,
                 max_abs_diff(actor_values, q3_scalar_values));
  };

  if (!run_layer_with_scalar_attention(actor_backend, 0, position) ||
      !run_layer_with_matmul_mode_scalar_attention(
          q2_exact_backend, 0, position, exact_q2_only) ||
      !run_layer_with_matmul_mode_scalar_attention(
          q2_reference_backend, 0, position, reference_q2_only) ||
      !run_layer_with_matmul_mode_scalar_attention(
          q3_reference_backend, 0, position, reference_q3_only) ||
      !run_layer_with_matmul_mode_scalar_attention(
          q3_scalar_backend, 0, position, scalar_quant_q3_only)) {
    std::fprintf(stdout, "generation_debug.qref_stage: layer0 replay failed\n");
    return;
  }

  const std::string stage_prefix =
      "gen" + std::to_string(generated_index) + ".layer" + std::to_string(layer_index);
  dump_stage((stage_prefix + ".input_hidden").c_str(),
             actor_backend.hidden,
             q2_exact_backend.hidden,
             q2_reference_backend.hidden,
             q3_reference_backend.hidden,
             q3_scalar_backend.hidden);

  auto & actor_block = actor_backend.blocks[static_cast<size_t>(layer_index)];
  auto & q2_exact_block = q2_exact_backend.blocks[static_cast<size_t>(layer_index)];
  auto & q2_block = q2_reference_backend.blocks[static_cast<size_t>(layer_index)];
  auto & q3_block = q3_reference_backend.blocks[static_cast<size_t>(layer_index)];
  auto & q3_scalar_block = q3_scalar_backend.blocks[static_cast<size_t>(layer_index)];
  if (!emel::generator::detail::rms_norm(
          actor_backend.hidden,
          actor_block.attention_norm,
          actor_backend.rms_epsilon,
          actor_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q2_exact_backend.hidden,
          q2_exact_block.attention_norm,
          q2_exact_backend.rms_epsilon,
          q2_exact_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q2_reference_backend.hidden,
          q2_block.attention_norm,
          q2_reference_backend.rms_epsilon,
          q2_reference_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q3_reference_backend.hidden,
          q3_block.attention_norm,
          q3_reference_backend.rms_epsilon,
          q3_reference_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q3_scalar_backend.hidden,
          q3_scalar_block.attention_norm,
          q3_scalar_backend.rms_epsilon,
          q3_scalar_backend.norm)) {
    std::fprintf(stdout, "generation_debug.qref_stage: attn rms_norm failed\n");
    return;
  }
  dump_stage((stage_prefix + ".attn_norm").c_str(),
             actor_backend.norm,
             q2_exact_backend.norm,
             q2_reference_backend.norm,
             q3_reference_backend.norm,
             q3_scalar_backend.norm);
  dump_q8_quantize_compare((stage_prefix + ".attn_norm_q8").c_str(), actor_backend.norm);
  dump_matrix_compare(
      (stage_prefix + ".attn_q_matmul").c_str(), actor_backend, actor_block.attention_q, actor_backend.norm);
  dump_matrix_compare_reference_q8((stage_prefix + ".attn_q_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.attention_q,
                                   actor_backend.norm);
  dump_matrix_compare(
      (stage_prefix + ".attn_k_matmul").c_str(), actor_backend, actor_block.attention_k, actor_backend.norm);
  dump_matrix_compare_reference_q8((stage_prefix + ".attn_k_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.attention_k,
                                   actor_backend.norm);
  dump_matrix_compare(
      (stage_prefix + ".attn_v_matmul").c_str(), actor_backend, actor_block.attention_v, actor_backend.norm);
  dump_matrix_compare_reference_q8((stage_prefix + ".attn_v_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.attention_v,
                                   actor_backend.norm);

  const auto run_mode_matmul = [&](emel::generator::detail::native_backend & backend,
                                   const emel::generator::detail::tensor_matrix & matrix,
                                   std::span<const float> input,
                                   std::span<float> output,
                                   const exact_matmul_mode mode,
                                   const bool attention_mode) {
    return matmul_vector_mode(backend,
                              matrix,
                              input,
                              output,
                              attention_mode ? mode.attention : mode.ffn,
                              mode.only_dtype,
                              mode.dtype_mask,
                              mode.use_reference_q8,
                              mode.use_scalar_quantized);
  };

  if (!emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_q, actor_backend.norm, actor_backend.q) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.attention_q,
                       q2_exact_backend.norm,
                       q2_exact_backend.q,
                       exact_q2_only,
                       true) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.attention_q,
                       q2_reference_backend.norm,
                       q2_reference_backend.q,
                       reference_q2_only,
                       true) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.attention_q,
                       q3_reference_backend.norm,
                       q3_reference_backend.q,
                       reference_q3_only,
                       true) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.attention_q,
                       q3_scalar_backend.norm,
                       q3_scalar_backend.q,
                       scalar_quant_q3_only,
                       true) ||
      !emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_k, actor_backend.norm, actor_backend.k) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.attention_k,
                       q2_exact_backend.norm,
                       q2_exact_backend.k,
                       exact_q2_only,
                       true) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.attention_k,
                       q2_reference_backend.norm,
                       q2_reference_backend.k,
                       reference_q2_only,
                       true) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.attention_k,
                       q3_reference_backend.norm,
                       q3_reference_backend.k,
                       reference_q3_only,
                       true) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.attention_k,
                       q3_scalar_backend.norm,
                       q3_scalar_backend.k,
                       scalar_quant_q3_only,
                       true) ||
      !emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_v, actor_backend.norm, actor_backend.v) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.attention_v,
                       q2_exact_backend.norm,
                       q2_exact_backend.v,
                       exact_q2_only,
                       true) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.attention_v,
                       q2_reference_backend.norm,
                       q2_reference_backend.v,
                       reference_q2_only,
                       true) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.attention_v,
                       q3_reference_backend.norm,
                       q3_reference_backend.v,
                       reference_q3_only,
                       true) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.attention_v,
                       q3_scalar_backend.norm,
                       q3_scalar_backend.v,
                       scalar_quant_q3_only,
                       true)) {
    std::fprintf(stdout, "generation_debug.qref_stage: qkv matmul failed\n");
    return;
  }
  dump_stage((stage_prefix + ".q_pre_rope").c_str(),
             actor_backend.q,
             q2_exact_backend.q,
             q2_reference_backend.q,
             q3_reference_backend.q,
             q3_scalar_backend.q);
  dump_stage((stage_prefix + ".k_pre_rope").c_str(),
             actor_backend.k,
             q2_exact_backend.k,
             q2_reference_backend.k,
             q3_reference_backend.k,
             q3_scalar_backend.k);
  dump_stage((stage_prefix + ".v_pre_cache").c_str(),
             actor_backend.v,
             q2_exact_backend.v,
             q2_reference_backend.v,
             q3_reference_backend.v,
             q3_scalar_backend.v);

  auto apply_rope_and_cache = [&](emel::generator::detail::native_backend & backend) {
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
  };
  apply_rope_and_cache(actor_backend);
  apply_rope_and_cache(q2_exact_backend);
  apply_rope_and_cache(q2_reference_backend);
  apply_rope_and_cache(q3_reference_backend);
  apply_rope_and_cache(q3_scalar_backend);
  dump_stage((stage_prefix + ".q").c_str(),
             actor_backend.q,
             q2_exact_backend.q,
             q2_reference_backend.q,
             q3_reference_backend.q,
             q3_scalar_backend.q);
  dump_stage((stage_prefix + ".k").c_str(),
             actor_backend.k,
             q2_exact_backend.k,
             q2_reference_backend.k,
             q3_reference_backend.k,
             q3_scalar_backend.k);
  dump_stage((stage_prefix + ".q_attn").c_str(),
             actor_backend.q_attn,
             q2_exact_backend.q_attn,
             q2_reference_backend.q_attn,
             q3_reference_backend.q_attn,
             q3_scalar_backend.q_attn);
  dump_stage((stage_prefix + ".key_cache").c_str(),
             actor_backend.key_cache,
             q2_exact_backend.key_cache,
             q2_reference_backend.key_cache,
             q3_reference_backend.key_cache,
             q3_scalar_backend.key_cache);
  dump_stage((stage_prefix + ".value_cache").c_str(),
             actor_backend.value_cache,
             q2_exact_backend.value_cache,
             q2_reference_backend.value_cache,
             q3_reference_backend.value_cache,
             q3_scalar_backend.value_cache);

  if (!emel::generator::detail::compute_attention(
          actor_backend, layer_index, position + 1, actor_backend.q_attn) ||
      !emel::generator::detail::compute_attention(
          q2_exact_backend, layer_index, position + 1, q2_exact_backend.q_attn) ||
      !emel::generator::detail::compute_attention(
          q2_reference_backend, layer_index, position + 1, q2_reference_backend.q_attn) ||
      !emel::generator::detail::compute_attention(
          q3_reference_backend, layer_index, position + 1, q3_reference_backend.q_attn) ||
      !emel::generator::detail::compute_attention(
          q3_scalar_backend, layer_index, position + 1, q3_scalar_backend.q_attn)) {
    std::fprintf(stdout, "generation_debug.qref_stage: attention failed\n");
    return;
  }
  dump_stage((stage_prefix + ".kqv_out").c_str(),
             actor_backend.attn_ctx,
             q2_exact_backend.attn_ctx,
             q2_reference_backend.attn_ctx,
             q3_reference_backend.attn_ctx,
             q3_scalar_backend.attn_ctx);
  dump_q8_quantize_compare((stage_prefix + ".attn_ctx_q8").c_str(), actor_backend.attn_ctx);
  dump_matrix_compare((stage_prefix + ".attn_out_matmul").c_str(),
                      actor_backend,
                      actor_block.attention_output,
                      actor_backend.attn_ctx);
  dump_matrix_compare_reference_q8((stage_prefix + ".attn_out_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.attention_output,
                                   actor_backend.attn_ctx);

  if (!emel::generator::detail::matmul_vector(
          actor_backend, actor_block.attention_output, actor_backend.attn_ctx, actor_backend.projected) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.attention_output,
                       q2_exact_backend.attn_ctx,
                       q2_exact_backend.projected,
                       exact_q2_only,
                       true) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.attention_output,
                       q2_reference_backend.attn_ctx,
                       q2_reference_backend.projected,
                       reference_q2_only,
                       true) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.attention_output,
                       q3_reference_backend.attn_ctx,
                       q3_reference_backend.projected,
                       reference_q3_only,
                       true) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.attention_output,
                       q3_scalar_backend.attn_ctx,
                       q3_scalar_backend.projected,
                       scalar_quant_q3_only,
                       true)) {
    std::fprintf(stdout, "generation_debug.qref_stage: attention output matmul failed\n");
    return;
  }
  dump_stage((stage_prefix + ".attn_out").c_str(),
             actor_backend.projected,
             q2_exact_backend.projected,
             q2_reference_backend.projected,
             q3_reference_backend.projected,
             q3_scalar_backend.projected);

  for (int32_t idx = 0; idx < actor_backend.n_embd; ++idx) {
    actor_backend.hidden[static_cast<size_t>(idx)] += actor_backend.projected[static_cast<size_t>(idx)];
    q2_exact_backend.hidden[static_cast<size_t>(idx)] +=
        q2_exact_backend.projected[static_cast<size_t>(idx)];
    q2_reference_backend.hidden[static_cast<size_t>(idx)] +=
        q2_reference_backend.projected[static_cast<size_t>(idx)];
    q3_reference_backend.hidden[static_cast<size_t>(idx)] +=
        q3_reference_backend.projected[static_cast<size_t>(idx)];
    q3_scalar_backend.hidden[static_cast<size_t>(idx)] +=
        q3_scalar_backend.projected[static_cast<size_t>(idx)];
  }
  dump_stage((stage_prefix + ".attn_residual").c_str(),
             actor_backend.hidden,
             q2_exact_backend.hidden,
             q2_reference_backend.hidden,
             q3_reference_backend.hidden,
             q3_scalar_backend.hidden);

  if (!emel::generator::detail::rms_norm(
          actor_backend.hidden,
          actor_block.feed_forward_norm,
          actor_backend.rms_epsilon,
          actor_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q2_exact_backend.hidden,
          q2_exact_block.feed_forward_norm,
          q2_exact_backend.rms_epsilon,
          q2_exact_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q2_reference_backend.hidden,
          q2_block.feed_forward_norm,
          q2_reference_backend.rms_epsilon,
          q2_reference_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q3_reference_backend.hidden,
          q3_block.feed_forward_norm,
          q3_reference_backend.rms_epsilon,
          q3_reference_backend.norm) ||
      !emel::generator::detail::rms_norm(
          q3_scalar_backend.hidden,
          q3_scalar_block.feed_forward_norm,
          q3_scalar_backend.rms_epsilon,
          q3_scalar_backend.norm)) {
    std::fprintf(stdout, "generation_debug.qref_stage: ffn rms_norm failed\n");
    return;
  }
  dump_stage((stage_prefix + ".ffn_norm").c_str(),
             actor_backend.norm,
             q2_exact_backend.norm,
             q2_reference_backend.norm,
             q3_reference_backend.norm,
             q3_scalar_backend.norm);
  dump_q8_quantize_compare((stage_prefix + ".ffn_norm_q8").c_str(), actor_backend.norm);
  dump_matrix_compare((stage_prefix + ".ffn_gate_matmul").c_str(),
                      actor_backend,
                      actor_block.feed_forward_gate,
                      actor_backend.norm);
  dump_matrix_compare_reference_q8((stage_prefix + ".ffn_gate_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.feed_forward_gate,
                                   actor_backend.norm);
  dump_matrix_compare((stage_prefix + ".ffn_up_matmul").c_str(),
                      actor_backend,
                      actor_block.feed_forward_up,
                      actor_backend.norm);
  dump_matrix_compare_reference_q8((stage_prefix + ".ffn_up_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.feed_forward_up,
                                   actor_backend.norm);

  if (!emel::generator::detail::matmul_vector(
          actor_backend, actor_block.feed_forward_gate, actor_backend.norm, actor_backend.gate) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.feed_forward_gate,
                       q2_exact_backend.norm,
                       q2_exact_backend.gate,
                       exact_q2_only,
                       false) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.feed_forward_gate,
                       q2_reference_backend.norm,
                       q2_reference_backend.gate,
                       reference_q2_only,
                       false) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.feed_forward_gate,
                       q3_reference_backend.norm,
                       q3_reference_backend.gate,
                       reference_q3_only,
                       false) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.feed_forward_gate,
                       q3_scalar_backend.norm,
                       q3_scalar_backend.gate,
                       scalar_quant_q3_only,
                       false) ||
      !emel::generator::detail::matmul_vector(
          actor_backend, actor_block.feed_forward_up, actor_backend.norm, actor_backend.up) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.feed_forward_up,
                       q2_exact_backend.norm,
                       q2_exact_backend.up,
                       exact_q2_only,
                       false) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.feed_forward_up,
                       q2_reference_backend.norm,
                       q2_reference_backend.up,
                       reference_q2_only,
                       false) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.feed_forward_up,
                       q3_reference_backend.norm,
                       q3_reference_backend.up,
                       reference_q3_only,
                       false) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.feed_forward_up,
                       q3_scalar_backend.norm,
                       q3_scalar_backend.up,
                       scalar_quant_q3_only,
                       false)) {
    std::fprintf(stdout, "generation_debug.qref_stage: ffn gate/up matmul failed\n");
    return;
  }
  dump_stage((stage_prefix + ".ffn_gate").c_str(),
             actor_backend.gate,
             q2_exact_backend.gate,
             q2_reference_backend.gate,
             q3_reference_backend.gate,
             q3_scalar_backend.gate);
  dump_stage((stage_prefix + ".ffn_up").c_str(),
             actor_backend.up,
             q2_exact_backend.up,
             q2_reference_backend.up,
             q3_reference_backend.up,
             q3_scalar_backend.up);

  for (size_t idx = 0; idx < actor_backend.gate.size(); ++idx) {
    actor_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(actor_backend.gate[idx]) * actor_backend.up[idx];
    q2_exact_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(q2_exact_backend.gate[idx]) * q2_exact_backend.up[idx];
    q2_reference_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(q2_reference_backend.gate[idx]) *
        q2_reference_backend.up[idx];
    q3_reference_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(q3_reference_backend.gate[idx]) *
        q3_reference_backend.up[idx];
    q3_scalar_backend.ffn_hidden[idx] =
        emel::generator::detail::silu(q3_scalar_backend.gate[idx]) *
        q3_scalar_backend.up[idx];
  }
  dump_stage((stage_prefix + ".ffn_swiglu").c_str(),
             actor_backend.ffn_hidden,
             q2_exact_backend.ffn_hidden,
             q2_reference_backend.ffn_hidden,
             q3_reference_backend.ffn_hidden,
             q3_scalar_backend.ffn_hidden);
  dump_q8_quantize_compare((stage_prefix + ".ffn_hidden_q8").c_str(), actor_backend.ffn_hidden);
  dump_matrix_compare((stage_prefix + ".ffn_down_matmul").c_str(),
                      actor_backend,
                      actor_block.feed_forward_down,
                      actor_backend.ffn_hidden);
  dump_matrix_compare_reference_q8((stage_prefix + ".ffn_down_matmul_refq8").c_str(),
                                   actor_backend,
                                   actor_block.feed_forward_down,
                                   actor_backend.ffn_hidden);

  if (!emel::generator::detail::matmul_vector(
          actor_backend, actor_block.feed_forward_down, actor_backend.ffn_hidden, actor_backend.projected) ||
      !run_mode_matmul(q2_exact_backend,
                       q2_exact_block.feed_forward_down,
                       q2_exact_backend.ffn_hidden,
                       q2_exact_backend.projected,
                       exact_q2_only,
                       false) ||
      !run_mode_matmul(q2_reference_backend,
                       q2_block.feed_forward_down,
                       q2_reference_backend.ffn_hidden,
                       q2_reference_backend.projected,
                       reference_q2_only,
                       false) ||
      !run_mode_matmul(q3_reference_backend,
                       q3_block.feed_forward_down,
                       q3_reference_backend.ffn_hidden,
                       q3_reference_backend.projected,
                       reference_q3_only,
                       false) ||
      !run_mode_matmul(q3_scalar_backend,
                       q3_scalar_block.feed_forward_down,
                       q3_scalar_backend.ffn_hidden,
                       q3_scalar_backend.projected,
                       scalar_quant_q3_only,
                       false)) {
    std::fprintf(stdout, "generation_debug.qref_stage: ffn down matmul failed\n");
    return;
  }
  dump_stage((stage_prefix + ".ffn_out").c_str(),
             actor_backend.projected,
             q2_exact_backend.projected,
             q2_reference_backend.projected,
             q3_reference_backend.projected,
             q3_scalar_backend.projected);

  for (int32_t idx = 0; idx < actor_backend.n_embd; ++idx) {
    actor_backend.hidden[static_cast<size_t>(idx)] += actor_backend.projected[static_cast<size_t>(idx)];
    q2_exact_backend.hidden[static_cast<size_t>(idx)] +=
        q2_exact_backend.projected[static_cast<size_t>(idx)];
    q2_reference_backend.hidden[static_cast<size_t>(idx)] +=
        q2_reference_backend.projected[static_cast<size_t>(idx)];
    q3_reference_backend.hidden[static_cast<size_t>(idx)] +=
        q3_reference_backend.projected[static_cast<size_t>(idx)];
    q3_scalar_backend.hidden[static_cast<size_t>(idx)] +=
        q3_scalar_backend.projected[static_cast<size_t>(idx)];
  }
  dump_stage((stage_prefix + ".l_out").c_str(),
             actor_backend.hidden,
             q2_exact_backend.hidden,
             q2_reference_backend.hidden,
             q3_reference_backend.hidden,
             q3_scalar_backend.hidden);
}

void dump_generation_gen0_attention_debug(const generation_load_state & state,
                                          const emel::paritychecker::parity_options & opts,
                                          const generation_result & emel_result,
                                          const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0 ||
      reference_result.trace.token_count <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.gen0: tokenize failed\n");
    return;
  }

  std::vector<int32_t> prefix_tokens;
  prefix_tokens.reserve(prompt_tokens.size() + 1u);
  for (const llama_token token : prompt_tokens) {
    prefix_tokens.push_back(static_cast<int32_t>(token));
  }
  prefix_tokens.push_back(reference_result.trace.token_ids[0]);

  const std::span<const int32_t> generated_prefix{
      prefix_tokens.data() + prompt_tokens.size(),
      prefix_tokens.size() - prompt_tokens.size(),
  };
  reference_graph_capture graph_capture = {};
  if (!capture_reference_graph_for_generation_prefix(
          state, prompt_tokens, generated_prefix, graph_capture)) {
    std::fprintf(stdout, "generation_debug.gen0: reference graph capture failed\n");
    return;
  }

  llama_context_ptr reference_ctx =
      make_reference_context(const_cast<initialize_backend &>(state.backend));
  if (reference_ctx == nullptr ||
      !run_reference_prefix_decode(reference_ctx.get(), prompt_tokens, generated_prefix)) {
    std::fprintf(stdout, "generation_debug.gen0: reference decode replay failed\n");
    return;
  }

  std::vector<float> reference_value_cache_rows;
  if (!capture_reference_value_cache_rows(reference_ctx.get(), 0, reference_value_cache_rows)) {
    reference_value_cache_rows.clear();
  }

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, *state.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.gen0: backend prepare failed\n");
    return;
  }

  const std::span<const int32_t> prior_tokens{prefix_tokens.data(), prefix_tokens.size() - 1u};
  if (!run_prefill_from_token_prefix(backend, prior_tokens)) {
    std::fprintf(stdout, "generation_debug.gen0: prior prefix replay failed\n");
    return;
  }

  const int32_t token_id = prefix_tokens.back();
  const int32_t position = static_cast<int32_t>(prefix_tokens.size() - 1u);
  if (!emel::generator::detail::copy_tensor_row(
          *backend.token_embedding.tensor, token_id, backend.hidden)) {
    std::fprintf(stdout, "generation_debug.gen0: token embedding replay failed\n");
    return;
  }

  auto & block = backend.blocks[0];
  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
      !emel::generator::detail::matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
    std::fprintf(stdout, "generation_debug.gen0: qkv matmul failed\n");
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
      emel::generator::detail::layer_cache_offset(backend, 0, position, kv_dim);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
      backend.key_cache.data() + cache_offset);
  emel::generator::detail::store_fp16_rounded_cache(
      std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
      backend.value_cache.data() + cache_offset);

  auto dump_gen0_state = [&](const char * suffix,
                             const auto emel_values,
                             const char * tensor_name) {
    const std::span<const float> reference_values = reference_last_token_row(
        find_reference_tensor(graph_capture, tensor_name), emel_values.size());
    std::fprintf(stdout,
                 "generation_debug.gen0.layer0.%s: max_abs=%g\n",
                 suffix,
                 max_abs_diff(emel_values, reference_values));
  };

  dump_gen0_state("q", backend.q, "Qcur-0");
  dump_gen0_state("k", backend.k, "Kcur-0");
  dump_gen0_state("v", backend.v, "Vcur-0");
  dump_gen0_state("q_attn", backend.q_attn, "Qcur-0");
  dump_gen0_state("key_cache", std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                                       static_cast<size_t>(kv_dim)),
                  "Kcur-0");
  dump_gen0_state("value_cache_state", std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                                              static_cast<size_t>(kv_dim)),
                  "Vcur-0");

  {
    auto flash_request = emel::generator::detail::make_flash_attn_request(backend, 0, position);
    std::vector<float> shared_flash_ctx(backend.attn_ctx.size(), 0.0f);
    flash_request.dst = emel::generator::detail::make_dst_view_3d(
        shared_flash_ctx.data(),
        flash_request.dst.ne[0],
        flash_request.dst.ne[1],
        flash_request.dst.ne[2]);
    emel::kernel::detail::flash_attn_workspace shared_flash_workspace = {};
    if (emel::kernel::detail::run_flash_attn_ext_with_workspace(
            flash_request, shared_flash_workspace)) {
      dump_gen0_state("kqv_out_shared_flash", shared_flash_ctx, "kqv_out-0");
    } else {
      std::fprintf(stdout, "generation_debug.gen0.layer0.kqv_out_shared_flash: compute failed\n");
    }

#if defined(__aarch64__) || defined(__ARM_NEON)
    std::vector<float> neon_flash_ctx(backend.attn_ctx.size(), 0.0f);
    auto neon_flash_request = flash_request;
    neon_flash_request.dst = emel::generator::detail::make_dst_view_3d(
        neon_flash_ctx.data(),
        neon_flash_request.dst.ne[0],
        neon_flash_request.dst.ne[1],
        neon_flash_request.dst.ne[2]);
    emel::kernel::detail::flash_attn_workspace neon_flash_workspace = {};
    if (emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
            neon_flash_request, true, neon_flash_workspace)) {
      dump_gen0_state("kqv_out_neon_flash", neon_flash_ctx, "kqv_out-0");
    } else {
      std::fprintf(stdout, "generation_debug.gen0.layer0.kqv_out_neon_flash: compute failed\n");
    }
#endif
  }

  (void) compute_attention_with_softmax_debug(
      backend,
      graph_capture,
      0,
      position + 1,
      "generation_debug.gen0.layer0",
      reference_value_cache_rows);

  dump_matrix_compare(
      "generation_debug.gen0.layer0.attn_out_matmul", backend, block.attention_output, backend.attn_ctx);
  if (!emel::generator::detail::matmul_vector(
          backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    std::fprintf(stdout, "generation_debug.gen0: attention output matmul failed\n");
    return;
  }
  dump_gen0_state("attn_out", backend.projected, "attn_out-0");

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }
  dump_gen0_state("ffn_inp", backend.hidden, "ffn_inp-0");

  if (!emel::generator::detail::rms_norm(
          backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm)) {
    std::fprintf(stdout, "generation_debug.gen0: ffn rms_norm failed\n");
    return;
  }
  dump_gen0_state("ffn_norm", backend.norm, "ffn_norm-0");
  dump_matrix_compare(
      "generation_debug.gen0.layer0.ffn_gate_matmul", backend, block.feed_forward_gate, backend.norm);
  dump_matrix_compare(
      "generation_debug.gen0.layer0.ffn_up_matmul", backend, block.feed_forward_up, backend.norm);
  if (!emel::generator::detail::matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
      !emel::generator::detail::matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
    std::fprintf(stdout, "generation_debug.gen0: ffn gate/up matmul failed\n");
    return;
  }
  dump_gen0_state("ffn_gate", backend.gate, "ffn_gate-0");
  dump_gen0_state("ffn_up", backend.up, "ffn_up-0");

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = emel::generator::detail::silu(backend.gate[idx]) * backend.up[idx];
  }
  dump_gen0_state("ffn_swiglu", backend.ffn_hidden, "ffn_swiglu-0");
  dump_matrix_compare(
      "generation_debug.gen0.layer0.ffn_down_matmul", backend, block.feed_forward_down, backend.ffn_hidden);
  if (!emel::generator::detail::matmul_vector(
          backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
    std::fprintf(stdout, "generation_debug.gen0: ffn down matmul failed\n");
    return;
  }
  dump_gen0_state("ffn_out", backend.projected, "ffn_out-0");

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }
  dump_gen0_state("l_out", backend.hidden, "l_out-0");
}

void dump_generation_target_attention_debug(const generation_load_state & state,
                                            const emel::paritychecker::parity_options & opts,
                                            const generation_result & emel_result,
                                            const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens) || prompt_tokens.empty()) {
    std::fprintf(stdout, "generation_debug.target: tokenize failed\n");
    return;
  }

  const auto dump_target_generated_index = [&](const int32_t target_generated_index) {
    if (target_generated_index < 0 || target_generated_index >= reference_result.trace.token_count) {
      return;
    }

    std::vector<int32_t> prefix_tokens;
    prefix_tokens.reserve(prompt_tokens.size() +
                          static_cast<size_t>(target_generated_index + 1));
    for (const llama_token token : prompt_tokens) {
      prefix_tokens.push_back(static_cast<int32_t>(token));
    }
    for (int32_t idx = 0; idx <= target_generated_index; ++idx) {
      prefix_tokens.push_back(reference_result.trace.token_ids[static_cast<size_t>(idx)]);
    }

    const std::span<const int32_t> generated_prefix{
        prefix_tokens.data() + prompt_tokens.size(),
        prefix_tokens.size() - prompt_tokens.size(),
    };
    reference_graph_capture graph_capture = {};
    if (!capture_reference_graph_for_generation_prefix(
            state, prompt_tokens, generated_prefix, graph_capture)) {
      std::fprintf(stdout, "generation_debug.target: reference graph capture failed\n");
      return;
    }

    llama_context_ptr reference_ctx =
        make_reference_context(const_cast<initialize_backend &>(state.backend));
    if (reference_ctx == nullptr ||
        !run_reference_prefix_decode(reference_ctx.get(), prompt_tokens, generated_prefix)) {
      std::fprintf(stdout, "generation_debug.target: reference decode replay failed\n");
      return;
    }

    std::vector<float> reference_key_cache_rows;
    if (!capture_reference_key_cache_rows(reference_ctx.get(), 0, reference_key_cache_rows)) {
      reference_key_cache_rows.clear();
    }
    std::vector<float> reference_value_cache_rows;
    if (!capture_reference_value_cache_rows(reference_ctx.get(), 0, reference_value_cache_rows)) {
      reference_value_cache_rows.clear();
    }

    emel::generator::detail::native_backend backend = {};
    if (emel::generator::detail::prepare(backend, *state.model_data) !=
        emel::error::cast(emel::model::loader::error::none)) {
      std::fprintf(stdout, "generation_debug.target: backend prepare failed\n");
      return;
    }

    const std::span<const int32_t> prior_tokens{prefix_tokens.data(), prefix_tokens.size() - 1u};
    if (!run_prefill_from_token_prefix(backend, prior_tokens)) {
      std::fprintf(stdout, "generation_debug.target: prior prefix replay failed\n");
      return;
    }

    const int32_t token_id = prefix_tokens.back();
    const int32_t position = static_cast<int32_t>(prefix_tokens.size() - 1u);
    if (!emel::generator::detail::copy_tensor_row(
            *backend.token_embedding.tensor, token_id, backend.hidden)) {
      std::fprintf(stdout, "generation_debug.target: token embedding replay failed\n");
      return;
    }

    auto & block = backend.blocks[0];
    if (!emel::generator::detail::rms_norm(
            backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
        !emel::generator::detail::matmul_vector(
            backend, block.attention_q, backend.norm, backend.q) ||
        !emel::generator::detail::matmul_vector(
            backend, block.attention_k, backend.norm, backend.k) ||
        !emel::generator::detail::matmul_vector(
            backend, block.attention_v, backend.norm, backend.v)) {
      std::fprintf(stdout, "generation_debug.target: qkv matmul failed\n");
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
        emel::generator::detail::layer_cache_offset(backend, 0, position, kv_dim);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
        backend.key_cache.data() + cache_offset);
    emel::generator::detail::store_fp16_rounded_cache(
        std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
        backend.value_cache.data() + cache_offset);

    const std::string layer_prefix =
        "generation_debug.target.gen" + std::to_string(target_generated_index) + ".layer0";
    const auto dump_target_state = [&](const char * suffix,
                                       const auto emel_values,
                                       std::span<const float> reference_values) {
      std::fprintf(stdout,
                   "%s.%s: max_abs=%g\n",
                   layer_prefix.c_str(),
                   suffix,
                   max_abs_diff(emel_values, reference_values));
    };

    dump_target_state("q",
                      backend.q,
                      reference_last_token_row(find_reference_tensor(graph_capture, "Qcur-0"),
                                               backend.q.size()));
    dump_target_state("k",
                      backend.k,
                      reference_last_token_row(find_reference_tensor(graph_capture, "Kcur-0"),
                                               backend.k.size()));
    dump_target_state("v",
                      backend.v,
                      reference_last_token_row(find_reference_tensor(graph_capture, "Vcur-0"),
                                               backend.v.size()));
    {
      const std::span<const float> reference_q = reference_last_token_row(
          find_reference_tensor(graph_capture, "Qcur-0"), backend.q_attn.size());
      if (reference_q.size() == backend.q_attn.size()) {
        std::vector<float> reference_q_attn(reference_q.begin(), reference_q.end());
        emel::generator::detail::store_fp16_rounded_cache(reference_q, reference_q_attn.data());
        dump_target_state("q_attn", backend.q_attn, reference_q_attn);
      } else {
        std::fprintf(stdout, "%s.q_attn: unavailable\n", layer_prefix.c_str());
      }
    }
    if (reference_key_cache_rows.size() ==
        static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)) {
      dump_target_state("key_cache",
                        std::span<const uint16_t>(backend.key_cache.data() + cache_offset,
                                               static_cast<size_t>(kv_dim)),
                        std::span<const float>(reference_key_cache_rows)
                            .subspan(static_cast<size_t>(position) * static_cast<size_t>(kv_dim),
                                     static_cast<size_t>(kv_dim)));
    } else {
      std::fprintf(stdout, "%s.key_cache: unavailable\n", layer_prefix.c_str());
    }
    if (reference_value_cache_rows.size() ==
        static_cast<size_t>(position + 1) * static_cast<size_t>(kv_dim)) {
      dump_target_state("value_cache_state",
                        std::span<const uint16_t>(backend.value_cache.data() + cache_offset,
                                               static_cast<size_t>(kv_dim)),
                        std::span<const float>(reference_value_cache_rows)
                            .subspan(static_cast<size_t>(position) * static_cast<size_t>(kv_dim),
                                     static_cast<size_t>(kv_dim)));
    } else {
      std::fprintf(stdout, "%s.value_cache_state: unavailable\n", layer_prefix.c_str());
    }

    (void) compute_attention_with_softmax_debug(
        backend,
        graph_capture,
        0,
        position + 1,
        layer_prefix,
        reference_value_cache_rows);
  };

  dump_target_generated_index(0);
  if (token_mismatch_index > 5) {
    dump_target_generated_index(5);
  }
  dump_target_generated_index(token_mismatch_index - 1);
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

void dump_generation_live_backend_prefix_debug(const generation_load_state & state,
                                               const emel::paritychecker::parity_options & opts,
                                               const generation_result & emel_result,
                                               const generation_result & reference_result) {
  const int32_t token_mismatch_index = first_token_mismatch_index(emel_result, reference_result);
  if (state.model_data == nullptr || token_mismatch_index <= 0) {
    return;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(state.backend, opts, prompt_tokens)) {
    std::fprintf(stdout, "generation_debug.live: tokenize failed\n");
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

  std::vector<llama_token> prefix_tokens_llama;
  prefix_tokens_llama.reserve(prefix_tokens.size());
  for (const int32_t token : prefix_tokens) {
    prefix_tokens_llama.push_back(static_cast<llama_token>(token));
  }

  reference_graph_capture graph_capture = {};
  if (!capture_reference_graph_for_tokens(state, prefix_tokens_llama, graph_capture)) {
    std::fprintf(stdout, "generation_debug.live: reference graph capture failed\n");
    return;
  }

  emel::generator::detail::native_backend dispatch_backend = {};
  emel::generator::detail::native_backend shared_backend = {};
  if (emel::generator::detail::prepare(dispatch_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::generator::detail::prepare(shared_backend, *state.model_data) !=
          emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stdout, "generation_debug.live: backend prepare failed\n");
    return;
  }

  shared_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  shared_backend.kernel.set_kind(shared_backend.kernel_kind);

  const size_t prompt_count = prompt_tokens.size();
  const int32_t timeline_window = 12;
  const int32_t timeline_start_generated =
      std::max<int32_t>(0, token_mismatch_index - timeline_window);

  for (size_t token_index = 0; token_index < prefix_tokens.size(); ++token_index) {
    const int32_t token_id = prefix_tokens[token_index];
    const int32_t position = static_cast<int32_t>(token_index);
    if (!emel::generator::detail::copy_tensor_row(
            *dispatch_backend.token_embedding.tensor, token_id, dispatch_backend.hidden) ||
        !emel::generator::detail::copy_tensor_row(
            *shared_backend.token_embedding.tensor, token_id, shared_backend.hidden)) {
      std::fprintf(stdout, "generation_debug.live: token embedding replay failed\n");
      return;
    }

    for (int32_t layer = 0; layer < dispatch_backend.n_layer; ++layer) {
      const std::vector<float> dispatch_hidden_in = dispatch_backend.hidden;
      const std::vector<float> shared_hidden_in = shared_backend.hidden;
      if (!emel::generator::detail::run_layer_flash(dispatch_backend, layer, position) ||
          !emel::generator::detail::run_layer_flash(shared_backend, layer, position)) {
        std::fprintf(stdout, "generation_debug.live: layer replay failed\n");
        return;
      }

      const bool is_prompt_token = token_index < prompt_count;
      const int32_t generated_index = static_cast<int32_t>(token_index - prompt_count);
      if (!is_prompt_token && generated_index < timeline_start_generated) {
        continue;
      }

      const std::string token_prefix = is_prompt_token
                                           ? "generation_debug.live.prompt" +
                                                 std::to_string(token_index)
                                           : "generation_debug.live.gen" +
                                                 std::to_string(generated_index);
      const std::string layer_prefix =
          token_prefix + ".layer" + std::to_string(layer);
      const std::string layer_suffix = std::to_string(layer);
      auto & block = dispatch_backend.blocks[static_cast<size_t>(layer)];
      const int32_t kv_dim = dispatch_backend.n_head_kv * dispatch_backend.head_dim_kv;
      const size_t cache_offset = emel::generator::detail::layer_cache_offset(
          dispatch_backend, layer, position, kv_dim);
      const std::span<const uint16_t> dispatch_key_cache{
          dispatch_backend.key_cache.data() + cache_offset,
          static_cast<size_t>(kv_dim),
      };
      const std::span<const uint16_t> shared_key_cache{
          shared_backend.key_cache.data() + cache_offset,
          static_cast<size_t>(kv_dim),
      };
      const std::span<const uint16_t> dispatch_value_cache{
          dispatch_backend.value_cache.data() + cache_offset,
          static_cast<size_t>(kv_dim),
      };
      const std::span<const uint16_t> shared_value_cache{
          shared_backend.value_cache.data() + cache_offset,
          static_cast<size_t>(kv_dim),
      };

      std::fprintf(stdout,
                   "%s.q: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.q, shared_backend.q));
      std::fprintf(stdout,
                   "%s.k: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.k, shared_backend.k));
      std::fprintf(stdout,
                   "%s.v: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.v, shared_backend.v));
      std::fprintf(stdout,
                   "%s.key_cache: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_key_cache, shared_key_cache));
      std::fprintf(stdout,
                   "%s.value_cache: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_value_cache, shared_value_cache));
      std::fprintf(stdout,
                   "%s.kqv_out: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.attn_ctx, shared_backend.attn_ctx));
      std::fprintf(stdout,
                   "%s.ffn_norm: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.norm, shared_backend.norm));
      std::fprintf(stdout,
                   "%s.ffn_gate: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.gate, shared_backend.gate));
      std::fprintf(stdout,
                   "%s.ffn_up: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.up, shared_backend.up));
      std::fprintf(stdout,
                   "%s.ffn_out: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.projected, shared_backend.projected));
      std::fprintf(stdout,
                   "%s.l_out: max_abs=%g\n",
                   layer_prefix.c_str(),
                   max_abs_diff(dispatch_backend.hidden, shared_backend.hidden));
      {
        std::vector<float> dispatch_attn_norm_reference(static_cast<size_t>(dispatch_backend.norm.size()));
        if (emel::generator::detail::rms_norm(dispatch_hidden_in,
                                              block.attention_norm,
                                              dispatch_backend.rms_epsilon,
                                              dispatch_attn_norm_reference)) {
          const std::span<const float> reference_attn_norm = reference_token_row(
              find_reference_tensor(graph_capture, ("attn_norm-" + layer_suffix).c_str()),
              dispatch_attn_norm_reference.size(),
              static_cast<int32_t>(token_index));
          if (!reference_attn_norm.empty()) {
            std::fprintf(stdout,
                         "%s.dispatch_reference_attn_norm: max_abs=%g\n",
                         layer_prefix.c_str(),
                         max_abs_diff(dispatch_attn_norm_reference, reference_attn_norm));
          }
          dump_matrix_compare((layer_prefix + ".dispatch_attn_q_matmul_ggml").c_str(),
                              dispatch_backend,
                              block.attention_q,
                              dispatch_attn_norm_reference);
          dump_matrix_compare((layer_prefix + ".dispatch_attn_k_matmul_ggml").c_str(),
                              dispatch_backend,
                              block.attention_k,
                              dispatch_attn_norm_reference);
          dump_matrix_compare((layer_prefix + ".dispatch_attn_v_matmul_ggml").c_str(),
                              dispatch_backend,
                              block.attention_v,
                              dispatch_attn_norm_reference);
        }
      }
      if (const reference_tensor_capture * v_capture =
              find_reference_capture(graph_capture, ("Vcur-" + layer_suffix).c_str())) {
        const std::vector<float> reference_v =
            reference_token_tensor_values(*v_capture, static_cast<int32_t>(token_index));
        std::fprintf(stdout,
                     "%s.dispatch_reference_v: max_abs=%g\n",
                     layer_prefix.c_str(),
                     reference_v.empty() ? -1.0f : max_abs_diff(dispatch_backend.v, reference_v));
      }
      if (const reference_tensor_capture * q_capture =
              find_reference_capture(graph_capture, ("Qcur-" + layer_suffix).c_str())) {
        const std::vector<float> reference_q =
            reference_token_tensor_values(*q_capture, static_cast<int32_t>(token_index));
        std::fprintf(stdout,
                     "%s.dispatch_reference_q: max_abs=%g\n",
                     layer_prefix.c_str(),
                     reference_q.empty() ? -1.0f : max_abs_diff(dispatch_backend.q, reference_q));
      }
      if (const reference_tensor_capture * k_capture =
              find_reference_capture(graph_capture, ("Kcur-" + layer_suffix).c_str())) {
        const std::vector<float> reference_k =
            reference_token_tensor_values(*k_capture, static_cast<int32_t>(token_index));
        std::fprintf(stdout,
                     "%s.dispatch_reference_k: max_abs=%g\n",
                     layer_prefix.c_str(),
                     reference_k.empty() ? -1.0f : max_abs_diff(dispatch_backend.k, reference_k));
      }
      {
        const std::span<const float> reference_kqv_out = reference_token_row(
            find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()),
            dispatch_backend.attn_ctx.size(),
            static_cast<int32_t>(token_index));
        if (!reference_kqv_out.empty()) {
          std::fprintf(stdout,
                       "%s.dispatch_reference_kqv_out: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(dispatch_backend.attn_ctx, reference_kqv_out));
        }
      }
      {
        const std::span<const float> reference_l_out = reference_token_row(
            find_reference_tensor(graph_capture, ("l_out-" + layer_suffix).c_str()),
            dispatch_backend.hidden.size(),
            static_cast<int32_t>(token_index));
        if (!reference_l_out.empty()) {
          std::fprintf(stdout,
                       "%s.dispatch_reference_l_out: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(dispatch_backend.hidden, reference_l_out));
        }
      }

      std::vector<float> shared_attn_norm(static_cast<size_t>(shared_backend.norm.size()));
      std::vector<float> dispatch_attn_q_same(static_cast<size_t>(dispatch_backend.q.size()));
      std::vector<float> shared_attn_q_same(static_cast<size_t>(shared_backend.q.size()));
      std::vector<float> dispatch_attn_k_same(static_cast<size_t>(dispatch_backend.k.size()));
      std::vector<float> shared_attn_k_same(static_cast<size_t>(shared_backend.k.size()));
      std::vector<float> dispatch_attn_v_same(static_cast<size_t>(dispatch_backend.v.size()));
      std::vector<float> shared_attn_v_same(static_cast<size_t>(shared_backend.v.size()));
      std::vector<float> reference_attn_q_same(static_cast<size_t>(shared_backend.q.size()));
      std::vector<float> reference_attn_k_same(static_cast<size_t>(shared_backend.k.size()));
      std::vector<float> reference_attn_v_same(static_cast<size_t>(shared_backend.v.size()));
      if (emel::generator::detail::rms_norm(
              shared_hidden_in,
              block.attention_norm,
              dispatch_backend.rms_epsilon,
              shared_attn_norm) &&
          emel::generator::detail::matmul_vector(dispatch_backend,
                                                 block.attention_q,
                                                 shared_attn_norm,
                                                 dispatch_attn_q_same) &&
          emel::generator::detail::matmul_vector(shared_backend,
                                                 block.attention_q,
                                                 shared_attn_norm,
                                                 shared_attn_q_same) &&
          emel::generator::detail::matmul_vector(dispatch_backend,
                                                 block.attention_k,
                                                 shared_attn_norm,
                                                 dispatch_attn_k_same) &&
          emel::generator::detail::matmul_vector(shared_backend,
                                                 block.attention_k,
                                                 shared_attn_norm,
                                                 shared_attn_k_same) &&
          emel::generator::detail::matmul_vector(dispatch_backend,
                                                 block.attention_v,
                                                 shared_attn_norm,
                                                 dispatch_attn_v_same) &&
          emel::generator::detail::matmul_vector(shared_backend,
                                                 block.attention_v,
                                                 shared_attn_norm,
                                                 shared_attn_v_same)) {
        std::fprintf(stdout,
                     "%s.same_input_q: max_abs=%g\n",
                     layer_prefix.c_str(),
                     max_abs_diff(dispatch_attn_q_same, shared_attn_q_same));
        std::fprintf(stdout,
                     "%s.same_input_k: max_abs=%g\n",
                     layer_prefix.c_str(),
                     max_abs_diff(dispatch_attn_k_same, shared_attn_k_same));
        std::fprintf(stdout,
                     "%s.same_input_v: max_abs=%g\n",
                     layer_prefix.c_str(),
                     max_abs_diff(dispatch_attn_v_same, shared_attn_v_same));
        if (matmul_vector_reference_q8(block.attention_q,
                                       shared_attn_norm,
                                       reference_attn_q_same)) {
          std::fprintf(stdout,
                       "%s.same_input_q_dispatch_reference_q8: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(dispatch_attn_q_same, reference_attn_q_same));
          std::fprintf(stdout,
                       "%s.same_input_q_shared_reference_q8: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(shared_attn_q_same, reference_attn_q_same));
        }
        if (matmul_vector_reference_q8(block.attention_k,
                                       shared_attn_norm,
                                       reference_attn_k_same)) {
          std::fprintf(stdout,
                       "%s.same_input_k_dispatch_reference_q8: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(dispatch_attn_k_same, reference_attn_k_same));
          std::fprintf(stdout,
                       "%s.same_input_k_shared_reference_q8: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(shared_attn_k_same, reference_attn_k_same));
        }
        if (matmul_vector_reference_q8(block.attention_v,
                                       shared_attn_norm,
                                       reference_attn_v_same)) {
          std::fprintf(stdout,
                       "%s.same_input_v_dispatch_reference_q8: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(dispatch_attn_v_same, reference_attn_v_same));
          std::fprintf(stdout,
                       "%s.same_input_v_shared_reference_q8: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(shared_attn_v_same, reference_attn_v_same));
        }
        {
          const auto compare_flash_request =
              [&](const char * label, emel::generator::detail::native_backend & backend) {
                auto neon_request =
                    emel::generator::detail::make_flash_attn_request(backend, layer, position);
                std::vector<float> neon_dst(backend.attn_ctx.size(), 0.0f);
                std::vector<float> shared_dst(backend.attn_ctx.size(), 0.0f);
                neon_request.dst = emel::generator::detail::make_dst_view_3d(
                    neon_dst.data(),
                    neon_request.dst.ne[0],
                    neon_request.dst.ne[1],
                    neon_request.dst.ne[2]);
                auto shared_request = neon_request;
                shared_request.dst = emel::generator::detail::make_dst_view_3d(
                    shared_dst.data(),
                    shared_request.dst.ne[0],
                    shared_request.dst.ne[1],
                    shared_request.dst.ne[2]);
                emel::kernel::detail::flash_attn_workspace neon_workspace = {};
                emel::kernel::detail::flash_attn_workspace shared_workspace = {};
                if (emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
                        neon_request, true, neon_workspace) &&
                    emel::kernel::detail::run_flash_attn_ext_with_workspace(
                        shared_request, shared_workspace)) {
                  std::fprintf(stdout,
                               "%s.%s: max_abs=%g\n",
                               layer_prefix.c_str(),
                               label,
                               max_abs_diff(neon_dst, shared_dst));
                }
              };
          compare_flash_request("same_request_flash_dispatch_state", dispatch_backend);
          compare_flash_request("same_request_flash_shared_state", shared_backend);
        }
        {
          auto prepare_attention_scratch =
              [&](emel::generator::detail::native_backend & scratch) {
                return emel::generator::detail::prepare(scratch, *state.model_data) ==
                           emel::error::cast(emel::model::loader::error::none) &&
                    ((scratch.kernel_kind = shared_backend.kernel_kind), true) &&
                    ((scratch.kernel.set_kind(scratch.kernel_kind)), true) &&
                    ((scratch.key_cache = shared_backend.key_cache), true) &&
                    ((scratch.value_cache = shared_backend.value_cache), true) &&
                    ((scratch.q_attn = shared_backend.q_attn), true);
              };
          const std::span<const float> reference_kqv_out = reference_token_row(
              find_reference_tensor(graph_capture, ("kqv_out-" + layer_suffix).c_str()),
              dispatch_backend.attn_ctx.size(),
              static_cast<int32_t>(token_index));

          emel::generator::detail::native_backend prod_backend = {};
          if (prepare_attention_scratch(prod_backend) &&
              compute_attention_with_emel_prod_style(
                  prod_backend,
                  layer,
                  position + 1,
                  std::span<const float>(prod_backend.q_attn.data(),
                                         static_cast<size_t>(prod_backend.n_embd)))) {
            std::fprintf(stdout,
                         "%s.same_input_flash_vs_emel_prod_style: max_abs=%g\n",
                         layer_prefix.c_str(),
                         max_abs_diff(dispatch_backend.attn_ctx, prod_backend.attn_ctx));
            if (!reference_kqv_out.empty()) {
              std::fprintf(stdout,
                           "%s.same_input_emel_prod_style_reference_kqv_out: max_abs=%g\n",
                           layer_prefix.c_str(),
                           max_abs_diff(prod_backend.attn_ctx, reference_kqv_out));
            }
          }

          emel::generator::detail::native_backend exact_masked_backend = {};
          if (prepare_attention_scratch(exact_masked_backend) &&
              compute_attention_with_ggml_nonflash_exact_masked(
                  exact_masked_backend,
                  layer,
                  position + 1,
                  std::span<const float>(exact_masked_backend.q_attn.data(),
                                         static_cast<size_t>(exact_masked_backend.n_embd)))) {
            std::fprintf(stdout,
                         "%s.same_input_flash_vs_exact_masked: max_abs=%g\n",
                         layer_prefix.c_str(),
                         max_abs_diff(dispatch_backend.attn_ctx, exact_masked_backend.attn_ctx));
            if (!reference_kqv_out.empty()) {
              std::fprintf(stdout,
                           "%s.same_input_exact_masked_reference_kqv_out: max_abs=%g\n",
                           layer_prefix.c_str(),
                           max_abs_diff(exact_masked_backend.attn_ctx, reference_kqv_out));
            }
          }
        }
      }

      std::vector<float> dispatch_ffn_gate_same(static_cast<size_t>(dispatch_backend.gate.size()));
      std::vector<float> shared_ffn_gate_same(static_cast<size_t>(shared_backend.gate.size()));
      std::vector<float> dispatch_ffn_up_same(static_cast<size_t>(dispatch_backend.up.size()));
      std::vector<float> shared_ffn_up_same(static_cast<size_t>(shared_backend.up.size()));
      std::vector<float> dispatch_ffn_out_same(static_cast<size_t>(dispatch_backend.projected.size()));
      std::vector<float> shared_ffn_out_same(static_cast<size_t>(shared_backend.projected.size()));
      std::vector<float> shared_ffn_hidden(static_cast<size_t>(shared_backend.ffn_hidden.size()));
      if (emel::generator::detail::matmul_vector(dispatch_backend,
                                                 block.feed_forward_gate,
                                                 shared_backend.norm,
                                                 dispatch_ffn_gate_same) &&
          emel::generator::detail::matmul_vector(shared_backend,
                                                 block.feed_forward_gate,
                                                 shared_backend.norm,
                                                 shared_ffn_gate_same) &&
          emel::generator::detail::matmul_vector(dispatch_backend,
                                                 block.feed_forward_up,
                                                 shared_backend.norm,
                                                 dispatch_ffn_up_same) &&
          emel::generator::detail::matmul_vector(shared_backend,
                                                 block.feed_forward_up,
                                                 shared_backend.norm,
                                                 shared_ffn_up_same)) {
        for (size_t idx = 0; idx < shared_ffn_hidden.size(); ++idx) {
          shared_ffn_hidden[idx] = emel::generator::detail::silu(shared_ffn_gate_same[idx]) *
              shared_ffn_up_same[idx];
        }
        std::fprintf(stdout,
                     "%s.same_input_ffn_gate: max_abs=%g\n",
                     layer_prefix.c_str(),
                     max_abs_diff(dispatch_ffn_gate_same, shared_ffn_gate_same));
        std::fprintf(stdout,
                     "%s.same_input_ffn_up: max_abs=%g\n",
                     layer_prefix.c_str(),
                     max_abs_diff(dispatch_ffn_up_same, shared_ffn_up_same));
        if (emel::generator::detail::matmul_vector(dispatch_backend,
                                                   block.feed_forward_down,
                                                   shared_ffn_hidden,
                                                   dispatch_ffn_out_same) &&
            emel::generator::detail::matmul_vector(shared_backend,
                                                   block.feed_forward_down,
                                                   shared_ffn_hidden,
                                                   shared_ffn_out_same)) {
          std::fprintf(stdout,
                       "%s.same_input_ffn_out: max_abs=%g\n",
                       layer_prefix.c_str(),
                       max_abs_diff(dispatch_ffn_out_same, shared_ffn_out_same));
        }
      }
    }

    dispatch_backend.kv_cache_tokens = position + 1;
    shared_backend.kv_cache_tokens = position + 1;
  }
}

void dump_reference_decode_seam(const generation_load_state & state) {
  const emel::kernel::kernel_kind kernel_kind = state.generator->generation_kernel_kind();
  const uint64_t native_q8_0_dispatch_calls =
      state.generator->generation_native_q8_0_dispatch_calls();
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
               " shared_q6_dispatch_calls=%" PRIu64
               " native_q8_0_dispatch_calls=%" PRIu64 "\n",
               optimized_q2_dispatch_calls,
               shared_q2_dispatch_calls,
               optimized_q3_dispatch_calls,
               shared_q3_dispatch_calls,
               optimized_q6_dispatch_calls,
               shared_q6_dispatch_calls,
               native_q8_0_dispatch_calls);
  const auto runtime_contract = runtime_quantized_contract_summary(state);
  std::fprintf(stdout,
               "quantized_runtime_contract: native_quantized=%u "
               "approved_dense_f32_by_contract=%u disallowed_fallback=%u explicit_no_claim=%u\n",
               runtime_contract.native_quantized,
               runtime_contract.approved_dense_f32_by_contract,
               runtime_contract.disallowed_fallback,
               runtime_contract.explicit_no_claim);

  emel::model::llama::detail::execution_view execution{};
  if (emel::model::llama::detail::build_execution_view(*state.model_data, execution) ==
      emel::error::cast(emel::model::loader::error::none)) {
    const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
    const auto audit_contract = build_quantized_contract_summary(audit);

    std::fprintf(stdout,
                 "quantized_stage_inventory: native_quantized=%u "
                 "approved_dense_f32_by_contract=%u disallowed_fallback=%u explicit_no_claim=%u\n",
                 audit_contract.native_quantized,
                 audit_contract.approved_dense_f32_by_contract,
                 audit_contract.disallowed_fallback,
                 audit_contract.explicit_no_claim);
    for (const auto & stage : audit.stages) {
      const auto stage_name = emel::model::llama::detail::quantized_stage_family_name(stage.family);
      const auto tensor_name = emel::model::llama::detail::tensor_type_name(stage.tensor_type);
      const auto contract_name =
          emel::model::llama::detail::quantized_contract_kind_name(stage.contract);
      const uint32_t supported =
          stage.contract == emel::model::llama::detail::quantized_contract_kind::explicit_no_claim
              ? 0u
              : 1u;
      std::fprintf(stdout,
                   "quantized_stage_audit: stage=%.*s tensor_type=%.*s contract=%.*s "
                   "supported=%u consistent_across_layers=%u\n",
                   static_cast<int>(stage_name.size()),
                   stage_name.data(),
                   static_cast<int>(tensor_name.size()),
                   tensor_name.data(),
                   static_cast<int>(contract_name.size()),
                   contract_name.data(),
                   supported,
                   stage.consistent_across_layers ? 1u : 0u);
    }
  }
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
    dump_generation_live_backend_prefix_debug(state, opts, *emel_result, *reference_result);
    dump_generation_gen0_attention_debug(state, opts, *emel_result, *reference_result);
    dump_generation_target_attention_debug(state, opts, *emel_result, *reference_result);
    dump_generation_prefix_state_debug(state, opts, *emel_result, *reference_result);
    dump_generation_reference_q_stage_debug(state, opts, *emel_result, *reference_result);
    dump_scalar_attention_debug(state, opts, *emel_result, *reference_result);
    return;
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
  const bool is_llama = architecture == "llama";
  const bool is_qwen3 = architecture == "qwen3";
  if (!is_llama && !is_qwen3) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

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

  if (is_llama) {
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
  }

  if (is_qwen3) {
    int32_t qwen3_key_length = 0;
    int32_t qwen3_value_length = 0;
    if (!assign_i32("qwen3.context_length", model_data.params.n_ctx) ||
        !assign_i32("qwen3.embedding_length", model_data.params.n_embd) ||
        !assign_i32("qwen3.feed_forward_length", model_data.params.n_ff) ||
        !assign_i32("qwen3.attention.head_count", model_data.params.n_head) ||
        !assign_i32("qwen3.attention.head_count_kv", model_data.params.n_head_kv) ||
        !assign_i32("qwen3.attention.key_length", qwen3_key_length) ||
        !assign_i32("qwen3.attention.value_length", qwen3_value_length) ||
        !assign_i32("qwen3.block_count", model_data.params.n_layer) ||
        !assign_f32(
            "qwen3.attention.layer_norm_rms_epsilon",
            model_data.params.attention_layer_norm_rms_epsilon) ||
        !assign_f32("qwen3.rope.freq_base", model_data.params.rope_freq_base)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    model_data.params.attention_key_length = qwen3_key_length;
    model_data.params.attention_value_length = qwen3_value_length;
    if (model_data.params.n_embd_out == 0) {
      model_data.params.n_embd_out = model_data.params.n_embd;
    }
    if (model_data.params.n_rot == 0) {
      model_data.params.n_rot = qwen3_key_length;
    }
    if (qwen3_key_length <= 0 || qwen3_value_length <= 0) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
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

void resolve_generation_formatter_binding(generation_load_state & state) {
  std::string_view primary_template = {};
  const auto * entry = find_kv_entry(state, "tokenizer.chat_template");
  if (entry != nullptr && !decode_string_value(state, *entry, primary_template)) {
    state.formatter_binding = emel::tools::generation_formatter_contract::formatter_binding{
      .formatter_ctx = nullptr,
      .format_prompt = emel::text::formatter::format_raw,
      .support = emel::tools::generation_formatter_contract::support_kind::unsupported_template,
      .contract = emel::tools::generation_formatter_contract::k_unsupported_template_contract,
    };
    return;
  }

  uint32_t named_template_count = 0u;
  for (const auto & candidate : state.kv_entries) {
    const std::string_view key = kv_key_view(state, candidate);
    if (key.starts_with("tokenizer.chat_template.") &&
        key != "tokenizer.chat_template") {
      named_template_count += 1u;
    }
  }

  state.formatter_binding =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          primary_template,
          named_template_count);
}

void print_generation_formatter_contract(FILE * stream,
                                         const generation_load_state & state) {
  if (stream == nullptr || state.formatter_binding.contract.empty()) {
    return;
  }
  std::fprintf(stream,
               "formatter_contract=%.*s\n",
               static_cast<int>(state.formatter_binding.contract.size()),
               state.formatter_binding.contract.data());
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
  const std::string_view architecture = emel::model::architecture_name_view(req.model_data);
  return (architecture == "llama" || architecture == "qwen3")
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

bool compute_attention_with_ggml_nonflash_exact_scores_prod_value(
    emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position_limit,
    std::span<const float> q_vector) {
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_tokens = backend.n_ctx;
  const int32_t head_count = backend.n_head;
  const int32_t kv_head_count = backend.n_head_kv;
  const int32_t kv_dim = kv_head_count * backend.head_dim_kv;
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_vector.size() != q_size || position_limit <= 0 || position_limit > kv_tokens) {
    return false;
  }

  ggml_case_context c{};
  ggml_tensor * q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, head_dim, 1, head_count);
  ggml_tensor * k = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, head_dim, kv_tokens, kv_head_count);
  ggml_tensor * mask = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, kv_tokens, 1);
  if (q == nullptr || k == nullptr || mask == nullptr) {
    return false;
  }

  std::memcpy(ggml_get_data(q), q_vector.data(), q_size * sizeof(float));

  const size_t layer_offset =
      emel::generator::detail::layer_cache_offset(backend, layer_index, 0, kv_dim);
  std::vector<ggml_fp16_t> k_f16(kv_size);
  for (int32_t head = 0; head < kv_head_count; ++head) {
    for (int32_t token = 0; token < kv_tokens; ++token) {
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const size_t src_index =
            layer_offset + static_cast<size_t>(((token * kv_head_count) + head) * head_dim + dim);
        const size_t dst_index =
            static_cast<size_t>(((head * kv_tokens) + token) * head_dim + dim);
        k_f16[dst_index] = ggml_fp32_to_fp16(fp16_storage_to_fp32(backend.key_cache[src_index]));
      }
    }
  }
  std::memcpy(ggml_get_data(k), k_f16.data(), k_f16.size() * sizeof(ggml_fp16_t));

  auto * mask_data = ggml_get_data_f32(mask);
  for (int32_t token = 0; token < kv_tokens; ++token) {
    mask_data[token] = token < position_limit ? 0.0f : -INFINITY;
  }

  ggml_tensor * kq = ggml_mul_mat(c.ctx, k, q);
  if (kq == nullptr) {
    return false;
  }
  ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
  kq = ggml_soft_max_ext(
      c.ctx, kq, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
  if (kq == nullptr || !compute_graph(c, kq)) {
    return false;
  }

  const float * weights = ggml_get_data_f32(kq);
  if (weights == nullptr) {
    return false;
  }

  reference_tensor_capture weight_capture{};
  weight_capture.shape = {
      kq->ne[0],
      kq->ne[1],
      kq->ne[2],
      kq->ne[3],
  };
  weight_capture.values.assign(weights, weights + ggml_nelements(kq));

  std::fill(backend.attn_ctx.begin(), backend.attn_ctx.end(), 0.0f);
  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head * head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head * backend.head_dim_kv);
    const std::span<const float> head_weights =
        reference_softmax_query_head_slice(weight_capture, head, 0);
    if (head_weights.size() != static_cast<size_t>(kv_tokens)) {
      return false;
    }
    for (int32_t token = 0; token < position_limit; ++token) {
      const float weight = head_weights[static_cast<size_t>(token)];
      const size_t cache_offset =
          emel::generator::detail::layer_cache_offset(backend, layer_index, token, kv_dim) +
          kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * fp16_storage_to_fp32(backend.value_cache[cache_offset + static_cast<size_t>(dim)]);
      }
    }
  }

  return true;
}

bool run_ggml_nonflash_attn_case(std::span<const float> q_data,
                                 std::span<const float> k_data,
                                 std::span<const float> v_data,
                                 const int64_t head_dim,
                                 const int64_t kv_tokens,
                                 const int64_t active_kv_tokens,
                                 const int64_t head_count,
                                 const int64_t kv_head_count,
                                 const float scale,
                                 std::vector<float> & out) {
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_data.size() != q_size || k_data.size() != kv_size || v_data.size() != kv_size ||
      head_dim <= 0 || kv_tokens <= 0 || active_kv_tokens <= 0 || active_kv_tokens > kv_tokens ||
      head_count <= 0 || kv_head_count <= 0) {
    return false;
  }

  ggml_case_context c{};
  ggml_tensor * q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, head_dim, 1, head_count);
  ggml_tensor * k = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, head_dim, kv_tokens, kv_head_count);
  ggml_tensor * v = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, kv_tokens, head_dim, kv_head_count);
  ggml_tensor * mask = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, kv_tokens, 1);
  if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
    return false;
  }

  std::memcpy(ggml_get_data(q), q_data.data(), q_size * sizeof(float));

  std::vector<ggml_fp16_t> k_f16(kv_size);
  std::vector<ggml_fp16_t> v_f16(kv_size);
  for (int64_t head = 0; head < kv_head_count; ++head) {
    for (int64_t token = 0; token < kv_tokens; ++token) {
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        const size_t src_index = static_cast<size_t>(
            ((token * kv_head_count) + head) * head_dim + dim);
        const size_t k_dst_index = static_cast<size_t>(
            ((head * kv_tokens) + token) * head_dim + dim);
        const size_t v_dst_index = static_cast<size_t>(
            ((head * head_dim) + dim) * kv_tokens + token);
        k_f16[k_dst_index] = ggml_fp32_to_fp16(k_data[src_index]);
        v_f16[v_dst_index] = ggml_fp32_to_fp16(v_data[src_index]);
      }
    }
  }

  std::memcpy(ggml_get_data(k), k_f16.data(), k_f16.size() * sizeof(ggml_fp16_t));
  std::memcpy(ggml_get_data(v), v_f16.data(), v_f16.size() * sizeof(ggml_fp16_t));
  auto * mask_data = ggml_get_data_f32(mask);
  for (int64_t token = 0; token < kv_tokens; ++token) {
    mask_data[token] = token < active_kv_tokens ? 0.0f : -INFINITY;
  }

  ggml_tensor * kq = ggml_mul_mat(c.ctx, k, q);
  if (kq == nullptr) {
    return false;
  }
  ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
  kq = ggml_soft_max_ext(c.ctx, kq, mask, scale, 0.0f);
  if (kq == nullptr) {
    return false;
  }

  ggml_tensor * kqv = ggml_mul_mat(c.ctx, v, kq);
  if (kqv == nullptr || !compute_graph(c, kqv)) {
    return false;
  }

  const float * out_data = ggml_get_data_f32(kqv);
  out.assign(out_data, out_data + q_size);
  return true;
}

bool run_ggml_nonflash_attn_case(std::span<const float> q_data,
                                 std::span<const uint16_t> k_data,
                                 std::span<const uint16_t> v_data,
                                 const int64_t head_dim,
                                 const int64_t kv_tokens,
                                 const int64_t active_kv_tokens,
                                 const int64_t head_count,
                                 const int64_t kv_head_count,
                                 const float scale,
                                 std::vector<float> & out) {
  const std::vector<float> k_decoded = decode_fp16_storage(k_data);
  const std::vector<float> v_decoded = decode_fp16_storage(v_data);
  return run_ggml_nonflash_attn_case(q_data,
                                     k_decoded,
                                     v_decoded,
                                     head_dim,
                                     kv_tokens,
                                     active_kv_tokens,
                                     head_count,
                                     kv_head_count,
                                     scale,
                                     out);
}

bool run_emel_nonflash_f16_ggml_softmax_case(std::span<const float> q_data,
                                             std::span<const float> k_data,
                                             std::span<const float> v_data,
                                             const int64_t head_dim,
                                             const int64_t kv_tokens,
                                             const int64_t active_kv_tokens,
                                             const int64_t head_count,
                                             const int64_t kv_head_count,
                                             const float scale,
                                             std::vector<float> & out) {
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_data.size() != q_size || k_data.size() != kv_size || v_data.size() != kv_size ||
      head_dim <= 0 || kv_tokens <= 0 || active_kv_tokens <= 0 || active_kv_tokens > kv_tokens ||
      head_count <= 0 || kv_head_count <= 0 || head_count % kv_head_count != 0) {
    return false;
  }

  out.assign(q_size, 0.0f);
  std::vector<float> scores(static_cast<size_t>(kv_tokens), -INFINITY);
  std::vector<float> probs(static_cast<size_t>(kv_tokens), 0.0f);
  std::vector<ggml_fp16_t> q_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> k_f16(static_cast<size_t>(head_dim));
  std::vector<ggml_fp16_t> value_f16(static_cast<size_t>(kv_tokens));
  std::vector<ggml_fp16_t> weight_f16(static_cast<size_t>(kv_tokens));
  const int64_t n_rep = head_count / kv_head_count;

  for (int64_t head = 0; head < head_count; ++head) {
    const int64_t kv_head = head / n_rep;
    const size_t q_offset = static_cast<size_t>(head * head_dim);

    for (int64_t dim = 0; dim < head_dim; ++dim) {
      q_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(q_data[q_offset + static_cast<size_t>(dim)]);
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const size_t kv_offset = static_cast<size_t>(((token * kv_head_count) + kv_head) * head_dim);
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        k_f16[static_cast<size_t>(dim)] = ggml_fp32_to_fp16(k_data[kv_offset + static_cast<size_t>(dim)]);
      }

      float score = 0.0f;
      ggml_vec_dot_f16(static_cast<int>(head_dim),
                       &score,
                       0u,
                       k_f16.data(),
                       0u,
                       q_f16.data(),
                       0u,
                       1);
      score *= scale;
      scores[static_cast<size_t>(token)] = score;
      max_score = std::max(max_score, score);
    }

    const reference_ggml_float score_sum =
        ggml_vec_soft_max_f32(static_cast<int>(kv_tokens), probs.data(), scores.data(), max_score);
    const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int64_t token = 0; token < kv_tokens; ++token) {
      const float weight = probs[static_cast<size_t>(token)] * inv_score_sum;
      weight_f16[static_cast<size_t>(token)] = ggml_fp32_to_fp16(weight);
    }

    for (int64_t dim = 0; dim < head_dim; ++dim) {
      for (int64_t token = 0; token < kv_tokens; ++token) {
        const size_t kv_offset = static_cast<size_t>(((token * kv_head_count) + kv_head) * head_dim);
        value_f16[static_cast<size_t>(token)] = ggml_fp32_to_fp16(v_data[kv_offset + static_cast<size_t>(dim)]);
      }

      float dot = 0.0f;
      ggml_vec_dot_f16(static_cast<int>(kv_tokens),
                       &dot,
                       0u,
                       value_f16.data(),
                       0u,
                       weight_f16.data(),
                       0u,
                       1);
      out[q_offset + static_cast<size_t>(dim)] = dot;
    }
  }

  return true;
}

bool run_emel_nonflash_f16_ggml_softmax_case(std::span<const float> q_data,
                                             std::span<const uint16_t> k_data,
                                             std::span<const uint16_t> v_data,
                                             const int64_t head_dim,
                                             const int64_t kv_tokens,
                                             const int64_t active_kv_tokens,
                                             const int64_t head_count,
                                             const int64_t kv_head_count,
                                             const float scale,
                                             std::vector<float> & out) {
  const std::vector<float> k_decoded = decode_fp16_storage(k_data);
  const std::vector<float> v_decoded = decode_fp16_storage(v_data);
  return run_emel_nonflash_f16_ggml_softmax_case(q_data,
                                                 k_decoded,
                                                 v_decoded,
                                                 head_dim,
                                                 kv_tokens,
                                                 active_kv_tokens,
                                                 head_count,
                                                 kv_head_count,
                                                 scale,
                                                 out);
}

bool run_emel_prod_style_attn_case(std::span<const float> q_data,
                                   std::span<const float> k_data,
                                   std::span<const float> v_data,
                                   const int64_t head_dim,
                                   const int64_t kv_tokens,
                                   const int64_t active_kv_tokens,
                                   const int64_t head_count,
                                   const int64_t kv_head_count,
                                   const float scale,
                                   std::vector<float> & out) {
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_data.size() != q_size || k_data.size() != kv_size || v_data.size() != kv_size ||
      head_dim <= 0 || kv_tokens <= 0 || active_kv_tokens <= 0 || active_kv_tokens > kv_tokens ||
      head_count <= 0 || kv_head_count <= 0 || head_count % kv_head_count != 0) {
    return false;
  }

  out.assign(q_size, 0.0f);
  std::vector<float> scores(static_cast<size_t>(kv_tokens), -INFINITY);
  std::vector<float> rounded_probs(static_cast<size_t>(kv_tokens), 0.0f);
  std::vector<float> value_column(static_cast<size_t>(kv_tokens), 0.0f);
  const int64_t n_rep = head_count / kv_head_count;

  for (int64_t head = 0; head < head_count; ++head) {
    const int64_t kv_head = head / n_rep;
    const size_t q_offset = static_cast<size_t>(head * head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const size_t kv_offset = static_cast<size_t>(((token * kv_head_count) + kv_head) * head_dim);
      const float score = ::emel::kernel::detail::dot_product_ggml_f16_scores(
                              q_data.data() + static_cast<std::ptrdiff_t>(q_offset),
                              k_data.data() + static_cast<std::ptrdiff_t>(kv_offset),
                              static_cast<uint64_t>(head_dim)) *
          scale;
      scores[static_cast<size_t>(token)] = score;
      max_score = std::max(max_score, score);
    }

    double score_sum = 0.0;
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const float prob = std::exp(scores[static_cast<size_t>(token)] - max_score);
      rounded_probs[static_cast<size_t>(token)] = prob;
      score_sum += static_cast<double>(prob);
    }

    const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const float weight = rounded_probs[static_cast<size_t>(token)] * inv_score_sum;
      rounded_probs[static_cast<size_t>(token)] =
          ::emel::kernel::detail::round_fp16_weight(weight);
    }
    for (int64_t token = active_kv_tokens; token < kv_tokens; ++token) {
      rounded_probs[static_cast<size_t>(token)] = 0.0f;
    }

    for (int64_t dim = 0; dim < head_dim; ++dim) {
      for (int64_t token = 0; token < kv_tokens; ++token) {
        if (token < active_kv_tokens) {
          const size_t kv_offset = static_cast<size_t>(((token * kv_head_count) + kv_head) * head_dim);
          value_column[static_cast<size_t>(token)] = v_data[kv_offset + static_cast<size_t>(dim)];
        } else {
          value_column[static_cast<size_t>(token)] = 0.0f;
        }
      }

      out[q_offset + static_cast<size_t>(dim)] = ::emel::kernel::detail::dot_product_ggml_f16_scores(
          value_column.data(), rounded_probs.data(), static_cast<uint64_t>(kv_tokens));
    }
  }

  return true;
}

bool run_emel_prod_style_attn_case(std::span<const float> q_data,
                                   std::span<const uint16_t> k_data,
                                   std::span<const uint16_t> v_data,
                                   const int64_t head_dim,
                                   const int64_t kv_tokens,
                                   const int64_t active_kv_tokens,
                                   const int64_t head_count,
                                   const int64_t kv_head_count,
                                   const float scale,
                                   std::vector<float> & out) {
  const std::vector<float> k_decoded = decode_fp16_storage(k_data);
  const std::vector<float> v_decoded = decode_fp16_storage(v_data);
  return run_emel_prod_style_attn_case(q_data,
                                       k_decoded,
                                       v_decoded,
                                       head_dim,
                                       kv_tokens,
                                       active_kv_tokens,
                                       head_count,
                                       kv_head_count,
                                       scale,
                                       out);
}

bool run_emel_prod_style_float_value_attn_case(std::span<const float> q_data,
                                               std::span<const float> k_data,
                                               std::span<const float> v_data,
                                               const int64_t head_dim,
                                               const int64_t kv_tokens,
                                               const int64_t active_kv_tokens,
                                               const int64_t head_count,
                                               const int64_t kv_head_count,
                                               const float scale,
                                               std::vector<float> & out) {
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_data.size() != q_size || k_data.size() != kv_size || v_data.size() != kv_size ||
      head_dim <= 0 || kv_tokens <= 0 || active_kv_tokens <= 0 || active_kv_tokens > kv_tokens ||
      head_count <= 0 || kv_head_count <= 0 || head_count % kv_head_count != 0) {
    return false;
  }

  out.assign(q_size, 0.0f);
  std::vector<float> scores(static_cast<size_t>(kv_tokens), -INFINITY);
  std::vector<float> rounded_probs(static_cast<size_t>(kv_tokens), 0.0f);
  const int64_t n_rep = head_count / kv_head_count;

  for (int64_t head = 0; head < head_count; ++head) {
    const int64_t kv_head = head / n_rep;
    const size_t q_offset = static_cast<size_t>(head * head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const size_t kv_offset = static_cast<size_t>(((token * kv_head_count) + kv_head) * head_dim);
      const float score = ::emel::kernel::detail::dot_product_ggml_f16_scores(
                              q_data.data() + static_cast<std::ptrdiff_t>(q_offset),
                              k_data.data() + static_cast<std::ptrdiff_t>(kv_offset),
                              static_cast<uint64_t>(head_dim)) *
          scale;
      scores[static_cast<size_t>(token)] = score;
      max_score = std::max(max_score, score);
    }

    double score_sum = 0.0;
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const float prob = std::exp(scores[static_cast<size_t>(token)] - max_score);
      rounded_probs[static_cast<size_t>(token)] = prob;
      score_sum += static_cast<double>(prob);
    }

    const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (int64_t token = 0; token < active_kv_tokens; ++token) {
      const float weight = rounded_probs[static_cast<size_t>(token)] * inv_score_sum;
      rounded_probs[static_cast<size_t>(token)] =
          ::emel::kernel::detail::round_fp16_weight(weight);
    }
    for (int64_t token = active_kv_tokens; token < kv_tokens; ++token) {
      rounded_probs[static_cast<size_t>(token)] = 0.0f;
    }

    for (int64_t dim = 0; dim < head_dim; ++dim) {
      float value_sum = 0.0f;
      for (int64_t token = 0; token < active_kv_tokens; ++token) {
        const size_t kv_offset = static_cast<size_t>(((token * kv_head_count) + kv_head) * head_dim);
        value_sum += rounded_probs[static_cast<size_t>(token)] *
            v_data[kv_offset + static_cast<size_t>(dim)];
      }
      out[q_offset + static_cast<size_t>(dim)] = value_sum;
    }
  }

  return true;
}

bool run_emel_prod_style_float_value_attn_case(std::span<const float> q_data,
                                               std::span<const uint16_t> k_data,
                                               std::span<const uint16_t> v_data,
                                               const int64_t head_dim,
                                               const int64_t kv_tokens,
                                               const int64_t active_kv_tokens,
                                               const int64_t head_count,
                                               const int64_t kv_head_count,
                                               const float scale,
                                               std::vector<float> & out) {
  const std::vector<float> k_decoded = decode_fp16_storage(k_data);
  const std::vector<float> v_decoded = decode_fp16_storage(v_data);
  return run_emel_prod_style_float_value_attn_case(q_data,
                                                   k_decoded,
                                                   v_decoded,
                                                   head_dim,
                                                   kv_tokens,
                                                   active_kv_tokens,
                                                   head_count,
                                                   kv_head_count,
                                                   scale,
                                                   out);
}

bool run_ggml_flash_attn_ext_case(std::span<const float> q_data,
                                  std::span<const float> k_data,
                                  std::span<const float> v_data,
                                  const int64_t head_dim,
                                  const int64_t kv_tokens,
                                  const int64_t head_count,
                                  const int64_t kv_head_count,
                                  const float scale,
                                  std::vector<float> & out) {
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_data.size() != q_size || k_data.size() != kv_size || v_data.size() != kv_size ||
      head_dim <= 0 || kv_tokens <= 0 || head_count <= 0 || kv_head_count <= 0) {
    return false;
  }

  ggml_case_context c{};
  ggml_tensor * q_backing = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, static_cast<int64_t>(q_size));
  ggml_tensor * q = q_backing == nullptr
      ? nullptr
      : ggml_view_3d(c.ctx,
                     q_backing,
                     head_dim,
                     1,
                     head_count,
                     sizeof(float) * static_cast<size_t>(head_dim),
                     sizeof(float) * static_cast<size_t>(head_dim),
                     0);

  ggml_tensor * k_backing =
      ggml_new_tensor_1d(c.ctx, GGML_TYPE_F16, static_cast<int64_t>(kv_size));
  ggml_tensor * k = k_backing == nullptr
      ? nullptr
      : ggml_view_3d(c.ctx,
                     k_backing,
                     head_dim,
                     kv_tokens,
                     kv_head_count,
                     sizeof(ggml_fp16_t) *
                         static_cast<size_t>(head_dim * kv_head_count),
                     sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim),
                     0);

  ggml_tensor * v_backing =
      ggml_new_tensor_1d(c.ctx, GGML_TYPE_F16, static_cast<int64_t>(kv_size));
  ggml_tensor * v = v_backing == nullptr
      ? nullptr
      : ggml_view_3d(c.ctx,
                     v_backing,
                     head_dim,
                     kv_tokens,
                     kv_head_count,
                     sizeof(ggml_fp16_t) *
                         static_cast<size_t>(head_dim * kv_head_count),
                     sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim),
                     0);

  if (q == nullptr || k == nullptr || v == nullptr) {
    return false;
  }

  std::memcpy(ggml_get_data(q_backing), q_data.data(), q_size * sizeof(float));

  std::vector<ggml_fp16_t> k_f16(kv_size);
  std::vector<ggml_fp16_t> v_f16(kv_size);
  for (size_t idx = 0; idx < kv_size; ++idx) {
    k_f16[idx] = ggml_fp32_to_fp16(k_data[idx]);
    v_f16[idx] = ggml_fp32_to_fp16(v_data[idx]);
  }
  std::memcpy(ggml_get_data(k_backing), k_f16.data(), k_f16.size() * sizeof(ggml_fp16_t));
  std::memcpy(ggml_get_data(v_backing), v_f16.data(), v_f16.size() * sizeof(ggml_fp16_t));

  ggml_tensor * out_tensor =
      ggml_flash_attn_ext(c.ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
  if (out_tensor == nullptr) {
    return false;
  }
  ggml_flash_attn_ext_set_prec(out_tensor, GGML_PREC_F32);
  if (!compute_graph(c, out_tensor)) {
    return false;
  }

  const float * out_data = ggml_get_data_f32(out_tensor);
  out.assign(out_data, out_data + q_size);
  return true;
}

bool run_ggml_flash_attn_ext_case(std::span<const float> q_data,
                                  std::span<const uint16_t> k_data,
                                  std::span<const uint16_t> v_data,
                                  const int64_t head_dim,
                                  const int64_t kv_tokens,
                                  const int64_t head_count,
                                  const int64_t kv_head_count,
                                  const float scale,
                                  std::vector<float> & out) {
  const std::vector<float> k_decoded = decode_fp16_storage(k_data);
  const std::vector<float> v_decoded = decode_fp16_storage(v_data);
  return run_ggml_flash_attn_ext_case(
      q_data, k_decoded, v_decoded, head_dim, kv_tokens, head_count, kv_head_count, scale, out);
}

bool run_ggml_flash_attn_ext_masked_case(std::span<const float> q_data,
                                         std::span<const float> k_data,
                                         std::span<const float> v_data,
                                         const int64_t head_dim,
                                         const int64_t kv_tokens,
                                         const int64_t active_kv_tokens,
                                         const int64_t head_count,
                                         const int64_t kv_head_count,
                                         const float scale,
                                         std::vector<float> & out) {
  const size_t q_size = static_cast<size_t>(head_dim * head_count);
  const size_t kv_size = static_cast<size_t>(head_dim * kv_tokens * kv_head_count);
  if (q_data.size() != q_size || k_data.size() != kv_size || v_data.size() != kv_size ||
      head_dim <= 0 || kv_tokens <= 0 || active_kv_tokens <= 0 || active_kv_tokens > kv_tokens ||
      head_count <= 0 || kv_head_count <= 0) {
    return false;
  }

  ggml_case_context c{};
  ggml_tensor * q_backing = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, static_cast<int64_t>(q_size));
  ggml_tensor * q = q_backing == nullptr
      ? nullptr
      : ggml_view_3d(c.ctx,
                     q_backing,
                     head_dim,
                     1,
                     head_count,
                     sizeof(float) * static_cast<size_t>(head_dim),
                     sizeof(float) * static_cast<size_t>(head_dim),
                     0);

  ggml_tensor * k_backing =
      ggml_new_tensor_1d(c.ctx, GGML_TYPE_F16, static_cast<int64_t>(kv_size));
  ggml_tensor * k = k_backing == nullptr
      ? nullptr
      : ggml_view_3d(c.ctx,
                     k_backing,
                     head_dim,
                     kv_tokens,
                     kv_head_count,
                     sizeof(ggml_fp16_t) *
                         static_cast<size_t>(head_dim * kv_head_count),
                     sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim),
                     0);

  ggml_tensor * v_backing =
      ggml_new_tensor_1d(c.ctx, GGML_TYPE_F16, static_cast<int64_t>(kv_size));
  ggml_tensor * v = v_backing == nullptr
      ? nullptr
      : ggml_view_3d(c.ctx,
                     v_backing,
                     head_dim,
                     kv_tokens,
                     kv_head_count,
                     sizeof(ggml_fp16_t) *
                         static_cast<size_t>(head_dim * kv_head_count),
                     sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim),
                     0);

  ggml_tensor * mask = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, kv_tokens, 1);
  if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
    return false;
  }

  std::memcpy(ggml_get_data(q_backing), q_data.data(), q_size * sizeof(float));

  std::vector<ggml_fp16_t> k_f16(kv_size);
  std::vector<ggml_fp16_t> v_f16(kv_size);
  for (size_t idx = 0; idx < kv_size; ++idx) {
    k_f16[idx] = ggml_fp32_to_fp16(k_data[idx]);
    v_f16[idx] = ggml_fp32_to_fp16(v_data[idx]);
  }
  std::memcpy(ggml_get_data(k_backing), k_f16.data(), k_f16.size() * sizeof(ggml_fp16_t));
  std::memcpy(ggml_get_data(v_backing), v_f16.data(), v_f16.size() * sizeof(ggml_fp16_t));

  auto * mask_data = reinterpret_cast<ggml_fp16_t *>(ggml_get_data(mask));
  for (int64_t token = 0; token < kv_tokens; ++token) {
    const float value = token < active_kv_tokens ? 0.0f : -INFINITY;
    mask_data[token] = ggml_fp32_to_fp16(value);
  }

  ggml_tensor * out_tensor =
      ggml_flash_attn_ext(c.ctx, q, k, v, mask, scale, 0.0f, 0.0f);
  if (out_tensor == nullptr) {
    return false;
  }
  ggml_flash_attn_ext_set_prec(out_tensor, GGML_PREC_F32);
  if (!compute_graph(c, out_tensor)) {
    return false;
  }

  const float * out_data = ggml_get_data_f32(out_tensor);
  out.assign(out_data, out_data + q_size);
  return true;
}

bool run_ggml_flash_attn_ext_masked_case(std::span<const float> q_data,
                                         std::span<const uint16_t> k_data,
                                         std::span<const uint16_t> v_data,
                                         const int64_t head_dim,
                                         const int64_t kv_tokens,
                                         const int64_t active_kv_tokens,
                                         const int64_t head_count,
                                         const int64_t kv_head_count,
                                         const float scale,
                                         std::vector<float> & out) {
  const std::vector<float> k_decoded = decode_fp16_storage(k_data);
  const std::vector<float> v_decoded = decode_fp16_storage(v_data);
  return run_ggml_flash_attn_ext_masked_case(q_data,
                                             k_decoded,
                                             v_decoded,
                                             head_dim,
                                             kv_tokens,
                                             active_kv_tokens,
                                             head_count,
                                             kv_head_count,
                                             scale,
                                             out);
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
    resolve_generation_formatter_binding(state);
    print_generation_formatter_contract(stderr, state);
    const emel::error::type err = state.load.error
                                      ? state.load.err
                                      : emel::error::cast(emel::model::loader::error::internal_error);
    std::fprintf(stderr,
                 "generation load failed (fixture=%s err=%s)\n",
                 k_generation_fixture_name,
                 model_loader_error_name(err));
    return 1;
  }

  resolve_generation_formatter_binding(state);
  if (!emel::tools::generation_formatter_contract::binding_supported(
          state.formatter_binding)) {
    print_generation_formatter_contract(stderr, state);
    std::fprintf(stderr,
                 "generation load failed (fixture=%s err=%s)\n",
                 k_generation_fixture_name,
                 model_loader_error_name(
                     emel::error::cast(emel::model::loader::error::model_invalid)));
    return 1;
  }

  if (!load_generation_vocab_from_llama(opts.model_path, state)) {
    print_generation_formatter_contract(stderr, state);
    std::fprintf(stderr,
                 "generation vocab load failed (fixture=%s)\n",
                 k_generation_fixture_name);
    return 1;
  }

  const emel::error::type initialize_err = run_emel_initialize_generator(state, opts);
  if (initialize_err != emel::error::cast(emel::generator::error::none)) {
    print_generation_formatter_contract(stderr, state);
    int32_t output_weight_count = 0;
    int32_t q_norm_count = 0;
    int32_t k_norm_count = 0;
    for (uint32_t idx = 0; idx < state.model_data->n_tensors; ++idx) {
      const std::string_view name =
          emel::model::tensor_name_view(*state.model_data, state.model_data->tensors[idx]);
      output_weight_count += static_cast<int32_t>(name == "output.weight");
      q_norm_count += static_cast<int32_t>(name.find("attn_q_norm.weight") != std::string_view::npos);
      k_norm_count += static_cast<int32_t>(name.find("attn_k_norm.weight") != std::string_view::npos);
    }
    std::fprintf(stderr,
                 "generation initialize debug architecture=%.*s n_layers=%d n_tensors=%u "
                 "output_weight=%d q_norm=%d k_norm=%d\n",
                 static_cast<int>(emel::model::architecture_name_view(*state.model_data).size()),
                 emel::model::architecture_name_view(*state.model_data).data(),
                 state.model_data->n_layers,
                 state.model_data->n_tensors,
                 output_weight_count,
                 q_norm_count,
                 k_norm_count);
    emel::model::llama::detail::execution_view debug_execution{};
    if (emel::model::llama::detail::build_execution_view(*state.model_data, debug_execution) ==
        emel::error::cast(emel::model::loader::error::none)) {
      emel::model::llama::detail::block_view debug_block{};
      if (emel::model::llama::detail::lookup_block_view(debug_execution, 0, debug_block) ==
          emel::error::cast(emel::model::loader::error::none)) {
        const auto print_tensor_shape = [&](const char * label,
                                            const emel::model::data::tensor_record * tensor) {
          if (tensor == nullptr) {
            std::fprintf(stderr, "generation initialize debug %s=<null>\n", label);
            return;
          }
          const uint64_t dim0 = tensor->n_dims > 0 ? tensor->dims[0] : 0u;
          const uint64_t dim1 = tensor->n_dims > 1 ? tensor->dims[1] : 1u;
          std::fprintf(stderr,
                       "generation initialize debug %s type=%d dims=%" PRIu64 "x%" PRIu64 "\n",
                       label,
                       tensor->type,
                       dim0,
                       dim1);
        };
        print_tensor_shape("attn_q", debug_block.attention_q.tensor);
        print_tensor_shape("attn_k", debug_block.attention_k.tensor);
        print_tensor_shape("attn_q_norm", debug_block.attention_q_norm.tensor);
        print_tensor_shape("attn_k_norm", debug_block.attention_k_norm.tensor);
        print_tensor_shape("output", debug_execution.output.tensor);
      }
    } else {
      std::fprintf(stderr, "generation initialize debug build_execution_view failed\n");
    }
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
    print_generation_formatter_contract(stderr, state);
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
  emel_result.trace_available =
      emel_result.trace.token_count == emel_result.tokens_generated;
  const emel::kernel::kernel_kind generation_kernel_kind =
      state.generator->generation_kernel_kind();
  const uint64_t flash_dispatch_calls =
      state.generator->generation_flash_attention_dispatch_calls();
  const uint64_t optimized_flash_dispatch_calls =
      state.generator->generation_optimized_flash_dispatch_calls();
  const uint64_t shared_flash_dispatch_calls =
      state.generator->generation_shared_flash_dispatch_calls();
  const uint64_t native_q8_0_dispatch_calls =
      state.generator->generation_native_q8_0_dispatch_calls();
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
  const auto runtime_contract = runtime_quantized_contract_summary(state);
  if (generation_kernel_kind == emel::kernel::kernel_kind::aarch64 &&
      (flash_dispatch_calls == 0u || optimized_flash_dispatch_calls == 0u ||
       shared_flash_dispatch_calls != 0u)) {
    std::fprintf(stderr,
                 "generation flash proof failed (fixture=%s kernel_kind=%s "
                 "flash_dispatch_calls=%" PRIu64
                 " optimized_flash_dispatch_calls=%" PRIu64
                 " shared_flash_dispatch_calls=%" PRIu64 ")\n",
                 k_generation_fixture_name,
                 kernel_kind_name(generation_kernel_kind),
                 flash_dispatch_calls,
                 optimized_flash_dispatch_calls,
                 shared_flash_dispatch_calls);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  if (generation_kernel_kind != emel::kernel::kernel_kind::aarch64 &&
      (optimized_flash_dispatch_calls != 0u || shared_flash_dispatch_calls != 0u)) {
    std::fprintf(stderr,
                 "generation non-arm flash attribution failed (fixture=%s kernel_kind=%s "
                 "optimized_flash_dispatch_calls=%" PRIu64
                 " shared_flash_dispatch_calls=%" PRIu64 ")\n",
                 k_generation_fixture_name,
                 kernel_kind_name(generation_kernel_kind),
                 optimized_flash_dispatch_calls,
                 shared_flash_dispatch_calls);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  if (optimized_q2_dispatch_calls != 0u || shared_q2_dispatch_calls != 0u ||
      optimized_q3_dispatch_calls != 0u || shared_q3_dispatch_calls != 0u ||
      optimized_q6_dispatch_calls != 0u || shared_q6_dispatch_calls != 0u) {
    std::fprintf(stderr,
                 "generation quantized legacy-dispatch proof failed (fixture=%s kernel_kind=%s "
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
  if (native_q8_0_dispatch_calls == 0u) {
    std::fprintf(stderr,
                 "generation q8_0 dispatch proof failed (fixture=%s kernel_kind=%s "
                 "native_q8_0_dispatch_calls=%" PRIu64 ")\n",
                 k_generation_fixture_name,
                 kernel_kind_name(generation_kernel_kind),
                 native_q8_0_dispatch_calls);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  if (runtime_contract.disallowed_fallback != 0u || runtime_contract.explicit_no_claim != 0u) {
    std::fprintf(stderr,
                 "generation quantized contract failed (fixture=%s native_quantized=%u "
                 "approved_dense_f32_by_contract=%u disallowed_fallback=%u explicit_no_claim=%u)\n",
                 k_generation_fixture_name,
                 runtime_contract.native_quantized,
                 runtime_contract.approved_dense_f32_by_contract,
                 runtime_contract.disallowed_fallback,
                 runtime_contract.explicit_no_claim);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }

  emel::model::llama::detail::execution_view execution{};
  if (emel::model::llama::detail::build_execution_view(*state.model_data, execution) ==
      emel::error::cast(emel::model::loader::error::none)) {
    const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
    const auto audit_contract = build_quantized_contract_summary(audit);
    if (!quantized_contract_matches(runtime_contract, audit_contract)) {
      std::fprintf(stderr,
                   "generation quantized attribution mismatch (fixture=%s "
                   "runtime_native_quantized=%u runtime_approved_dense_f32_by_contract=%u "
                   "runtime_disallowed_fallback=%u runtime_explicit_no_claim=%u "
                   "audit_native_quantized=%u audit_approved_dense_f32_by_contract=%u "
                   "audit_disallowed_fallback=%u audit_explicit_no_claim=%u)\n",
                   k_generation_fixture_name,
                   runtime_contract.native_quantized,
                   runtime_contract.approved_dense_f32_by_contract,
                   runtime_contract.disallowed_fallback,
                   runtime_contract.explicit_no_claim,
                   audit_contract.native_quantized,
                   audit_contract.approved_dense_f32_by_contract,
                   audit_contract.disallowed_fallback,
                   audit_contract.explicit_no_claim);
      dump_generation_failure_surface(state, &emel_result, nullptr, opts);
      return 1;
    }
  }

  const std::filesystem::path baseline_path = opts.write_generation_baseline_path.empty()
      ? default_generation_baseline_path(opts)
      : std::filesystem::path(opts.write_generation_baseline_path);
  if (!opts.write_generation_baseline_path.empty()) {
    if (!write_generation_baseline_file(baseline_path, opts, emel_result)) {
      std::fprintf(stderr,
                   "generation baseline write failed (fixture=%s baseline=%s)\n",
                   k_generation_fixture_name,
                   baseline_path.string().c_str());
      dump_generation_failure_surface(state, &emel_result, nullptr, opts);
      return 1;
    }

    std::fprintf(stdout,
                 "generation baseline written (fixture=%s baseline=%s contract=%.*s "
                 "prompt_bytes=%zu max_tokens=%d generated_tokens=%d output_bytes=%zu)\n",
                 k_generation_fixture_name,
                 baseline_path.string().c_str(),
                 static_cast<int>(k_generation_baseline_contract.size()),
                 k_generation_baseline_contract.data(),
                 opts.text.size(),
                 opts.max_tokens,
                 emel_result.tokens_generated,
                 emel_result.output_length);
    print_generation_formatter_contract(stdout, state);
    std::fprintf(stdout,
                 "reference_impl: source=%.*s contract=%.*s baseline=%s\n",
                 static_cast<int>(k_generation_baseline_source.size()),
                 k_generation_baseline_source.data(),
                 static_cast<int>(k_generation_baseline_contract.size()),
                 k_generation_baseline_contract.data(),
                 baseline_path.string().c_str());
    dump_reference_decode_seam(state);
    return 0;
  }

  generation_baseline_record baseline_record{};
  if (!load_generation_baseline_file(baseline_path, baseline_record)) {
    std::fprintf(stderr,
                 "generation baseline load failed (fixture=%s baseline=%s)\n",
                 k_generation_fixture_name,
                 baseline_path.string().c_str());
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  if (baseline_record.fixture_name != k_generation_fixture_name ||
      baseline_record.prompt != opts.text ||
      baseline_record.max_tokens != opts.max_tokens) {
    std::fprintf(stderr,
                 "generation baseline contract mismatch (fixture=%s baseline=%s "
                 "baseline_fixture=%s baseline_prompt_bytes=%zu baseline_max_tokens=%d)\n",
                 k_generation_fixture_name,
                 baseline_path.string().c_str(),
                 baseline_record.fixture_name.c_str(),
                 baseline_record.prompt.size(),
                 baseline_record.max_tokens);
    dump_generation_failure_surface(state, &emel_result, nullptr, opts);
    return 1;
  }
  generation_result & reference_result = baseline_record.result;

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
  print_generation_formatter_contract(stdout, state);
  std::fprintf(stdout,
               "reference_impl: source=%.*s contract=%.*s baseline=%s\n",
               static_cast<int>(k_generation_baseline_source.size()),
               k_generation_baseline_source.data(),
               static_cast<int>(k_generation_baseline_contract.size()),
               k_generation_baseline_contract.data(),
               baseline_path.string().c_str());
  dump_reference_decode_seam(state);
  if (opts.attribution) {
    generation_result attributed_result{};
    generation_attribution attribution{};
    const emel::error::type attribution_err = run_attributed_native_generate(
        state.backend, *state.model_data, opts, attributed_result, attribution);
    if (attribution_err != emel::error::cast(emel::generator::error::none)) {
      std::fprintf(stderr,
                   "generation attribution failed (fixture=%s err=%s)\n",
                   k_generation_fixture_name,
                   generator_error_name(attribution_err));
      return 1;
    }
    if (!generation_results_match(attributed_result, reference_result)) {
      std::fprintf(stderr,
                   "generation attribution replay mismatch (fixture=%s emel_tokens=%d "
                   "reference_tokens=%d)\n",
                   k_generation_fixture_name,
                   attributed_result.tokens_generated,
                   reference_result.tokens_generated);
      return 1;
    }
    dump_generation_attribution(attribution);
  }
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
