#include "parity_runner.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"
#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/detail.hpp"

#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-grammar.h"

namespace {

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_internal = 3;

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

  explicit ggml_case_context(const size_t arena_bytes = 32u * 1024u * 1024u)
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
  const emel::kernel::x86_64::action::context x86_ctx{};
  const emel::kernel::aarch64::action::context aarch_ctx{};

  auto x86_exec = [&](const auto & ev) {
    return emel::kernel::x86_64::detail::execute_request(ev, x86_ctx);
  };
  auto aarch_exec = [&](const auto & ev) {
    return emel::kernel::aarch64::detail::execute_request(ev, aarch_ctx);
  };

  const bool x86_ok = run_backend_kernel_parity("x86_64", x86_exec);
  const bool aarch_ok = run_backend_kernel_parity("aarch64", aarch_exec);

  if (x86_ok && aarch_ok) {
    std::fprintf(stdout, "kernel parity ok\n");
    return 0;
  }
  return 1;
}

int run_tokenizer_parity(const emel::paritychecker::parity_options &) {
  std::fprintf(stderr, "tokenizer parity is scaffolded\n");
  return 1;
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

}  // namespace

namespace emel::paritychecker {

int run_parity(const parity_options & opts) {
  switch (opts.mode) {
    case parity_mode::gbnf_parser:
      return run_gbnf_parser_parity(opts);
    case parity_mode::kernel:
      return run_kernel_parity(opts);
    case parity_mode::tokenizer:
    default:
      return run_tokenizer_parity(opts);
  }
}

}  // namespace emel::paritychecker
