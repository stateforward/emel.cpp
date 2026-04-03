#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>
#include <ggml.h>

#include "../../tests/kernel/test_helpers.hpp"
#include "emel/generator/detail.hpp"
#include "emel/kernel/detail.hpp"
#include "../generation_formatter_contract.hpp"
#include "../generation_fixture_registry.hpp"

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

struct ggml_case_context {
  std::vector<uint8_t> arena = {};
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

bool compute_graph(ggml_case_context & c, ggml_tensor * out) {
  ggml_cgraph * graph = ggml_new_graph(c.ctx);
  if (graph == nullptr || out == nullptr) {
    return false;
  }
  ggml_build_forward_expand(graph, out);
  return ggml_graph_compute_with_ctx(c.ctx, graph, 1) == GGML_STATUS_SUCCESS;
}

bool run_ggml_flash_attn_ext_masked_case_local(std::span<const float> q_data,
                                               std::span<const uint16_t> k_data,
                                               std::span<const uint16_t> v_data,
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
  std::memcpy(ggml_get_data(k_backing), k_data.data(), k_data.size_bytes());
  std::memcpy(ggml_get_data(v_backing), v_data.data(), v_data.size_bytes());

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

using emel::tools::generation_fixture_registry::maintained_fixture;

constexpr std::string_view k_current_reference_repository =
#ifdef PARITYCHECKER_REFERENCE_REPOSITORY
    PARITYCHECKER_REFERENCE_REPOSITORY;
#else
    "unknown";
#endif

constexpr std::string_view k_current_reference_ref =
#ifdef PARITYCHECKER_REFERENCE_REF
    PARITYCHECKER_REFERENCE_REF;
#else
    {};
#endif

bool fixture_matches_current_reference_lane(const maintained_fixture & fixture) {
  if (fixture.reference_repository != k_current_reference_repository) {
    return false;
  }
  if (!fixture.reference_ref.empty() && fixture.reference_ref != k_current_reference_ref) {
    return false;
  }
  return true;
}

bool fixture_uses_append_only_baseline(const maintained_fixture & fixture) {
  return fixture.generation_parity_contract ==
         emel::tools::generation_fixture_registry::k_generation_parity_contract_append_only_baseline;
}

bool fixture_uses_live_reference_generation(const maintained_fixture & fixture) {
  return fixture.generation_parity_contract ==
         emel::tools::generation_fixture_registry::k_generation_parity_contract_live_reference_generation;
}

std::filesystem::path models_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "models";
#else
  return std::filesystem::path("tests") / "models";
#endif
}

std::filesystem::path parity_texts_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "text" / "tokenizer" / "parity_texts";
#else
  return std::filesystem::path("tests") / "text" / "tokenizer" / "parity_texts";
#endif
}

std::filesystem::path parity_snapshot_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "snapshots" / "parity";
#else
  return std::filesystem::path("snapshots") / "parity";
#endif
}

std::filesystem::path gbnf_parity_texts_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "gbnf" / "parity_texts";
#else
  return std::filesystem::path("tests") / "gbnf" / "parity_texts";
#endif
}

std::filesystem::path jinja_parity_texts_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "text" / "jinja" / "parity_texts";
#else
  return std::filesystem::path("tests") / "text" / "jinja" / "parity_texts";
#endif
}

bool file_exists(const std::filesystem::path & path) {
  std::FILE * file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

std::filesystem::path maintained_generation_fixture_path(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture) {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / fixture.fixture_rel;
#else
  return std::filesystem::path(fixture.fixture_rel);
#endif
}

std::filesystem::path maintained_generation_baseline_path(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture,
    const int32_t max_tokens) {
  return parity_snapshot_dir() /
         ("generation_" + std::string(fixture.slug) + "_prompt_hello_max_tokens_" +
          std::to_string(max_tokens) + ".txt");
}

std::vector<std::string> discover_models() {
  std::vector<std::string> models;
  const auto dir = models_dir();
  if (!std::filesystem::exists(dir)) {
    return models;
  }
  for (const auto & entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto path = entry.path();
    if (path.extension() != ".gguf") {
      continue;
    }
    // Maintained generation fixtures are covered by dedicated generation-path tests, not the
    // generic tiny-model tokenizer parity sweep.
    if (path.filename() == "Qwen3-0.6B-Q8_0.gguf" ||
        path.filename() == "Bonsai-1.7B.gguf") {
      continue;
    }
    models.push_back(path.string());
  }
  std::sort(models.begin(), models.end());
  return models;
}

struct parity_case {
  std::string label;
  std::filesystem::path text_path;
  bool add_special = false;
  bool parse_special = false;
};

std::string quote_arg_posix(const std::string & arg) {
  std::string out = "'";
  for (const char c : arg) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

std::string quote_arg_windows(const std::string & arg) {
  std::string out = "\"";
  for (const char c : arg) {
    if (c == '"') {
      out += "\\\"";
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
  return out;
}

std::string special_text_for_model(const std::filesystem::path & model_path) {
  const std::string name = model_path.filename().string();
  const auto texts = parity_texts_dir();
  if (name.find("Llama-") != std::string::npos) {
    return (texts / "special_llama.txt").string();
  }
  if (name.find("distilgpt2") != std::string::npos) {
    return (texts / "special_gpt2.txt").string();
  }
  if (name.find("bert-base-uncased") != std::string::npos) {
    return (texts / "special_bert.txt").string();
  }
  if (name.find("flan-t5") != std::string::npos) {
    return (texts / "special_t5.txt").string();
  }
  if (name.find("rwkv") != std::string::npos) {
    return (texts / "special_rwkv.txt").string();
  }
  return {};
}

std::vector<parity_case> base_cases() {
  const auto texts = parity_texts_dir();
  return {
    {"basic_add_special", texts / "basic.txt", true, false},
    {"basic_no_special", texts / "basic.txt", false, false},
    {"whitespace", texts / "whitespace.txt", true, false},
    {"unicode", texts / "unicode.txt", true, false},
    {"long", texts / "long.txt", false, false},
  };
}

TEST_CASE("paritychecker flash attention matches ggml masked flash on grouped-query head-major cache") {
  using emel::kernel::test::within_flash_online_f16_tolerance;

  emel::generator::detail::native_backend backend{};
  backend.n_head = 16;
  backend.n_head_kv = 4;
  backend.n_rep = backend.n_head / backend.n_head_kv;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 32768;

  const int32_t position = 22;
  const int32_t active_kv_tokens = position + 1;
  const size_t q_size =
      static_cast<size_t>(backend.n_head) * static_cast<size_t>(backend.head_dim);
  const size_t kv_storage_size =
      static_cast<size_t>(backend.n_head_kv) *
      static_cast<size_t>(backend.n_ctx) *
      static_cast<size_t>(backend.head_dim_kv);

  backend.q.resize(q_size);
  backend.attn_ctx.resize(q_size);
  backend.flash_key_cache.resize(kv_storage_size);
  backend.flash_value_cache.resize(kv_storage_size);

  for (size_t idx = 0; idx < backend.q.size(); ++idx) {
    const double base = static_cast<double>((idx + 1u) * 7u);
    backend.q[idx] = static_cast<float>(std::sin(base * 0.0078125));
  }

  for (int32_t token = 0; token < active_kv_tokens; ++token) {
    for (int32_t kv_head = 0; kv_head < backend.n_head_kv; ++kv_head) {
      for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
        const size_t offset = emel::generator::detail::flash_layer_cache_head_position_offset(
            backend, 0, kv_head, token, backend.head_dim_kv) + static_cast<size_t>(dim);
        const double base =
            static_cast<double>((token + 1) * (kv_head + 5) * (dim + 11));
        backend.flash_key_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::cos(base * 0.001953125)));
        backend.flash_value_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(base * 0.0029296875)));
      }
    }
  }

  const auto request = emel::generator::detail::make_flash_attn_request(backend, 0, position);
  emel::kernel::detail::flash_attn_workspace workspace = {};
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(request, workspace));

  std::vector<uint16_t> ggml_key_cache(
      static_cast<size_t>(backend.n_ctx) *
          static_cast<size_t>(backend.n_head_kv) *
          static_cast<size_t>(backend.head_dim_kv),
      0u);
  std::vector<uint16_t> ggml_value_cache(ggml_key_cache.size(), 0u);
  for (int32_t token = 0; token < active_kv_tokens; ++token) {
    for (int32_t kv_head = 0; kv_head < backend.n_head_kv; ++kv_head) {
      const size_t dst_offset =
          (static_cast<size_t>(token) * static_cast<size_t>(backend.n_head_kv) +
           static_cast<size_t>(kv_head)) *
          static_cast<size_t>(backend.head_dim_kv);
      const size_t src_offset = emel::generator::detail::flash_layer_cache_head_position_offset(
          backend, 0, kv_head, token, backend.head_dim_kv);
      std::copy_n(backend.flash_key_cache.data() + src_offset,
                  static_cast<size_t>(backend.head_dim_kv),
                  ggml_key_cache.data() + dst_offset);
      std::copy_n(backend.flash_value_cache.data() + src_offset,
                  static_cast<size_t>(backend.head_dim_kv),
                  ggml_value_cache.data() + dst_offset);
    }
  }

  std::vector<float> ggml_ctx = {};
  REQUIRE(run_ggml_flash_attn_ext_masked_case_local(
      std::span<const float>(backend.q.data(), backend.q.size()),
      std::span<const uint16_t>(ggml_key_cache.data(), ggml_key_cache.size()),
      std::span<const uint16_t>(ggml_value_cache.data(), ggml_value_cache.size()),
      backend.head_dim,
      backend.n_ctx,
      active_kv_tokens,
      backend.n_head,
      backend.n_head_kv,
      1.0f / std::sqrt(static_cast<float>(backend.head_dim)),
      ggml_ctx));
  REQUIRE(ggml_ctx.size() == backend.attn_ctx.size());

  for (size_t idx = 0; idx < backend.attn_ctx.size(); ++idx) {
    CHECK(within_flash_online_f16_tolerance(backend.attn_ctx[idx], ggml_ctx[idx]));
  }
}

std::filesystem::path paritychecker_binary_path() {
#ifdef PARITYCHECKER_BINARY_PATH
  return PARITYCHECKER_BINARY_PATH;
#else
  return std::filesystem::path("paritychecker");
#endif
}

bool run_paritychecker_process(const std::string & model, const parity_case & test_case) {
  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(paritychecker_binary_path().string());
  command += " --model ";
  command += quote_arg_windows(model);
  command += " --text-file ";
  command += quote_arg_windows(test_case.text_path.string());
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(paritychecker_binary_path().string());
  command += " --model ";
  command += quote_arg_posix(model);
  command += " --text-file ";
  command += quote_arg_posix(test_case.text_path.string());
#endif
  if (test_case.add_special) {
    command += " --add-special";
  }
  if (test_case.parse_special) {
    command += " --parse-special";
  }
  const int status = std::system(command.c_str());
  if (status == -1) {
    return false;
  }
#if defined(_WIN32)
  return status == 0;
#else
  if (!WIFEXITED(status)) {
    return false;
  }
  return WEXITSTATUS(status) == 0;
#endif
}

bool run_gbnf_paritychecker_process(const std::filesystem::path & grammar_path) {
  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(paritychecker_binary_path().string());
  command += " --gbnf --text-file ";
  command += quote_arg_windows(grammar_path.string());
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(paritychecker_binary_path().string());
  command += " --gbnf --text-file ";
  command += quote_arg_posix(grammar_path.string());
#endif
  const int status = std::system(command.c_str());
  if (status == -1) {
    return false;
  }
#if defined(_WIN32)
  return status == 0;
#else
  if (!WIFEXITED(status)) {
    return false;
  }
  return WEXITSTATUS(status) == 0;
#endif
}

bool run_kernel_paritychecker_process() {
  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(paritychecker_binary_path().string());
  command += " --kernel --text kernel";
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(paritychecker_binary_path().string());
  command += " --kernel --text kernel";
#endif
  const int status = std::system(command.c_str());
  if (status == -1) {
    return false;
  }
#if defined(_WIN32)
  return status == 0;
#else
  if (!WIFEXITED(status)) {
    return false;
  }
  return WEXITSTATUS(status) == 0;
#endif
}

bool run_jinja_paritychecker_process(const std::filesystem::path & template_path) {
  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(paritychecker_binary_path().string());
  command += " --jinja --text-file ";
  command += quote_arg_windows(template_path.string());
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(paritychecker_binary_path().string());
  command += " --jinja --text-file ";
  command += quote_arg_posix(template_path.string());
#endif
  const int status = std::system(command.c_str());
  if (status == -1) {
    return false;
  }
#if defined(_WIN32)
  return status == 0;
#else
  if (!WIFEXITED(status)) {
    return false;
  }
  return WEXITSTATUS(status) == 0;
#endif
}

struct process_capture {
  int exit_code = -1;
  std::string stdout_text;
  std::string stderr_text;
};

std::filesystem::path make_temp_capture_path(const char * stem) {
  static uint32_t counter = 0;
  ++counter;
  return std::filesystem::temp_directory_path() /
         (std::string(stem) + "-" + std::to_string(counter) + ".txt");
}

std::filesystem::path make_temp_fixture_path(const char * stem, const std::string & filename) {
  static uint32_t counter = 0;
  ++counter;
  const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                    (std::string(stem) + "-" + std::to_string(counter));
  std::filesystem::create_directories(dir);
  return dir / filename;
}

std::string read_text_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

process_capture run_generation_paritychecker_capture_with_args(
    const std::vector<std::string> & args) {
  const std::filesystem::path stdout_path = make_temp_capture_path("paritychecker-stdout");
  const std::filesystem::path stderr_path = make_temp_capture_path("paritychecker-stderr");
  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(paritychecker_binary_path().string());
  for (const auto & arg : args) {
    command += " ";
    command += quote_arg_windows(arg);
  }
  command += " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(paritychecker_binary_path().string());
  for (const auto & arg : args) {
    command += " ";
    command += quote_arg_posix(arg);
  }
  command += " > ";
  command += quote_arg_posix(stdout_path.string());
  command += " 2> ";
  command += quote_arg_posix(stderr_path.string());
#endif

  const int status = std::system(command.c_str());
  process_capture capture{};
#if defined(_WIN32)
  capture.exit_code = status;
#else
  capture.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  capture.stdout_text = read_text_file(stdout_path);
  capture.stderr_text = read_text_file(stderr_path);
  std::filesystem::remove(stdout_path);
  std::filesystem::remove(stderr_path);
  return capture;
}

process_capture run_generation_paritychecker_capture(const std::filesystem::path & model_path,
                                                     const std::string & text,
                                                     const int32_t max_tokens = 1) {
  return run_generation_paritychecker_capture_with_args({
    "--generation",
    "--model",
    model_path.string(),
    "--text",
    text,
    "--max-tokens",
    std::to_string(max_tokens),
  });
}

int parse_named_metric(const std::string & text, const std::string & key) {
  const std::string needle = key + "=";
  const size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    return -1;
  }

  size_t value_pos = key_pos + needle.size();
  size_t value_end = value_pos;
  while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9') {
    ++value_end;
  }
  if (value_pos == value_end) {
    return -1;
  }
  return std::atoi(text.substr(value_pos, value_end - value_pos).c_str());
}

int parse_named_metric_on_line(const std::string & text,
                               const std::string & line_prefix,
                               const std::string & key) {
  const size_t line_pos = text.find(line_prefix);
  if (line_pos == std::string::npos) {
    return -1;
  }

  const size_t metric_pos = text.find(key + "=", line_pos);
  if (metric_pos == std::string::npos) {
    return -1;
  }

  const size_t line_end = text.find('\n', line_pos);
  if (line_end != std::string::npos && metric_pos > line_end) {
    return -1;
  }

  const size_t value_pos = metric_pos + key.size() + 1u;
  size_t value_end = value_pos;
  while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9') {
    ++value_end;
  }
  if (value_pos == value_end) {
    return -1;
  }
  return std::atoi(text.substr(value_pos, value_end - value_pos).c_str());
}

int parse_kernel_dispatch_calls(const std::string & text) {
  const size_t line_pos = text.find("kernel_dispatch:");
  if (line_pos == std::string::npos) {
    return -1;
  }

  const size_t calls_pos = text.find("calls=", line_pos);
  if (calls_pos == std::string::npos) {
    return -1;
  }

  const size_t value_pos = calls_pos + std::string("calls=").size();
  size_t value_end = value_pos;
  while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9') {
    ++value_end;
  }
  if (value_pos == value_end) {
    return -1;
  }
  return std::atoi(text.substr(value_pos, value_end - value_pos).c_str());
}

int parse_flash_dispatch_calls(const std::string & text) {
  const size_t line_pos = text.find("flash_dispatch:");
  if (line_pos == std::string::npos) {
    return -1;
  }

  const size_t calls_pos = text.find("calls=", line_pos);
  if (calls_pos == std::string::npos) {
    return -1;
  }

  const size_t value_pos = calls_pos + std::string("calls=").size();
  size_t value_end = value_pos;
  while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9') {
    ++value_end;
  }
  if (value_pos == value_end) {
    return -1;
  }
  return std::atoi(text.substr(value_pos, value_end - value_pos).c_str());
}

int parse_flash_dispatch_metric(const std::string & text, const std::string & key) {
  const size_t line_pos = text.find("flash_dispatch:");
  if (line_pos == std::string::npos) {
    return -1;
  }

  const size_t metric_pos = text.find(key + "=", line_pos);
  if (metric_pos == std::string::npos) {
    return -1;
  }

  const size_t value_pos = metric_pos + key.size() + 1u;
  size_t value_end = value_pos;
  while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9') {
    ++value_end;
  }
  if (value_pos == value_end) {
    return -1;
  }
  return std::atoi(text.substr(value_pos, value_end - value_pos).c_str());
}

std::string_view expected_generation_kernel_kind() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__wasm__)
  return "wasm";
#else
  return "x86_64";
#endif
}

void check_generation_flash_attribution(const process_capture & capture) {
  CHECK(parse_flash_dispatch_calls(capture.stdout_text) >= 0);
  CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") >= 0);
  CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") >= 0);
  CHECK(parse_flash_dispatch_calls(capture.stdout_text) > 0);
  if (expected_generation_kernel_kind() == "aarch64") {
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") > 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  } else {
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  }
}

void check_generation_quantized_attribution(const process_capture & capture) {
  CHECK(parse_named_metric(capture.stdout_text, "optimized_q2_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_q2_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_q3_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_q3_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_q6_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_q6_dispatch_calls") >= 0);
  const int native_q8_0_dispatch_calls =
      parse_named_metric(capture.stdout_text, "native_q8_0_dispatch_calls");
  const int packed_q8_0_dispatch_calls =
      parse_named_metric(capture.stdout_text, "packed_q8_0_dispatch_calls");
  CHECK(native_q8_0_dispatch_calls >= 0);
  CHECK(packed_q8_0_dispatch_calls >= 0);
  CHECK(native_q8_0_dispatch_calls + packed_q8_0_dispatch_calls > 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_q2_dispatch_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_q2_dispatch_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_q3_dispatch_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_q3_dispatch_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_q6_dispatch_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_q6_dispatch_calls") == 0);
}

void check_generation_quantized_stage_audit(const process_capture & capture) {
  CHECK(capture.stdout_text.find("quantized_runtime_contract:") != std::string::npos);
  CHECK(capture.stdout_text.find("quantized_stage_inventory:") != std::string::npos);
  CHECK(capture.stdout_text.find("quantized_stage_audit: stage=token_embedding") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("quantized_stage_audit: stage=attention_q") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("quantized_stage_audit: stage=attention_q_norm") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("quantized_stage_audit: stage=attention_k_norm") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("approved_dense_f32_by_contract") != std::string::npos);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "native_quantized") == 9);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "approved_dense_f32_by_contract") == 5);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "disallowed_fallback") == 0);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "explicit_no_claim") == 0);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_stage_inventory:",
                                   "native_quantized") == 9);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_stage_inventory:",
                                   "approved_dense_f32_by_contract") == 5);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_stage_inventory:",
                                   "disallowed_fallback") == 0);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_stage_inventory:",
                                   "explicit_no_claim") == 0);
}

}  // namespace

TEST_CASE("paritychecker matches llama tokens across tiny models") {
  const std::vector<std::string> models = discover_models();
  const std::vector<parity_case> cases = base_cases();

  REQUIRE(!models.empty());
  for (const auto & model : models) {
    INFO("model: " << model);
    REQUIRE(file_exists(std::filesystem::path(model)));
    for (const auto & test_case : cases) {
      INFO("case: " << test_case.label);
      REQUIRE(file_exists(test_case.text_path));
      CHECK(run_paritychecker_process(model, test_case));
    }
    const std::string special_text = special_text_for_model(model);
    if (!special_text.empty()) {
      INFO("case: special_parse");
      REQUIRE(file_exists(std::filesystem::path(special_text)));
      parity_case special_case;
      special_case.label = "special_parse";
      special_case.text_path = special_text;
      special_case.add_special = true;
      special_case.parse_special = true;
      CHECK(run_paritychecker_process(model, special_case));
    }
  }
}

TEST_CASE("paritychecker matches llama gbnf parser outputs") {
  const auto grammar_dir = gbnf_parity_texts_dir();
  const std::vector<std::filesystem::path> cases = {
      grammar_dir / "valid_basic.gbnf",
      grammar_dir / "valid_complex.gbnf",
      grammar_dir / "invalid_token_name.gbnf",
  };

  for (const auto & grammar_path : cases) {
    INFO("case: " << grammar_path.string());
    REQUIRE(file_exists(grammar_path));
    CHECK(run_gbnf_paritychecker_process(grammar_path));
  }
}

TEST_CASE("paritychecker matches llama kernel outputs") {
  CHECK(run_kernel_paritychecker_process());
}

TEST_CASE("paritychecker help describes the maintained generation fixture contract") {
  const process_capture capture = run_generation_paritychecker_capture_with_args({"--help"});

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());
  CHECK(capture.stderr_text.find("--generation mode requires --model one maintained fixture") !=
        std::string::npos);
  CHECK(capture.stderr_text.find("current reference lane: repo=") != std::string::npos);
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures) {
    if (fixture_matches_current_reference_lane(fixture)) {
      CHECK(capture.stderr_text.find(std::string(fixture.fixture_rel)) != std::string::npos);
    }
  }
}

TEST_CASE("paritychecker generation reports a deterministic missing-model failure") {
  const auto missing_model_path = models_dir() / "does-not-exist.gguf";
  REQUIRE(!file_exists(missing_model_path));

  const process_capture capture = run_generation_paritychecker_capture_with_args({
    "--generation",
    "--model",
    missing_model_path.string(),
    "--text",
    "hello",
    "--max-tokens",
    "1",
  });

  CHECK(capture.exit_code == 1);
  CHECK(capture.stdout_text.find("generation parity ok") == std::string::npos);
  CHECK(capture.stderr_text.find("generation load failed: missing model file") !=
        std::string::npos);
}

TEST_CASE("generation formatter contract classifier models supported and unsupported templates explicitly") {
  std::string supported_template = {};
  for (const std::string_view marker :
       emel::tools::generation_formatter_contract::k_supported_primary_template_markers) {
    supported_template.append(marker);
    supported_template.push_back('\n');
  }

  const auto supported =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          supported_template, 0u);
  CHECK(emel::tools::generation_formatter_contract::binding_supported(supported));
  CHECK(supported.contract ==
        emel::tools::generation_formatter_contract::k_supported_contract);

  std::string formatted_prompt = {};
  CHECK(emel::tools::generation_formatter_contract::format_single_user_prompt(
      supported, "hello", formatted_prompt));
  CHECK(formatted_prompt ==
        "<|startoftext|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");

  std::string supported_qwen_template = {};
  for (const std::string_view marker :
       emel::tools::generation_formatter_contract::k_supported_qwen_primary_template_markers) {
    supported_qwen_template.append(marker);
    supported_qwen_template.push_back('\n');
  }

  const auto supported_qwen =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          supported_qwen_template, 0u);
  CHECK(emel::tools::generation_formatter_contract::binding_supported(supported_qwen));
  CHECK(supported_qwen.contract ==
        emel::tools::generation_formatter_contract::k_supported_qwen_contract);

  std::string formatted_qwen_prompt = {};
  CHECK(emel::tools::generation_formatter_contract::format_single_user_prompt(
      supported_qwen, "hello", formatted_qwen_prompt));
  CHECK(formatted_qwen_prompt ==
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");

  std::string supported_qwen_tool_template = {};
  for (const std::string_view marker :
       emel::tools::generation_formatter_contract::k_supported_qwen_tool_markers) {
    supported_qwen_tool_template.append(marker);
    supported_qwen_tool_template.push_back('\n');
  }

  const auto supported_qwen_tool =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          supported_qwen_tool_template, 0u);
  CHECK(emel::tools::generation_formatter_contract::binding_supported(
      supported_qwen_tool));
  CHECK(supported_qwen_tool.contract ==
        emel::tools::generation_formatter_contract::k_supported_qwen_tool_contract);

  std::string formatted_qwen_tool_prompt = {};
  CHECK(emel::tools::generation_formatter_contract::format_single_user_prompt(
      supported_qwen_tool, "hello", formatted_qwen_tool_prompt));
  CHECK(formatted_qwen_tool_prompt ==
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");

  const auto unsupported =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          "{{ unsupported }}", 0u);
  CHECK_FALSE(emel::tools::generation_formatter_contract::binding_supported(unsupported));
  CHECK(unsupported.contract ==
        emel::tools::generation_formatter_contract::k_unsupported_template_contract);

  const auto named_variant =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          supported_template, 1u);
  CHECK(emel::tools::generation_formatter_contract::binding_supported(named_variant));
  CHECK(named_variant.contract ==
        emel::tools::generation_formatter_contract::k_supported_contract);
}

TEST_CASE("paritychecker generation keeps append-only maintained baselines for supported fixtures") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures) {
    if (!fixture_matches_current_reference_lane(fixture) ||
        !fixture_uses_append_only_baseline(fixture)) {
      continue;
    }
    INFO("fixture: " << fixture.name);
    for (const int32_t max_tokens : {1, 10, 100, 1000}) {
      const std::filesystem::path baseline_path =
          maintained_generation_baseline_path(fixture, max_tokens);
      INFO("baseline: " << baseline_path.string());
      CHECK(file_exists(baseline_path));
    }
  }
}

TEST_CASE("paritychecker matches maintained generation baselines across baseline fixtures") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures) {
    if (!fixture_matches_current_reference_lane(fixture) ||
        !fixture_uses_append_only_baseline(fixture)) {
      continue;
    }
    const std::filesystem::path model_path = maintained_generation_fixture_path(fixture);
    INFO("fixture: " << fixture.name);
    REQUIRE(file_exists(model_path));

    const process_capture capture = run_generation_paritychecker_capture(model_path, "hello");

    CHECK(capture.exit_code == 0);
    CHECK(capture.stderr_text.empty());
    CHECK(capture.stdout_text.find("generation parity ok") != std::string::npos);
    CHECK(capture.stdout_text.find(std::string("fixture=") + std::string(fixture.name)) !=
          std::string::npos);
    CHECK(capture.stdout_text.find("formatter_contract=") != std::string::npos);
    CHECK(capture.stdout_text.find("reference_impl:") != std::string::npos);
    CHECK(capture.stdout_text.find("repo=") != std::string::npos);
    CHECK(capture.stdout_text.find("ref=") != std::string::npos);
    CHECK(capture.stdout_text.find("contract=generation_online_f16_final_normalize_v1") !=
          std::string::npos);
    CHECK(capture.stdout_text.find("baseline=") !=
          std::string::npos);
  }
}

TEST_CASE("paritychecker matches maintained live reference generation across live fixtures") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures) {
    if (!fixture_matches_current_reference_lane(fixture) ||
        !fixture_uses_live_reference_generation(fixture)) {
      continue;
    }
    const std::filesystem::path model_path = maintained_generation_fixture_path(fixture);
    INFO("fixture: " << fixture.name);
    REQUIRE(file_exists(model_path));

    const process_capture capture = run_generation_paritychecker_capture(model_path, "hello");

    CHECK(capture.exit_code == 0);
    CHECK(capture.stderr_text.empty());
    CHECK(capture.stdout_text.find("generation parity ok") != std::string::npos);
    CHECK(capture.stdout_text.find(std::string("fixture=") + std::string(fixture.name)) !=
          std::string::npos);
    CHECK(capture.stdout_text.find("formatter_contract=") != std::string::npos);
    CHECK(capture.stdout_text.find("reference_impl:") != std::string::npos);
    CHECK(capture.stdout_text.find("repo=") != std::string::npos);
    CHECK(capture.stdout_text.find("ref=") != std::string::npos);
    CHECK(capture.stdout_text.find("contract=live_reference_generation") !=
          std::string::npos);
  }
}

TEST_CASE("paritychecker maintained generation fixtures reject same-basename files outside tests/models") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures) {
    if (!fixture_matches_current_reference_lane(fixture)) {
      continue;
    }
    const std::filesystem::path impostor_model_path =
        make_temp_fixture_path("paritychecker-fixture", std::string(fixture.name));
    {
      std::ofstream impostor(impostor_model_path, std::ios::binary);
      REQUIRE(impostor.good());
      impostor << "not-a-real-gguf";
    }

    const process_capture capture = run_generation_paritychecker_capture(impostor_model_path, "hello");

    INFO("fixture: " << fixture.name);
    CHECK(capture.exit_code == 1);
    CHECK(capture.stdout_text.find("generation parity ok") == std::string::npos);
    CHECK(capture.stderr_text.find("generation requires maintained fixture path") !=
          std::string::npos);

    std::filesystem::remove(impostor_model_path);
    std::filesystem::remove(impostor_model_path.parent_path());
  }
}

TEST_CASE("paritychecker matches llama jinja parser and formatter outputs") {
  const auto template_dir = jinja_parity_texts_dir();
  const std::vector<std::filesystem::path> cases = {
      template_dir / "literal_text.j2",
      template_dir / "invalid_unclosed_expression.j2",
  };

  for (const auto & template_path : cases) {
    INFO("case: " << template_path.string());
    REQUIRE(file_exists(template_path));
    CHECK(run_jinja_paritychecker_process(template_path));
  }
}
