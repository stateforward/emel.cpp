#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include "llama.h"

#include "../generation_formatter_contract.hpp"
#include "../generation_fixture_registry.hpp"
#include "parity_assets.hpp"
#include "parity_dependency_manifest.hpp"
#include "parity_engine.hpp"

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

std::filesystem::path models_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "models";
#else
  return std::filesystem::path("tests") / "models";
#endif
}

std::filesystem::path repo_root_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  return std::filesystem::path(PARITYCHECKER_REPO_ROOT);
#else
  return std::filesystem::current_path();
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

struct llama_backend_guard {
  llama_backend_guard() {
    llama_backend_init();
  }

  ~llama_backend_guard() {
    llama_backend_free();
  }
};

bool reference_tokenizer_lane_supported(const std::filesystem::path & model_path) {
  llama_backend_guard backend_guard{};
  llama_model_params model_params = llama_model_default_params();
  model_params.vocab_only = true;
  model_params.check_tensors = false;

  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      llama_model_load_from_file(model_path.string().c_str(), model_params),
      llama_model_free);
  return model != nullptr && llama_model_get_vocab(model.get()) != nullptr;
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
    // The canonical Qwen generation fixture is covered by dedicated maintained-generation tests,
    // not the generic tiny-model tokenizer parity sweep.
    if (path.filename() == "Qwen3-0.6B-Q8_0.gguf") {
      continue;
    }
    if (!reference_tokenizer_lane_supported(path)) {
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

#if !defined(_WIN32)
process_capture run_paritychecker_capture_with_stdin(const std::vector<std::string> & args,
                                                     const std::string & stdin_text) {
  const std::filesystem::path stdout_path = make_temp_capture_path("paritychecker-stdin-stdout");
  const std::filesystem::path stderr_path = make_temp_capture_path("paritychecker-stdin-stderr");

  std::string command = "ulimit -s 8192; printf %s ";
  command += quote_arg_posix(stdin_text);
  command += " | ";
  command += quote_arg_posix(paritychecker_binary_path().string());
  for (const auto & arg : args) {
    command += " ";
    command += quote_arg_posix(arg);
  }
  command += " > ";
  command += quote_arg_posix(stdout_path.string());
  command += " 2> ";
  command += quote_arg_posix(stderr_path.string());

  const int status = std::system(command.c_str());
  process_capture capture{};
  capture.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  capture.stdout_text = read_text_file(stdout_path);
  capture.stderr_text = read_text_file(stderr_path);
  std::filesystem::remove(stdout_path);
  std::filesystem::remove(stderr_path);
  return capture;
}
#endif

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
                                   "native_quantized") == 8);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "approved_dense_f32_by_contract") == 6);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "disallowed_fallback") == 0);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_runtime_contract:",
                                   "explicit_no_claim") == 0);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_stage_inventory:",
                                   "native_quantized") == 8);
  CHECK(parse_named_metric_on_line(capture.stdout_text,
                                   "quantized_stage_inventory:",
                                   "approved_dense_f32_by_contract") == 6);
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
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    CHECK(capture.stderr_text.find(std::string(fixture.fixture_rel)) != std::string::npos);
  }
  CHECK(capture.stderr_text.find("snapshots/parity/") != std::string::npos);
  CHECK(capture.stderr_text.find("append-only") != std::string::npos);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
}

TEST_CASE("parity runner owns cli parsing and validation") {
  const process_capture no_args = run_generation_paritychecker_capture_with_args({});
  CHECK(no_args.exit_code == 2);
  CHECK(no_args.stdout_text.empty());
  CHECK(no_args.stderr_text.find("usage:") != std::string::npos);

  const process_capture invalid_combo = run_generation_paritychecker_capture_with_args({
    "--gbnf",
    "--text",
    "root ::= \"a\"",
    "--attribution",
  });
  CHECK(invalid_combo.exit_code == 2);
  CHECK(invalid_combo.stdout_text.empty());
  CHECK(invalid_combo.stderr_text.find("usage:") != std::string::npos);

  const std::string main_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_main.cpp");
  REQUIRE_FALSE(main_source.empty());
  CHECK(main_source.find("run_parity_cli(argc, argv)") != std::string::npos);
  CHECK(main_source.find("parse_args") == std::string::npos);
  CHECK(main_source.find("print_usage") == std::string::npos);
  CHECK(main_source.find("generation_fixture_registry") == std::string::npos);
  CHECK(main_source.find("std::strtol") == std::string::npos);
  CHECK(main_source.find("std::fopen") == std::string::npos);

  const std::string runner_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_runner.cpp");
  REQUIRE_FALSE(runner_source.empty());
  CHECK(runner_source.find("int run_parity_cli(int argc, char ** argv)") != std::string::npos);
  CHECK(runner_source.find("bool parse_args(int argc, char ** argv") != std::string::npos);
  CHECK(runner_source.find("void print_usage(const char * exe)") != std::string::npos);
}

TEST_CASE("parity runner accepts non-seekable text-file inputs") {
#if defined(_WIN32)
  MESSAGE("POSIX /dev/stdin compatibility test skipped on Windows");
#else
  const process_capture capture = run_paritychecker_capture_with_stdin({
    "--gbnf",
    "--text-file",
    "/dev/stdin",
  }, "root ::= [a-z]+");

  CHECK(capture.exit_code == 0);
  CHECK(capture.stdout_text.find("parity ok") != std::string::npos);
  CHECK(capture.stderr_text.empty());
#endif
}

TEST_CASE("parity assets resolve maintained generation fixtures centrally") {
  const std::string fixture_list =
      emel::paritychecker::assets::maintained_generation_fixture_list();

  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    const auto expected_path =
        emel::paritychecker::assets::expected_generation_fixture_path(fixture);
    const auto normalized_expected =
        emel::paritychecker::assets::normalize_path(expected_path);
    REQUIRE_FALSE(normalized_expected.empty());
    CHECK(normalized_expected.filename().string() == std::string(fixture.name));

    const auto * resolved =
        emel::paritychecker::assets::find_generation_fixture(expected_path.string());
    REQUIRE(resolved != nullptr);
    CHECK(resolved->slug == fixture.slug);

    const auto impostor_path =
        make_temp_fixture_path("paritychecker-fixture-impostor", std::string(fixture.name));
    CHECK(emel::paritychecker::assets::find_generation_fixture(impostor_path.string()) ==
          nullptr);
    CHECK(fixture_list.find(std::string(fixture.fixture_rel)) != std::string::npos);
  }
}

TEST_CASE("parity assets own common runner file helpers") {
  const auto reference_ref = repo_root_dir() / "tools" / "paritychecker" / "reference_ref.txt";
  REQUIRE(emel::paritychecker::assets::file_exists(reference_ref.string()));

  std::vector<uint8_t> bytes;
  REQUIRE(emel::paritychecker::assets::read_file_bytes(reference_ref.string(), bytes));
  CHECK_FALSE(bytes.empty());

  const std::string source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_engines.cpp");
  REQUIRE_FALSE(source.empty());
  CHECK(source.find("#include \"parity_assets.hpp\"") != std::string::npos);
  CHECK(source.find("bool file_exists(") == std::string::npos);
  CHECK(source.find("bool read_file_bytes(") == std::string::npos);
  CHECK(source.find("const maintained_generation_fixture * find_generation_fixture(") ==
        std::string::npos);
  CHECK(source.find("std::string maintained_generation_fixture_list(") == std::string::npos);
}

TEST_CASE("parity runner dispatches through explicit engine adapters") {
  const auto assert_engine =
      [](const emel::paritychecker::parity_mode mode, const std::string_view name) {
        const emel::paritychecker::engine_adapter * engine =
            emel::paritychecker::find_engine(mode);
        REQUIRE(engine != nullptr);
        CHECK(engine->mode == mode);
        CHECK(engine->name == name);
        CHECK(engine->run != nullptr);
      };

  assert_engine(emel::paritychecker::parity_mode::tokenizer, "tokenizer");
  assert_engine(emel::paritychecker::parity_mode::gbnf_parser, "gbnf_parser");
  assert_engine(emel::paritychecker::parity_mode::kernel, "kernel");
  assert_engine(emel::paritychecker::parity_mode::jinja, "jinja");
  assert_engine(emel::paritychecker::parity_mode::generation, "generation");
  CHECK(emel::paritychecker::find_engine(
            static_cast<emel::paritychecker::parity_mode>(255)) == nullptr);

  const std::string runner_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_runner.cpp");
  REQUIRE_FALSE(runner_source.empty());
  CHECK(runner_source.find("find_engine(opts.mode)") != std::string::npos);
  CHECK(runner_source.find("run_generation_harness_contract") == std::string::npos);
  CHECK(runner_source.find("run_gbnf_parser_parity") == std::string::npos);
  CHECK(runner_source.find("run_kernel_parity") == std::string::npos);
  CHECK(runner_source.find("run_jinja_parity") == std::string::npos);
  CHECK(runner_source.find("run_tokenizer_parity") == std::string::npos);
  CHECK(runner_source.find("#include \"llama") == std::string::npos);
  CHECK(runner_source.find("#include \"ggml") == std::string::npos);
}

TEST_CASE("paritychecker build registration keeps runner and engine sources modular") {
  const std::string cmake_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "CMakeLists.txt");
  REQUIRE_FALSE(cmake_source.empty());

  CHECK(cmake_source.find("set(PARITYCHECKER_RUNNER_SOURCES") != std::string::npos);
  CHECK(cmake_source.find("set(PARITYCHECKER_ENGINE_REGISTRATION_SOURCES") !=
        std::string::npos);
  CHECK(cmake_source.find("set(PARITYCHECKER_ENGINE_IMPLEMENTATION_SOURCES") !=
        std::string::npos);
  CHECK(cmake_source.find("set(PARITYCHECKER_TOKENIZER_ENGINE_SOURCES") !=
        std::string::npos);
  CHECK(cmake_source.find("set(PARITYCHECKER_REFERENCE_SUPPORT_SOURCES") !=
        std::string::npos);
  CHECK(cmake_source.find("set(PARITYCHECKER_COMMON_SOURCES") != std::string::npos);
  CHECK(cmake_source.find("add_executable(paritychecker\n"
                          "  ${PARITYCHECKER_CLI_SOURCES}\n"
                          "  ${PARITYCHECKER_COMMON_SOURCES}") != std::string::npos);
  CHECK(cmake_source.find("add_executable(paritychecker_tests\n"
                          "  ${CMAKE_CURRENT_SOURCE_DIR}/paritychecker_tests.cpp\n"
                          "  ${PARITYCHECKER_COMMON_SOURCES}") != std::string::npos);

  const std::string engine_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_engine.cpp");
  REQUIRE_FALSE(engine_source.empty());
  CHECK(engine_source.find("default:\n      return nullptr;") != std::string::npos);
  CHECK(engine_source.find("default:\n      return &k_tokenizer_engine;") ==
        std::string::npos);
}

TEST_CASE("parity dependency manifest covers registered engines conservatively") {
  namespace manifest = emel::paritychecker::dependency_manifest;

  const auto records = manifest::records();
  REQUIRE_FALSE(records.empty());
  CHECK(manifest::kind_name(manifest::dependency_kind::source) == "source");
  CHECK(manifest::kind_name(manifest::dependency_kind::snapshot) == "snapshot");
  CHECK_FALSE(manifest::requires_full_gate({}));
  CHECK(manifest::requires_full_gate({.missing = true}));
  CHECK(manifest::requires_full_gate({.stale = true}));
  CHECK(manifest::requires_full_gate({.uncertain = true}));

  const auto assert_runner =
      [](const emel::paritychecker::parity_mode mode,
         const bool needs_fixture,
         const bool needs_model,
         const bool needs_snapshot) {
        namespace manifest = emel::paritychecker::dependency_manifest;

        const emel::paritychecker::engine_adapter * engine =
            emel::paritychecker::find_engine(mode);
        REQUIRE(engine != nullptr);

        bool has_source = false;
        bool has_config = false;
        bool has_fixture = false;
        bool has_model = false;
        bool has_script = false;
        bool has_snapshot = false;
        const auto runner_records = manifest::records_for(mode);
        REQUIRE_FALSE(runner_records.empty());
        for (const auto & record : runner_records) {
          CHECK(record.runner == engine->name);
          has_source = has_source || record.kind == manifest::dependency_kind::source;
          has_config = has_config || record.kind == manifest::dependency_kind::config;
          has_fixture = has_fixture || record.kind == manifest::dependency_kind::fixture;
          has_model = has_model || record.kind == manifest::dependency_kind::model;
          has_script = has_script || record.kind == manifest::dependency_kind::script;
          has_snapshot = has_snapshot || record.kind == manifest::dependency_kind::snapshot;
        }

        CHECK(has_source);
        CHECK(has_config);
        CHECK(has_script);
        CHECK(has_fixture == needs_fixture);
        CHECK(has_model == needs_model);
        CHECK(has_snapshot == needs_snapshot);
      };

  assert_runner(emel::paritychecker::parity_mode::tokenizer, true, true, false);
  assert_runner(emel::paritychecker::parity_mode::gbnf_parser, true, false, false);
  assert_runner(emel::paritychecker::parity_mode::kernel, false, false, false);
  assert_runner(emel::paritychecker::parity_mode::jinja, true, false, false);
  assert_runner(emel::paritychecker::parity_mode::generation, true, true, true);
  CHECK(manifest::records_for(static_cast<emel::paritychecker::parity_mode>(255)).empty());
}

TEST_CASE("parity dependency manifest renders and writes a deterministic format") {
  namespace manifest = emel::paritychecker::dependency_manifest;

  const std::string rendered = manifest::render();
  REQUIRE_FALSE(rendered.empty());
  CHECK(rendered == manifest::render());
  CHECK(rendered.rfind(std::string(manifest::k_schema) + "\n", 0) == 0);
  CHECK(rendered.find("full_gate_on=missing,stale,uncertain\n") != std::string::npos);
  CHECK(rendered.find("record runner=tokenizer mode=tokenizer kind=fixture "
                      "path=tests/text/tokenizer/parity_texts reason=tokenizer_text_cases\n") !=
        std::string::npos);
  CHECK(rendered.find("record runner=generation mode=generation kind=source "
                      "path=tools/generation_fixture_registry.hpp "
                      "reason=maintained_fixture_registration\n") != std::string::npos);
  CHECK(rendered.find("record runner=generation mode=generation kind=snapshot "
                      "path=snapshots/parity "
                      "reason=append_only_generation_baselines\n") != std::string::npos);

  const std::filesystem::path manifest_path =
      make_temp_capture_path("parity-dependency-manifest");
  REQUIRE(manifest::write(manifest_path));
  CHECK(read_text_file(manifest_path) == rendered);
  std::filesystem::remove(manifest_path);

  const std::string cmake_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "CMakeLists.txt");
  REQUIRE_FALSE(cmake_source.empty());
  CHECK(cmake_source.find("set(PARITYCHECKER_MANIFEST_SOURCES") != std::string::npos);
  CHECK(cmake_source.find("${PARITYCHECKER_MANIFEST_SOURCES}") != std::string::npos);

  const std::string docs =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "dependency_manifest.md");
  REQUIRE_FALSE(docs.empty());
  CHECK(docs.find(manifest::k_schema) != std::string::npos);
  CHECK(docs.find("full_gate_on=missing,stale,uncertain") != std::string::npos);
  CHECK(docs.find("Missing, stale, or uncertain") != std::string::npos);
}

TEST_CASE("paritychecker cli emits dependency manifest for operators") {
  namespace manifest = emel::paritychecker::dependency_manifest;

  const std::filesystem::path manifest_path =
      make_temp_capture_path("paritychecker-cli-dependency-manifest");
  const process_capture capture = run_generation_paritychecker_capture_with_args({
    "--write-dependency-manifest",
    manifest_path.string(),
  });

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("dependency_manifest: action=write") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("schema=parity_dependency_manifest/v1") !=
        std::string::npos);
  CHECK(read_text_file(manifest_path) == manifest::render());
  std::filesystem::remove(manifest_path);

  const std::string runner_source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_runner.cpp");
  REQUIRE_FALSE(runner_source.empty());
  CHECK(runner_source.find("--write-dependency-manifest") != std::string::npos);
  CHECK(runner_source.find("dependency_manifest::write") != std::string::npos);
}

TEST_CASE("paritychecker cli reports dependency manifest freshness conservatively") {
  namespace manifest = emel::paritychecker::dependency_manifest;

  const std::filesystem::path manifest_path =
      make_temp_capture_path("paritychecker-cli-dependency-manifest-freshness");
  REQUIRE(manifest::write(manifest_path));

  const process_capture fresh = run_generation_paritychecker_capture_with_args({
    "--check-dependency-manifest",
    manifest_path.string(),
  });
  CHECK(fresh.exit_code == 0);
  CHECK(fresh.stderr_text.empty());
  CHECK(fresh.stdout_text.find("dependency_manifest: action=check") != std::string::npos);
  CHECK(fresh.stdout_text.find("full_gate=0") != std::string::npos);
  CHECK(fresh.stdout_text.find("reason=fresh") != std::string::npos);

  const process_capture uncertain = run_generation_paritychecker_capture_with_args({
    "--check-dependency-manifest",
    manifest_path.string(),
    "--dependency-manifest-uncertain",
  });
  CHECK(uncertain.exit_code == 3);
  CHECK(uncertain.stdout_text.find("full_gate=1") != std::string::npos);
  CHECK(uncertain.stdout_text.find("reason=uncertain") != std::string::npos);

  {
    std::ofstream stale_manifest(manifest_path, std::ios::binary);
    stale_manifest << "stale manifest\n";
  }
  const process_capture stale = run_generation_paritychecker_capture_with_args({
    "--check-dependency-manifest",
    manifest_path.string(),
  });
  CHECK(stale.exit_code == 3);
  CHECK(stale.stdout_text.find("full_gate=1") != std::string::npos);
  CHECK(stale.stdout_text.find("reason=stale") != std::string::npos);

  std::filesystem::remove(manifest_path);
  const process_capture missing = run_generation_paritychecker_capture_with_args({
    "--check-dependency-manifest",
    manifest_path.string(),
  });
  CHECK(missing.exit_code == 3);
  CHECK(missing.stdout_text.find("full_gate=1") != std::string::npos);
  CHECK(missing.stdout_text.find("reason=missing") != std::string::npos);
}

TEST_CASE("quality gate escalates parity on dependency manifest freshness gaps") {
  namespace manifest = emel::paritychecker::dependency_manifest;

  const std::string baseline =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "dependency_manifest.txt");
  CHECK(baseline == manifest::render());

  const std::string gate_source =
      read_text_file(repo_root_dir() / "scripts" / "quality_gates.sh");
  REQUIRE_FALSE(gate_source.empty());
  CHECK(gate_source.find("PARITY_DEPENDENCY_MANIFEST_BASELINE") != std::string::npos);
  CHECK(gate_source.find("--write-dependency-manifest") != std::string::npos);
  CHECK(gate_source.find("--check-dependency-manifest") != std::string::npos);
  CHECK(gate_source.find("EMEL_PARITY_DEPENDENCY_MANIFEST_UNCERTAIN") != std::string::npos);
  CHECK(gate_source.find("dependency manifest requires full parity gate") !=
        std::string::npos);
}

TEST_CASE("paritychecker sources do not bridge into actor internals") {
  const auto paritychecker_dir = repo_root_dir() / "tools" / "paritychecker";
  REQUIRE(!file_exists(paritychecker_dir / ("generation_internal_" "diagnostics.hpp")));

  const std::vector<std::string> forbidden_patterns = {
    "#include \"emel/text/generator/" "detail.hpp\"",
    "#include \"emel/text/generator/" "actions.hpp\"",
    "#include \"emel/text/generator/" "guards.hpp\"",
    "#include \"emel/text/generator/prefill/" "guards.hpp\"",
    "emel::text::generator::" "detail::",
    "namespace emel::text::generator::" "detail",
    "emel::text::generator::" "action::",
    "emel::text::generator::" "guard::",
    "emel::text::generator::prefill::" "guard::",
    "#include \"emel/text/detokenizer/" "actions.hpp\"",
    "#include \"emel/text/jinja/parser/" "detail.hpp\"",
    "emel::text::detokenizer::" "action::detail::",
    "emel::text::jinja::parser::" "detail::",
    "->generation_",
    "generation_internal_" "diagnostics.hpp",
    "emel::gguf::loader::" "detail::",
    "emel::model::" "detail::",
    "emel::model::llama::" "detail::",
    "emel::model::loader::" "detail::",
  };

  for (const auto & entry : std::filesystem::directory_iterator(paritychecker_dir)) {
    const auto path = entry.path();
    if (!entry.is_regular_file() || path.filename() == "paritychecker_tests.cpp") {
      continue;
    }
    if (path.extension() != ".cpp" && path.extension() != ".hpp") {
      continue;
    }
    const std::string source = read_text_file(path);
    REQUIRE_FALSE(source.empty());
    for (const auto & pattern : forbidden_patterns) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    path.filename().string() << " contains forbidden pattern " << pattern);
    }

    size_t line_begin = 0u;
    while (line_begin <= source.size()) {
      size_t line_end = source.find('\n', line_begin);
      if (line_end == std::string::npos) {
        line_end = source.size();
      }
      const std::string line = source.substr(line_begin, line_end - line_begin);
      if (line.find("#include \"emel/") != std::string::npos) {
        const bool actor_internal_include =
            line.find("/actions.hpp\"") != std::string::npos ||
            line.find("/guards.hpp\"") != std::string::npos ||
            line.find("/detail.hpp\"") != std::string::npos;
        const bool approved_kernel_surface =
            line.find("#include \"emel/kernel/detail.hpp\"") != std::string::npos ||
            line.find("#include \"emel/kernel/aarch64/detail.hpp\"") != std::string::npos ||
            line.find("#include \"emel/kernel/x86_64/detail.hpp\"") != std::string::npos;
        const bool include_allowed = !actor_internal_include || approved_kernel_surface;
        CHECK_MESSAGE(include_allowed,
                      path.filename().string() << " contains actor-internal include " << line);
      }
      if (line_end == source.size()) {
        break;
      }
      line_begin = line_end + 1u;
    }
  }
}

TEST_CASE("parity shared runner sources stay free of lane-owned runtime objects") {
  const auto paritychecker_dir = repo_root_dir() / "tools" / "paritychecker";
  const std::vector<std::filesystem::path> shared_runner_sources = {
      paritychecker_dir / "parity_runner.cpp",
      paritychecker_dir / "parity_engine.cpp",
      paritychecker_dir / "parity_assets.cpp",
      paritychecker_dir / "parity_dependency_manifest.cpp",
  };
  const std::vector<std::string> forbidden = {
    "#include \"llama",
    "#include \"ggml",
    "llama_",
    "ggml_",
    "reference_backend",
    "llama_model",
    "llama_context",
    "llama_vocab",
    "emel::text::generator::action::",
    "emel::text::generator::guard::",
    "emel::text::generator::detail::",
    "emel::model::data",
  };

  for (const auto & path : shared_runner_sources) {
    const std::string source = read_text_file(path);
    REQUIRE_FALSE(source.empty());
    for (const auto & pattern : forbidden) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    path.filename().string() << " contains shared-lane pattern " << pattern);
    }
  }
}

TEST_CASE("parity engine source keeps EMEL and reference lane objects separate") {
  const std::string source =
      read_text_file(repo_root_dir() / "tools" / "paritychecker" / "parity_engines.cpp");
  REQUIRE_FALSE(source.empty());

  CHECK(source.find("const llama_vocab *llama_vocab_ptr") != std::string::npos);
  CHECK(source.find("auto emel_vocab = std::make_unique<emel::model::data::vocab>()") !=
        std::string::npos);
  CHECK(source.find("load_emel_vocab_from_gguf_file(opts.model_path, *emel_vocab)") !=
        std::string::npos);
  CHECK(source.find("struct reference_backend") != std::string::npos);
  CHECK(source.find("llama_model_ptr model") != std::string::npos);
  CHECK(source.find("std::unique_ptr<emel::model::data> model_data") != std::string::npos);
  CHECK(source.find("reference_backend reference") != std::string::npos);
  CHECK(source.find("load_generation_reference_backend(opts.model_path, state)") !=
        std::string::npos);
  CHECK(source.find("state.model_loader.process_event(request)") != std::string::npos);
  CHECK(source.find("run_emel_generate(state,") != std::string::npos);

  const size_t harness_pos = source.find("int run_generation_harness_contract(");
  REQUIRE(harness_pos != std::string::npos);
  const size_t live_reference_pos =
      source.find("run_reference_generate(state.reference", harness_pos);
  const size_t parity_compare_pos =
      source.find("generation_results_match(emel_result, reference_result)", harness_pos);
  const size_t baseline_load_pos =
      source.find("load_generation_baseline_file(baseline_path, baseline_record)", harness_pos);
  CHECK(live_reference_pos != std::string::npos);
  CHECK(parity_compare_pos != std::string::npos);
  CHECK(baseline_load_pos != std::string::npos);
  if (live_reference_pos != std::string::npos &&
      parity_compare_pos != std::string::npos &&
      baseline_load_pos != std::string::npos) {
    CHECK(live_reference_pos < parity_compare_pos);
    CHECK(parity_compare_pos < baseline_load_pos);
  }

  const std::vector<std::string> forbidden = {
    "emel_vocab = llama_vocab",
    "llama_vocab_ptr = emel_vocab",
    "state.model_data = state.reference",
    "state.reference.model = state.model_data",
    "state.reference.vocab = &state.model_data",
    "state.reference.vocab = state.model_data",
    "reference_result = emel_result",
    "emel_result = reference_result",
    "generation_result & reference_result = baseline_record.result",
    "parse_plamo2_byte_token(piece_view",
  };

  for (const auto & pattern : forbidden) {
    CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                  "parity_engines.cpp contains lane-sharing pattern " << pattern);
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

  std::string supported_gemma4_template = {};
  for (const std::string_view marker :
       emel::tools::generation_formatter_contract::k_supported_gemma4_primary_template_markers) {
    supported_gemma4_template.append(marker);
    supported_gemma4_template.push_back('\n');
  }

  const auto supported_gemma4 =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          supported_gemma4_template, 0u);
  CHECK(emel::tools::generation_formatter_contract::binding_supported(supported_gemma4));
  CHECK(supported_gemma4.contract ==
        emel::tools::generation_formatter_contract::k_supported_gemma4_contract);

  std::string formatted_gemma4_prompt = {};
  CHECK(emel::tools::generation_formatter_contract::format_single_user_prompt(
      supported_gemma4, "hello", formatted_gemma4_prompt));
  CHECK(formatted_gemma4_prompt ==
        "<bos><|turn>user\nhello<turn|>\n<|turn>model\n");

  const auto unsupported =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          "{{ unsupported }}", 0u);
  CHECK_FALSE(emel::tools::generation_formatter_contract::binding_supported(unsupported));
  CHECK(unsupported.contract ==
        emel::tools::generation_formatter_contract::k_unsupported_template_contract);

  const auto named_variant =
      emel::tools::generation_formatter_contract::resolve_primary_template_binding(
          supported_template, 1u);
  CHECK_FALSE(emel::tools::generation_formatter_contract::binding_supported(named_variant));
  CHECK(named_variant.contract ==
        emel::tools::generation_formatter_contract::k_unsupported_template_contract);
}

TEST_CASE("paritychecker generation keeps append-only maintained baselines for supported fixtures") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    INFO("fixture: " << fixture.name);
    for (const int32_t max_tokens : {1, 10, 100, 1000}) {
      const std::filesystem::path baseline_path =
          maintained_generation_baseline_path(fixture, max_tokens);
      INFO("baseline: " << baseline_path.string());
      CHECK(file_exists(baseline_path));
    }
  }
}

TEST_CASE("paritychecker matches current maintained generation publication against live reference") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    if (!fixture.current_publication) {
      continue;
    }
    const std::filesystem::path model_path = maintained_generation_fixture_path(fixture);
    INFO("fixture: " << fixture.name);
    if (!file_exists(model_path)) {
      INFO("skipping missing maintained fixture: " << model_path.string());
      continue;
    }

    const process_capture capture = run_generation_paritychecker_capture(model_path, "hello");

    CHECK(capture.exit_code == 0);
    CHECK(capture.stderr_text.empty());
    CHECK(capture.stdout_text.find("generation parity ok") != std::string::npos);
    CHECK(capture.stdout_text.find(std::string("fixture=") + std::string(fixture.name)) !=
          std::string::npos);
    CHECK(capture.stdout_text.find("formatter_contract=") != std::string::npos);
    CHECK(capture.stdout_text.find("reference_impl:") != std::string::npos);
    CHECK(capture.stdout_text.find("generation_baseline:") != std::string::npos);
    CHECK(capture.stdout_text.find("reference_impl: source=maintained_generation_baseline") ==
          std::string::npos);
    CHECK(parse_named_metric_on_line(
              capture.stdout_text, "reference_decode_seams:", "reference_decode_calls") > 0);
  }
}

TEST_CASE("paritychecker reports legacy maintained generation live-reference drift") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    if (fixture.current_publication) {
      continue;
    }
    const std::filesystem::path model_path = maintained_generation_fixture_path(fixture);
    INFO("fixture: " << fixture.name);
    if (!file_exists(model_path)) {
      INFO("skipping missing maintained fixture: " << model_path.string());
      continue;
    }

    const process_capture capture = run_generation_paritychecker_capture(model_path, "hello");

    CHECK(capture.exit_code == 1);
    CHECK(capture.stderr_text.find("generation parity mismatch") != std::string::npos);
    CHECK(capture.stdout_text.find("reference_impl:") != std::string::npos);
    CHECK(capture.stdout_text.find("generation_baseline:") == std::string::npos);
    CHECK(capture.stdout_text.find("reference_impl: source=maintained_generation_baseline") ==
          std::string::npos);
    CHECK(parse_named_metric_on_line(
              capture.stdout_text, "reference_decode_seams:", "reference_decode_calls") > 0);
  }
}

TEST_CASE("paritychecker maintained generation fixtures reject same-basename files outside tests/models") {
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
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
