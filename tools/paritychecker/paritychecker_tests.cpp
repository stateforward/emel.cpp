#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

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

std::filesystem::path parity_texts_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "text" / "tokenizer" / "parity_texts";
#else
  return std::filesystem::path("tests") / "text" / "tokenizer" / "parity_texts";
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

bool run_paritychecker_process(const std::string & model, const parity_case & test_case) {
  std::string command;
#if defined(_WIN32)
  command = ".\\paritychecker --model ";
  command += quote_arg_windows(model);
  command += " --text-file ";
  command += quote_arg_windows(test_case.text_path.string());
#else
  command = "ulimit -s 8192; ./paritychecker --model ";
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
  command = ".\\paritychecker --gbnf --text-file ";
  command += quote_arg_windows(grammar_path.string());
#else
  command = "ulimit -s 8192; ./paritychecker --gbnf --text-file ";
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
  command = ".\\paritychecker --kernel --text kernel";
#else
  command = "ulimit -s 8192; ./paritychecker --kernel --text kernel";
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
  command = ".\\paritychecker --jinja --text-file ";
  command += quote_arg_windows(template_path.string());
#else
  command = "ulimit -s 8192; ./paritychecker --jinja --text-file ";
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
  command = ".\\paritychecker";
  for (const auto & arg : args) {
    command += " ";
    command += quote_arg_windows(arg);
  }
  command += " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ./paritychecker";
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

TEST_CASE("paritychecker generation compares one bounded request against the reference path") {
  const auto model_path = models_dir() / "Llama-68M-Chat-v1-Q2_K.gguf";
  REQUIRE(file_exists(model_path));

  const process_capture capture = run_generation_paritychecker_capture(model_path, "hello", 1);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("generation parity ok") != std::string::npos);
  CHECK(capture.stdout_text.find("generated_tokens=1") != std::string::npos);
  CHECK(parse_named_metric(capture.stdout_text, "flash_dispatch_calls") > 0);
  CHECK(capture.stdout_text.find("reference_impl: source=") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_decode_seams:") != std::string::npos);
  CHECK(capture.stdout_text.find("flash_dispatch: calls=") != std::string::npos);
  CHECK(parse_flash_dispatch_calls(capture.stdout_text) > 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_flash_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_flash_dispatch_calls") >= 0);
  CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") >= 0);
  CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") >= 0);
  if (expected_generation_kernel_kind() == "aarch64") {
    CHECK(parse_named_metric(capture.stdout_text, "optimized_flash_dispatch_calls") > 0);
    CHECK(parse_named_metric(capture.stdout_text, "shared_flash_dispatch_calls") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") > 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  } else {
    CHECK(parse_named_metric(capture.stdout_text, "optimized_flash_dispatch_calls") == 0);
    CHECK(parse_named_metric(capture.stdout_text, "shared_flash_dispatch_calls") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  }
  CHECK(capture.stdout_text.find("generation initialize ok") == std::string::npos);
  CHECK(capture.stdout_text.find("emel generator path ready") == std::string::npos);
}

TEST_CASE("paritychecker generation keeps parity on a bounded longer decode") {
  const auto model_path = models_dir() / "Llama-68M-Chat-v1-Q2_K.gguf";
  REQUIRE(file_exists(model_path));

  const process_capture capture = run_generation_paritychecker_capture(model_path, "hello", 8);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("generation parity ok") != std::string::npos);
  CHECK(capture.stdout_text.find("max_tokens=8") != std::string::npos);
  CHECK(parse_named_metric(capture.stdout_text, "generated_tokens") > 1);
  CHECK(parse_named_metric(capture.stdout_text, "flash_dispatch_calls") > 0);
  CHECK(parse_named_metric(capture.stdout_text, "optimized_flash_dispatch_calls") >= 0);
  CHECK(parse_named_metric(capture.stdout_text, "shared_flash_dispatch_calls") >= 0);
  CHECK(capture.stdout_text.find("reference_impl: source=") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_decode_seams:") != std::string::npos);
  CHECK(capture.stdout_text.find("flash_dispatch: calls=") != std::string::npos);
  CHECK(parse_flash_dispatch_calls(capture.stdout_text) > 0);
  if (expected_generation_kernel_kind() == "aarch64") {
    CHECK(parse_named_metric(capture.stdout_text, "optimized_flash_dispatch_calls") > 0);
    CHECK(parse_named_metric(capture.stdout_text, "shared_flash_dispatch_calls") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") > 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  } else {
    CHECK(parse_named_metric(capture.stdout_text, "optimized_flash_dispatch_calls") == 0);
    CHECK(parse_named_metric(capture.stdout_text, "shared_flash_dispatch_calls") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  }
}

TEST_CASE("paritychecker generation dump proves the EMEL path avoids the reference decode seam") {
  const auto model_path = models_dir() / "Llama-68M-Chat-v1-Q2_K.gguf";
  REQUIRE(file_exists(model_path));

  const process_capture capture = run_generation_paritychecker_capture_with_args({
    "--generation",
    "--model",
    model_path.string(),
    "--text",
    "hello",
    "--max-tokens",
    "1",
    "--dump",
  });

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("reference_decode_seams:") != std::string::npos);
  CHECK(parse_named_metric(capture.stdout_text, "emel_decode_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "emel_logits_calls") == 0);
  CHECK(parse_named_metric(capture.stdout_text, "reference_decode_calls") > 0);
  CHECK(parse_named_metric(capture.stdout_text, "reference_logits_calls") > 0);
  CHECK(capture.stdout_text.find("kernel_dispatch: kind=") != std::string::npos);
  CHECK(capture.stdout_text.find("flash_dispatch: calls=") != std::string::npos);
  CHECK(capture.stdout_text.find(std::string("kernel_dispatch: kind=") +
                                 std::string(expected_generation_kernel_kind())) !=
        std::string::npos);
  CHECK(parse_kernel_dispatch_calls(capture.stdout_text) > 0);
  CHECK(parse_flash_dispatch_calls(capture.stdout_text) > 0);
  CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") >= 0);
  CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") >= 0);
  if (expected_generation_kernel_kind() == "aarch64") {
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") > 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  } else {
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "optimized") == 0);
    CHECK(parse_flash_dispatch_metric(capture.stdout_text, "shared") == 0);
  }
}

TEST_CASE("paritychecker help describes the canonical generation fixture contract") {
  const process_capture capture = run_generation_paritychecker_capture_with_args({"--help"});

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());
  CHECK(capture.stderr_text.find("--generation mode requires --model tests/models/"
                                 "Llama-68M-Chat-v1-Q2_K.gguf") != std::string::npos);
  CHECK(capture.stderr_text.find("reserves the generation CLI contract") == std::string::npos);
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

TEST_CASE("paritychecker generation rejects a same-basename fixture outside tests/models") {
  const auto canonical_model_path = models_dir() / "Llama-68M-Chat-v1-Q2_K.gguf";
  REQUIRE(file_exists(canonical_model_path));

  const std::filesystem::path impostor_model_path =
      make_temp_fixture_path("paritychecker-fixture", canonical_model_path.filename().string());
  std::filesystem::copy_file(canonical_model_path,
                             impostor_model_path,
                             std::filesystem::copy_options::overwrite_existing);

  const process_capture capture = run_generation_paritychecker_capture(impostor_model_path, "hello");

  CHECK(capture.exit_code == 1);
  CHECK(capture.stdout_text.find("generation parity ok") == std::string::npos);
  CHECK(capture.stderr_text.find("generation requires canonical fixture") != std::string::npos);

  std::filesystem::remove(impostor_model_path);
  std::filesystem::remove(impostor_model_path.parent_path());
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
