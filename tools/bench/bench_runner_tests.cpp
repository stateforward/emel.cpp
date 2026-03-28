#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

std::filesystem::path repo_root() {
#ifdef BENCH_REPO_ROOT
  return BENCH_REPO_ROOT;
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path bench_runner_binary_path() {
#ifdef BENCH_RUNNER_BINARY_PATH
  return BENCH_RUNNER_BINARY_PATH;
#else
  return std::filesystem::path("bench_runner");
#endif
}

std::filesystem::path models_dir() {
  return repo_root() / "tests" / "models";
}

bool file_exists(const std::filesystem::path & path) {
  std::FILE * file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

std::string read_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

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

struct process_capture {
  int exit_code = -1;
  std::string stdout_text = {};
  std::string stderr_text = {};
};

process_capture run_generation_bench_compare_capture() {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_BENCH_SUITE=generation && ";
  command += "set EMEL_BENCH_ITERS=1 && ";
  command += "set EMEL_BENCH_RUNS=1 && ";
  command += "set EMEL_BENCH_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_WARMUP_RUNS=0 && ";
  command += "set EMEL_BENCH_GENERATION_ITERS=1 && ";
  command += "set EMEL_BENCH_GENERATION_RUNS=1 && ";
  command += "set EMEL_BENCH_GENERATION_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_GENERATION_WARMUP_RUNS=0 && ";
  command += quote_arg_windows(bench_runner_binary_path().string());
  command += " --mode=compare > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += "EMEL_BENCH_SUITE=generation ";
  command += "EMEL_BENCH_ITERS=1 ";
  command += "EMEL_BENCH_RUNS=1 ";
  command += "EMEL_BENCH_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_WARMUP_RUNS=0 ";
  command += "EMEL_BENCH_GENERATION_ITERS=1 ";
  command += "EMEL_BENCH_GENERATION_RUNS=1 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_RUNS=0 ";
  command += quote_arg_posix(bench_runner_binary_path().string());
  command += " --mode=compare > ";
  command += quote_arg_posix(stdout_path.string());
  command += " 2> ";
  command += quote_arg_posix(stderr_path.string());
#endif

  const int status = std::system(command.c_str());
  process_capture capture{};
  capture.stdout_text = read_file(stdout_path);
  capture.stderr_text = read_file(stderr_path);

  std::error_code ec;
  std::filesystem::remove(stdout_path, ec);
  std::filesystem::remove(stderr_path, ec);

  if (status == -1) {
    return capture;
  }
#if defined(_WIN32)
  capture.exit_code = status;
#else
  if (!WIFEXITED(status)) {
    return capture;
  }
  capture.exit_code = WEXITSTATUS(status);
#endif
  return capture;
}

std::uint64_t parse_named_metric(const std::string & haystack, const std::string & name) {
  const std::string needle = name + "=";
  const size_t pos = haystack.find(needle);
  if (pos == std::string::npos) {
    return 0u;
  }

  size_t cursor = pos + needle.size();
  std::uint64_t value = 0u;
  while (cursor < haystack.size() && haystack[cursor] >= '0' && haystack[cursor] <= '9') {
    value = value * 10u + static_cast<std::uint64_t>(haystack[cursor] - '0');
    ++cursor;
  }
  return value;
}

}  // namespace

TEST_CASE("bench_runner qwen3 generation compare stays on the maintained canonical path") {
  const auto model_path = models_dir() / "Qwen3-0.6B-Q8_0.gguf";
  REQUIRE(file_exists(model_path));

  const process_capture capture = run_generation_bench_compare_capture();
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("llama_68m_prompt_hello_max_tokens_1") == std::string::npos);
  CHECK(capture.stdout_text.find("# benchmark_config: iterations=1 runs=1 "
                                 "warmup_iterations=0 warmup_runs=1 "
                                 "generation_iterations=1 generation_runs=1 "
                                 "generation_warmup_iterations=0 generation_warmup_runs=0") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("# generation_formatter_contract: source=tokenizer.chat_template "
                                 "support=supported_contract "
                                 "shape=structured_chat_messages_v1 tools=none "
                                 "add_generation_prompt=true enable_thinking=false") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("# generation_runtime_contract: case=generation/preloaded_request/"
                                 "qwen3_0_6b_q8_0_prompt_hello_max_tokens_1") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("native_quantized=8") != std::string::npos);
  CHECK(capture.stdout_text.find("approved_dense_f32_by_contract=6") != std::string::npos);
  CHECK(capture.stdout_text.find("# generation_quantized_evidence: case=generation/preloaded_request/"
                                 "qwen3_0_6b_q8_0_prompt_hello_max_tokens_1") !=
        std::string::npos);
  CHECK(parse_named_metric(capture.stdout_text, "native_q8_0_dispatch_calls") > 0u);
}
