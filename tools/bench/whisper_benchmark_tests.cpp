#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <sys/stat.h>
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

std::filesystem::path whisper_benchmark_script_path() {
#ifdef WHISPER_BENCHMARK_SCRIPT_PATH
  return WHISPER_BENCHMARK_SCRIPT_PATH;
#else
  return repo_root() / "tools" / "bench" / "whisper_benchmark.py";
#endif
}

std::filesystem::path whisper_compare_script_path() {
#ifdef WHISPER_COMPARE_SCRIPT_PATH
  return WHISPER_COMPARE_SCRIPT_PATH;
#else
  return repo_root() / "tools" / "bench" / "whisper_compare.py";
#endif
}

std::filesystem::path whisper_emel_parity_runner_source_path() {
#ifdef WHISPER_EMEL_PARITY_RUNNER_SOURCE_PATH
  return WHISPER_EMEL_PARITY_RUNNER_SOURCE_PATH;
#else
  return repo_root() / "tools" / "bench" / "whisper_emel_parity_runner.cpp";
#endif
}

std::filesystem::path whisper_single_thread_script_path() {
  return repo_root() / "scripts" / "bench_whisper_single_thread.sh";
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void write_text_file(const std::filesystem::path &path,
                     const std::string &text) {
  std::ofstream output(path, std::ios::binary);
  REQUIRE(output.good());
  output << text;
  REQUIRE(output.good());
}

std::string quote_arg_posix(const std::string &arg) {
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

struct process_capture {
  int exit_code = -1;
  std::string stdout_text = {};
  std::string stderr_text = {};
};

#if !defined(_WIN32)
void make_executable(const std::filesystem::path &path) {
  REQUIRE(::chmod(path.c_str(), 0755) == 0);
}

process_capture run_command_capture(const std::string &command,
                                    const std::filesystem::path &stdout_path,
                                    const std::filesystem::path &stderr_path) {
  process_capture capture = {};
  const int status = std::system(command.c_str());
  capture.stdout_text = read_file(stdout_path);
  capture.stderr_text = read_file(stderr_path);
  if (status == -1) {
    return capture;
  }
  capture.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  return capture;
}
#endif

std::filesystem::path prepare_fake_runner(const std::filesystem::path &dir) {
  const auto path = dir / "fake_emel_runner.sh";
  write_text_file(
      path, "#!/bin/sh\n"
            "out=''\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  case \"$1\" in\n"
            "    --output-dir) out=\"$2\"; shift 2 ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            "if [ -n \"$EMEL_FAIL_WARMUP\" ] && echo \"$out\" | grep -q "
            "'iter_-'; then\n"
            "  exit 7\n"
            "fi\n"
            "transcript=\"$EMEL_FAKE_TRANSCRIPT\"\n"
            "if [ -n \"$EMEL_FAKE_TRANSCRIPT_ITER0\" ] && echo \"$out\" | "
            "grep -q 'iter_0'; then\n"
            "  transcript=\"$EMEL_FAKE_TRANSCRIPT_ITER0\"\n"
            "fi\n"
            "if [ -n \"$EMEL_FAKE_SLEEP\" ]; then\n"
            "  sleep \"$EMEL_FAKE_SLEEP\"\n"
            "fi\n"
            "mkdir -p \"$out\"\n"
            "printf '%s' \"$transcript\" > \"$out/transcript.txt\"\n"
            "printf '{\"schema\":\"whisper_compare/v1\","
            "\"record_type\":\"result\",\"status\":\"ok\","
            "\"transcript\":\"%s\",\"wall_time_ns\":1}\\n' "
            "\"$transcript\"\n");
#if !defined(_WIN32)
  make_executable(path);
#endif
  return path;
}

std::filesystem::path prepare_fake_reference(const std::filesystem::path &dir) {
  const auto path = dir / "fake_reference.sh";
  write_text_file(path,
                  "#!/bin/sh\n"
                  "if [ -n \"$REF_ARGS_FILE\" ]; then\n"
                  "  printf '%s\\n' \"$@\" > \"$REF_ARGS_FILE\"\n"
                  "fi\n"
                  "out=''\n"
                  "while [ \"$#\" -gt 0 ]; do\n"
                  "  case \"$1\" in\n"
                  "    --output-file) out=\"$2\"; shift 2 ;;\n"
                  "    *) shift ;;\n"
                  "  esac\n"
                  "done\n"
                  "mkdir -p \"$(dirname \"$out\")\"\n"
                  "if [ -n \"$REF_FAIL_WARMUP\" ] && echo \"$out\" | grep "
                  "-q 'iter_-'; then\n"
                  "  exit 9\n"
                  "fi\n"
                  "if [ -n \"$REF_SKIP_TRANSCRIPT\" ]; then\n"
                  "  exit 0\n"
                  "fi\n"
                  "transcript=\"$REF_FAKE_TRANSCRIPT\"\n"
                  "if [ -n \"$REF_FAKE_TRANSCRIPT_ITER0\" ] && echo \"$out\" "
                  "| grep -q 'iter_0'; then\n"
                  "  transcript=\"$REF_FAKE_TRANSCRIPT_ITER0\"\n"
                  "fi\n"
                  "if [ -n \"$REF_FAKE_SLEEP\" ]; then\n"
                  "  sleep \"$REF_FAKE_SLEEP\"\n"
                  "fi\n"
                  "printf '%s' \"$transcript\" > \"$out.txt\"\n");
#if !defined(_WIN32)
  make_executable(path);
#endif
  return path;
}

process_capture run_whisper_benchmark(const std::filesystem::path &tmp_dir,
                                      const std::filesystem::path &emel_model,
                                      const std::filesystem::path &ref_model,
                                      const std::string &emel_transcript,
                                      const std::string &ref_transcript,
                                      const std::string &extra_env = {},
                                      const int warmups = 0,
                                      const int iterations = 1) {
  const auto stdout_path = tmp_dir / "stdout.txt";
  const auto stderr_path = tmp_dir / "stderr.txt";
  const auto fake_emel = prepare_fake_runner(tmp_dir);
  const auto fake_ref = prepare_fake_reference(tmp_dir);
  const auto output_dir = tmp_dir / "out";
  const auto audio = tmp_dir / "audio.wav";
  const auto ref_args = tmp_dir / "reference_args.txt";
  write_text_file(audio, "audio");

  const std::string command =
      extra_env + (extra_env.empty() ? "" : " ") +
      "EMEL_FAKE_TRANSCRIPT=" + quote_arg_posix(emel_transcript) + " " +
      "REF_FAKE_TRANSCRIPT=" + quote_arg_posix(ref_transcript) + " " +
      "REF_ARGS_FILE=" + quote_arg_posix(ref_args.string()) + " python3 " +
      quote_arg_posix(whisper_benchmark_script_path().string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " --emel-runner " + quote_arg_posix(fake_emel.string()) +
      " --emel-model " + quote_arg_posix(emel_model.string()) +
      " --tokenizer " +
      quote_arg_posix(
          (repo_root() / "tests" / "models" / "tokenizer-tiny.json").string()) +
      " --reference-cli " + quote_arg_posix(fake_ref.string()) +
      " --reference-model " + quote_arg_posix(ref_model.string()) +
      " --audio " + quote_arg_posix(audio.string()) + " --warmups " +
      std::to_string(warmups) + " --iterations " + std::to_string(iterations) +
      " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  return run_command_capture(command, stdout_path, stderr_path);
}

process_capture run_whisper_benchmark_defaults(
    const std::filesystem::path &tmp_dir,
    const std::filesystem::path &emel_model,
    const std::filesystem::path &ref_model,
    const std::string &emel_transcript,
    const std::string &ref_transcript,
    const std::string &extra_env = {}) {
  const auto stdout_path = tmp_dir / "stdout.txt";
  const auto stderr_path = tmp_dir / "stderr.txt";
  const auto fake_emel = prepare_fake_runner(tmp_dir);
  const auto fake_ref = prepare_fake_reference(tmp_dir);
  const auto output_dir = tmp_dir / "out";
  const auto audio = tmp_dir / "audio.wav";
  const auto ref_args = tmp_dir / "reference_args.txt";
  write_text_file(audio, "audio");

  const std::string command =
      extra_env + (extra_env.empty() ? "" : " ") +
      "EMEL_FAKE_TRANSCRIPT=" + quote_arg_posix(emel_transcript) + " " +
      "REF_FAKE_TRANSCRIPT=" + quote_arg_posix(ref_transcript) + " " +
      "REF_ARGS_FILE=" + quote_arg_posix(ref_args.string()) + " python3 " +
      quote_arg_posix(whisper_benchmark_script_path().string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " --emel-runner " + quote_arg_posix(fake_emel.string()) +
      " --emel-model " + quote_arg_posix(emel_model.string()) +
      " --tokenizer " +
      quote_arg_posix(
          (repo_root() / "tests" / "models" / "tokenizer-tiny.json").string()) +
      " --reference-cli " + quote_arg_posix(fake_ref.string()) +
      " --reference-model " + quote_arg_posix(ref_model.string()) +
      " --audio " + quote_arg_posix(audio.string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  return run_command_capture(command, stdout_path, stderr_path);
}

process_capture run_whisper_compare(const std::filesystem::path &tmp_dir,
                                    const std::filesystem::path &emel_model,
                                    const std::filesystem::path &ref_model,
                                    const std::string &emel_transcript,
                                    const std::string &ref_transcript) {
  const auto stdout_path = tmp_dir / "stdout.txt";
  const auto stderr_path = tmp_dir / "stderr.txt";
  const auto fake_emel = prepare_fake_runner(tmp_dir);
  const auto fake_ref = prepare_fake_reference(tmp_dir);
  const auto output_dir = tmp_dir / "out";
  const auto audio = tmp_dir / "audio.wav";
  write_text_file(audio, "audio");

  const std::string command =
      "EMEL_FAKE_TRANSCRIPT=" + quote_arg_posix(emel_transcript) + " " +
      "REF_FAKE_TRANSCRIPT=" + quote_arg_posix(ref_transcript) + " python3 " +
      quote_arg_posix(whisper_compare_script_path().string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " --emel-runner " + quote_arg_posix(fake_emel.string()) +
      " --emel-model " + quote_arg_posix(emel_model.string()) +
      " --tokenizer " +
      quote_arg_posix(
          (repo_root() / "tests" / "models" / "tokenizer-tiny.json").string()) +
      " --reference-cli " + quote_arg_posix(fake_ref.string()) +
      " --reference-model " + quote_arg_posix(ref_model.string()) +
      " --audio " + quote_arg_posix(audio.string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  return run_command_capture(command, stdout_path, stderr_path);
}

} // namespace

TEST_CASE("whisper emel parity runner stays on public runtime surfaces") {
  const std::string source = read_file(whisper_emel_parity_runner_source_path());
  REQUIRE(!source.empty());

  CHECK(source.find("emel/model/whisper/detail.hpp") == std::string::npos);
  CHECK(source.find("emel/speech/encoder/whisper/detail.hpp") ==
        std::string::npos);
  CHECK(source.find("emel/speech/decoder/whisper/detail.hpp") ==
        std::string::npos);
  CHECK(source.find("emel/speech/tokenizer/whisper/detail.hpp") ==
        std::string::npos);
  CHECK(source.find("emel/model/whisper/any.hpp") != std::string::npos);
  CHECK(source.find("emel/speech/recognizer/sm.hpp") != std::string::npos);
  CHECK(source.find("emel/speech/recognizer_routes/whisper/any.hpp") !=
        std::string::npos);
  CHECK(source.find("emel/speech/encoder/whisper/sm.hpp") ==
        std::string::npos);
  CHECK(source.find("emel/speech/decoder/whisper/sm.hpp") ==
        std::string::npos);
  CHECK(source.find("emel::speech::encoder::whisper::sm") ==
        std::string::npos);
  CHECK(source.find("emel::speech::decoder::whisper::sm") ==
        std::string::npos);
  CHECK(source.find("decode_token_ids") == std::string::npos);
  CHECK(source.find("emel/speech/tokenizer/whisper/any.hpp") !=
        std::string::npos);
}

TEST_CASE("whisper emel parity runner escapes transcript JSON") {
  const std::string source = read_file(whisper_emel_parity_runner_source_path());
  REQUIRE(!source.empty());

  CHECK(source.find("json_escape_string") != std::string::npos);
  CHECK(source.find("json_escape_string(transcript_text)") !=
        std::string::npos);
  CHECK(source.find("transcript_json.c_str(), static_cast<uint64_t>") !=
        std::string::npos);
  CHECK(source.find("transcript_text.c_str(), static_cast<uint64_t>") ==
        std::string::npos);
}

TEST_CASE("whisper single-thread wrapper defaults to stable closeout sample") {
  const std::string source = read_file(whisper_single_thread_script_path());
  REQUIRE(!source.empty());

  CHECK(source.find("ITERATIONS=\"${EMEL_WHISPER_BENCH_ITERATIONS:-20}\"") !=
        std::string::npos);
  CHECK(source.find("--iterations \"$ITERATIONS\"") != std::string::npos);
}

TEST_CASE("whisper benchmark defaults to stable closeout sample") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "default-iterations";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture = run_whisper_benchmark_defaults(
      tmp_dir, model, model, "[C]", "[C]", "REF_FAKE_SLEEP=0.01");
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("benchmark_status=ok") != std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"iteration_count\": 20") != std::string::npos);
  CHECK(summary.find("\"performance_tolerance_ppm\": 20000") !=
        std::string::npos);
  CHECK(summary.find("\"warmup_count\": 1") != std::string::npos);
  CHECK(summary.find("\"iterations\": 20") != std::string::npos);
#endif
}

TEST_CASE("whisper compare hard-fails transcript mismatch") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper compare test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" /
                       "compare-transcript-mismatch";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture =
      run_whisper_compare(tmp_dir, model, model, "[C]", "[Bell]");
  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("status=bounded_drift") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("reason=transcript_mismatch") !=
        std::string::npos);
  const std::string summary = read_file(tmp_dir / "out" / "summary.json");
  CHECK(summary.find("\"comparison_status\": \"bounded_drift\"") !=
        std::string::npos);
  CHECK(summary.find("\"reason\": \"transcript_mismatch\"") !=
        std::string::npos);
#endif
}

TEST_CASE("whisper compare exact match succeeds") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper compare test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "compare-exact-match";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture = run_whisper_compare(tmp_dir, model, model, "[C]", "[C]");
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("status=exact_match") != std::string::npos);
  CHECK(capture.stdout_text.find("reason=ok") != std::string::npos);
  const std::string summary = read_file(tmp_dir / "out" / "summary.json");
  CHECK(summary.find("\"backend_id\": \"emel.speech.recognizer.whisper\"") !=
        std::string::npos);
  CHECK(summary.find("\"runtime_surface\": \"speech/recognizer+speech/"
                     "recognizer_routes/whisper\"") != std::string::npos);
#endif
}

TEST_CASE("whisper benchmark hard-fails model mismatch") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "model-mismatch";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto emel_model = tmp_dir / "emel.bin";
  const auto ref_model = tmp_dir / "ref.bin";
  write_text_file(emel_model, "emel-model");
  write_text_file(ref_model, "reference-model");

  const auto capture =
      run_whisper_benchmark(tmp_dir, emel_model, ref_model, "[C]", "[C]");
  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("reason=model_mismatch") != std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"reason\": \"model_mismatch\"") != std::string::npos);
#endif
}

TEST_CASE("whisper benchmark hard-fails transcript mismatch") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "transcript-mismatch";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture =
      run_whisper_benchmark(tmp_dir, model, model, "[C]", "[Bell]");
  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("reason=transcript_mismatch") !=
        std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"reason\": \"transcript_mismatch\"") !=
        std::string::npos);
#endif
}

TEST_CASE("whisper benchmark hard-fails any measured iteration mismatch") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "iteration-mismatch";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture = run_whisper_benchmark(
      tmp_dir, model, model, "[C]", "[C]",
      "EMEL_FAKE_TRANSCRIPT_ITER0='[C]' REF_FAKE_TRANSCRIPT_ITER0='[Bell]'", 0,
      2);
  CHECK(capture.exit_code == 1);
  CHECK(capture.stdout_text.find("reason=transcript_mismatch") !=
        std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"first_mismatch\"") != std::string::npos);
  CHECK(summary.find("\"iteration_index\": 0") != std::string::npos);
#endif
}

TEST_CASE("whisper benchmark hard-fails warmup lane errors") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "warmup-error";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture = run_whisper_benchmark(tmp_dir, model, model, "[C]",
                                             "[C]", "EMEL_FAIL_WARMUP=1", 1, 1);
  CHECK(capture.exit_code == 1);
  CHECK(capture.stdout_text.find("reason=lane_error") != std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"error_kind\": \"runner_failed\"") != std::string::npos);
#endif
}

TEST_CASE("whisper benchmark hard-fails missing reference transcript") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" /
                       "missing-reference-transcript";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture = run_whisper_benchmark(tmp_dir, model, model, "[C]",
                                             "[C]", "REF_SKIP_TRANSCRIPT=1");
  CHECK(capture.exit_code == 1);
  CHECK(capture.stdout_text.find("reason=lane_error") != std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"error_kind\": \"missing_transcript\"") !=
        std::string::npos);
#endif
}

TEST_CASE("whisper benchmark hard-fails slower emel mean") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "slower-emel";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture = run_whisper_benchmark(tmp_dir, model, model, "[C]",
                                             "[C]", "EMEL_FAKE_SLEEP=0.2");
  CHECK(capture.exit_code == 1);
  CHECK(capture.stdout_text.find("reason=performance_regression") !=
        std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"reason\": \"performance_regression\"") !=
        std::string::npos);
  CHECK(summary.find("\"performance_comparison\"") != std::string::npos);
#endif
}

TEST_CASE("whisper benchmark uses deterministic reference policy flags") {
#if defined(_WIN32)
  MESSAGE("skipping shell-backed Whisper benchmark test on Windows");
#else
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       "emel-whisper-benchmark-tests" / "reference-policy";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const auto model = tmp_dir / "model.bin";
  write_text_file(model, "same-model");

  const auto capture =
      run_whisper_benchmark(tmp_dir, model, model, "[C]", "[C]",
                            "REF_FAKE_SLEEP=0.2");
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string args = read_file(tmp_dir / "reference_args.txt");
  CHECK(args.find("--audio-ctx\n50\n") != std::string::npos);
  CHECK(args.find("--beam-size\n1\n") != std::string::npos);
  CHECK(args.find("--best-of\n1\n") != std::string::npos);
  CHECK(args.find("--no-fallback\n") != std::string::npos);
  const std::string summary =
      read_file(tmp_dir / "out" / "benchmark_summary.json");
  CHECK(summary.find("\"backend_id\": \"emel.speech.recognizer.whisper\"") !=
        std::string::npos);
  CHECK(summary.find("\"runtime_surface\": \"speech/recognizer+speech/"
                     "recognizer_routes/whisper\"") != std::string::npos);
#endif
}
