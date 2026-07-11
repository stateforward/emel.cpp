#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

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

std::filesystem::path compare_script_path() {
#ifdef PERSONAPLEX_COMPARE_SCRIPT_PATH
  return PERSONAPLEX_COMPARE_SCRIPT_PATH;
#else
  return repo_root() / "tools" / "bench" / "personaplex_compare.py";
#endif
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::string quote_arg_posix(const std::string &arg) {
  std::string output = "'";
  for (const char value : arg) {
    if (value == '\'') {
      output += "'\\''";
    } else {
      output.push_back(value);
    }
  }
  output += "'";
  return output;
}

int command_exit_code(const std::string &command) {
  const int status = std::system(command.c_str());
  if (status == -1) {
    return -1;
  }
#if defined(_WIN32)
  return status;
#else
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

} // namespace

TEST_CASE("personaplex compare math reports exact and partial token behavior") {
  const std::string module = quote_arg_posix(compare_script_path().string());
  const std::string program =
      "import importlib.util; "
      "spec=importlib.util.spec_from_file_location('personaplex_compare', " +
      module +
      "); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
      "assert m.match_fraction([[1,2],[3,4]], [[1,2],[3,4]], 2) == 1.0; "
      "assert m.match_fraction([[1,9]], [[1,2]], 2) == 0.5; "
      "assert m.match_fraction([[1]], [[1,2]], 2) == 0.0; "
      "assert m.common_token_prefix([[1,2]], [[1,3]]) == 1; "
      "assert abs(m.correlation([1.0,2.0,3.0], [2.0,4.0,6.0])-1.0) < 1e-12";
  CHECK(command_exit_code("python3 -c " + quote_arg_posix(program)) == 0);
}

TEST_CASE("personaplex compare keeps CPU and lane isolation explicit") {
  const std::string compare_source = read_file(compare_script_path());
  const std::string wrapper_source =
      read_file(repo_root() / "scripts" / "bench_personaplex_compare.sh");
  const std::string cmake_source =
      read_file(repo_root() / "tools" / "bench" / "CMakeLists.txt");
  const std::string emel_source =
      read_file(repo_root() / "tools" / "bench" / "speech" /
                "personaplex_emel_runner.cpp");
  const std::string setup_source =
      read_file(repo_root() / "scripts" / "setup_moshi_cpp_reference.sh");
  const std::string converter_source =
      read_file(repo_root() / "tools" / "bench" / "moshi_gguf_convert.py");
  const std::string inference_source =
      read_file(repo_root() / "tools" / "bench" / "personaplex-inference.json");
  const std::string mimi_wrapper_source =
      read_file(repo_root() / "scripts" / "bench_mimi_compare.sh");
  const std::string reference_source =
      read_file(repo_root() / "tools" / "bench" / "speech" /
                "moshi_reference_driver.cpp");

  const std::size_t first_public_match =
      compare_source.find("public_codebook_match_fraction");
  REQUIRE(first_public_match != std::string::npos);
  CHECK(compare_source.find("public_codebook_match_fraction",
                            first_public_match + 1u) != std::string::npos);
  CHECK(compare_source.find("best_log_energy_correlation") !=
        std::string::npos);
  CHECK(compare_source.find("reference_over_emel") != std::string::npos);
  CHECK(wrapper_source.find("--threads \"$THREADS\"") != std::string::npos);
  CHECK(wrapper_source.find("--audio \"$AUDIO\"") != std::string::npos);
  CHECK(wrapper_source.find("--inference-config \"$INFERENCE_CONFIG\"") !=
        std::string::npos);
  CHECK(wrapper_source.find("--emel-lm PATH") != std::string::npos);
  CHECK(wrapper_source.find("--emel-lm) EMEL_LM=") != std::string::npos);
  CHECK(wrapper_source.find("--emel-mimi PATH") != std::string::npos);
  CHECK(wrapper_source.find("--emel-mimi) EMEL_MIMI=") != std::string::npos);
  CHECK(wrapper_source.find("--data-format=LEI16@24000") != std::string::npos);
  CHECK(wrapper_source.find("mimi-e351c8d8-125-personaplex-emel.gguf") !=
        std::string::npos);
  CHECK(wrapper_source.find("Hey, I'm Gabe. How are you doing?") !=
        std::string::npos);
  CHECK(cmake_source.find("set(GGML_METAL OFF CACHE BOOL \"CPU-only "
                          "PersonaPlex reference lane\" FORCE)") !=
        std::string::npos);
  CHECK(emel_source.find("#include <ggml") == std::string::npos);
  CHECK(emel_source.find(
            "predictor_initialize.sampling_seed = config.sampling_seed") !=
        std::string::npos);
  CHECK(emel_source.find("flush_steps == 0") != std::string::npos);
  CHECK(emel_source.find("sampling_seed = 1234") == std::string::npos);
  CHECK(emel_source.find(".max_blocks = 256") == std::string::npos);
  CHECK(compare_source.find("\"--n-q\", \"8\"") == std::string::npos);
  CHECK(compare_source.find("--audio-top-k") == std::string::npos);
  CHECK(compare_source.find("--text-top-k") == std::string::npos);
  CHECK(compare_source.find("\"emel_lm\": {") != std::string::npos);
  CHECK(compare_source.find("\"reference_lm\": {") != std::string::npos);
  CHECK(compare_source.find("\"emel_mimi\": {") != std::string::npos);
  CHECK(compare_source.find("\"reference_mimi\": {") != std::string::npos);
  CHECK(setup_source.find("--inference-config \"$MOSHI_INFERENCE_CONFIG\"") !=
        std::string::npos);
  CHECK(converter_source.find("--inference-config") != std::string::npos);
  CHECK(inference_source.find("\"prompt_tokens\"") != std::string::npos);
  CHECK(inference_source.find("\"audio_top_k\"") != std::string::npos);
  CHECK(inference_source.find("\"text_top_k\"") != std::string::npos);
  const std::size_t personaplex_build_guard =
      wrapper_source.find("if ! $RUN_ONLY; then");
  REQUIRE(personaplex_build_guard != std::string::npos);
  CHECK(wrapper_source.find("for tool in cmake ninja git zig",
                            personaplex_build_guard) != std::string::npos);
  const std::size_t mimi_build_guard =
      mimi_wrapper_source.find("if ! $RUN_ONLY; then");
  REQUIRE(mimi_build_guard != std::string::npos);
  CHECK(mimi_wrapper_source.find("for tool in cmake ninja git",
                                 mimi_build_guard) != std::string::npos);
  CHECK(mimi_wrapper_source.find("if [[ ! -f \"$MODEL_CONFIG\" ]]",
                                 mimi_build_guard) != std::string::npos);
  CHECK(reference_source.find("produced=%d") != std::string::npos);
}

TEST_CASE("personaplex conversion injects its inference contract") {
  const auto bench_dir = repo_root() / "tools" / "bench";
  const auto inference_path = bench_dir / "personaplex-inference.json";
  const std::string program =
      "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
      "import moshi_gguf_convert as m; config={}; "
      "m.inject_inference_config(config, Path(sys.argv[2])); "
      "assert config['inference']['dep_q'] == 8; "
      "assert len(config['inference']['prompt_tokens']) == 17";
  const std::string command = "PYTHONDONTWRITEBYTECODE=1 python3 -c " +
                              quote_arg_posix(program) + " " +
                              quote_arg_posix(bench_dir.string()) + " " +
                              quote_arg_posix(inference_path.string());
  CHECK(command_exit_code(command) == 0);
}

TEST_CASE(
    "personaplex runtime orchestration is owned by the generic generator") {
  const auto generator_machine =
      repo_root() / "src" / "emel" / "speech" / "generator" / "sm.hpp";
  const std::string emel_source =
      read_file(repo_root() / "tools" / "bench" / "speech" /
                "personaplex_emel_runner.cpp");

  CHECK(std::filesystem::exists(generator_machine));
  CHECK(emel_source.find("emel/speech/generator/any.hpp") != std::string::npos);
  CHECK(emel_source.find("speech/generator/moshi/personaplex/session") ==
        std::string::npos);
}
