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
      "assert abs(m.correlation([1.0,2.0,3.0], [2.0,4.0,6.0])-1.0) < 1e-12; "
      "reasons=[]; m.append_text_parity_reason(reasons, [1,2], [1,3], 0.5); "
      "assert reasons == ['text token match is 0.500000, expected 1.0']; "
      "reasons=[]; m.append_text_parity_reason(reasons, [1], [1,2], 0.0); "
      "assert reasons == ['text frame count mismatch']; "
      "reasons=[]; m.append_text_parity_reason(reasons, [1,2], [1,2], 1.0); "
      "assert reasons == []; observations=[]; "
      "m.append_input_parity_observation(observations, 0.980769); "
      "assert observations == ['same-WAV input token match is 0.980769, "
      "tracked separately from generated-output parity']; "
      "assert m.semantic_parity_reasons(1.0, 1.0, 1.0) == []; "
      "assert m.semantic_parity_reasons(0.980769, 1.0, 1.0) == "
      "['same-WAV input token match is 0.980769, expected 1.0']; "
      "assert len(m.semantic_parity_reasons(0.5, 0.75, 0.25)) == 3";
  CHECK(command_exit_code("python3 -B -c " + quote_arg_posix(program)) == 0);
}

TEST_CASE(
    "personaplex compare parses the complete CPU thread budget truthfully") {
  const std::string module = quote_arg_posix(compare_script_path().string());
  const std::string program =
      "import importlib.util, pathlib, tempfile; "
      "spec=importlib.util.spec_from_file_location('personaplex_compare', " +
      module +
      "); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
      "p=pathlib.Path(tempfile.mkstemp()[1]); "
      "p.write_text('EMEL_THREADS requested_total=1 owner_threads=1 "
      "stage_workers=0 matmul_workers=0 matmul_lanes=1 stage_mode=serial "
      "matmul_mode=serial\\n'); "
      "assert m.parse_emel_threads(p) == (1,1,0,0,1,'serial','serial'); "
      "p.write_text('EMEL_THREADS requested_total=8 owner_threads=1 "
      "stage_workers=2 matmul_workers=3 matmul_lanes=4 stage_mode=parallel "
      "matmul_mode=parallel\\n'); "
      "contract=m.parse_emel_threads(p); "
      "assert contract == (8,1,2,3,4,'parallel','parallel'); "
      "assert m.thread_contract_reasons(contract, 8) == []; "
      "assert m.thread_contract_reasons((8,1,2,7,8,'parallel','parallel'), 8) "
      "== ['EMEL runnable concurrency exceeds total CPU budget']; "
      "p.unlink()";
  CHECK(command_exit_code("python3 -B -c " + quote_arg_posix(program)) == 0);
}

TEST_CASE("personaplex compare binds reports to source and runner identity") {
  const std::string module = quote_arg_posix(compare_script_path().string());
  const std::string program =
      "import importlib.util, pathlib, tempfile; "
      "spec=importlib.util.spec_from_file_location('personaplex_compare', " +
      module +
      "); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
      "root=pathlib.Path(tempfile.mkdtemp()); "
      "runner=root/'runner'; runner.write_bytes(b'exact runner'); "
      "identity=m.runner_identity(runner); "
      "assert identity == {'path': str(runner.resolve()), "
      "'sha256': m.sha256(runner)}; "
      "assert len(m.implementation_identity(m.repo_root())"
      "['runtime_tree_sha256']) == 64";
  CHECK(command_exit_code("python3 -B -c " + quote_arg_posix(program)) == 0);

  const std::string missing_program =
      "import importlib.util, pathlib, tempfile; "
      "spec=importlib.util.spec_from_file_location('personaplex_compare', " +
      module +
      "); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
      "m.runner_identity(pathlib.Path(tempfile.mkdtemp())/'missing')";
  CHECK(command_exit_code("python3 -B -c " +
                          quote_arg_posix(missing_program)) != 0);
}

TEST_CASE("personaplex compare fails closed when a child lane is killed") {
  const std::string module = quote_arg_posix(compare_script_path().string());
  const std::string program =
      "import importlib.util, pathlib, sys, tempfile; "
      "spec=importlib.util.spec_from_file_location('personaplex_compare', " +
      module +
      "); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
      "root=pathlib.Path(tempfile.mkdtemp()); marker=root/'attempts'; "
      "payload=\"import os,pathlib,signal,sys\\n"
      "p=pathlib.Path(sys.argv[1])\\n"
      "count=int(p.read_text())+1 if p.exists() else 1\\n"
      "p.write_text(str(count))\\n"
      "os.kill(os.getpid(), signal.SIGKILL)\"; "
      "failed=False\n"
      "try: m.run_lane([sys.executable,'-c',payload,str(marker)], "
      "root/'stdout',root/'stderr')\n"
      "except RuntimeError as error: failed='status -9' in str(error)\n"
      "assert failed; assert marker.read_text() == '1'; "
      "assert (root/'stdout').read_text() == ''";
  CHECK(command_exit_code("python3 -B -c " + quote_arg_posix(program)) == 0);
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
  CHECK(
      wrapper_source.find(
          "for candidate in python3.12 python3.13 /usr/bin/python3 python3") !=
      std::string::npos);
  CHECK(wrapper_source.find("import hashlib, json, subprocess") !=
        std::string::npos);
  CHECK(
      wrapper_source.find(
          "\"$PYTHON_BIN\" \"$ROOT_DIR/tools/bench/personaplex_compare.py\"") !=
      std::string::npos);
  CHECK(compare_source.find("for attempt in range(3)") == std::string::npos);
  CHECK(wrapper_source.find("--audio \"$AUDIO\"") != std::string::npos);
  CHECK(wrapper_source.find("--inference-config \"$INFERENCE_CONFIG\"") !=
        std::string::npos);
  CHECK(wrapper_source.find(
            "bash \"$ROOT_DIR/scripts/setup_moshi_cpp_reference.sh\"") !=
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
  CHECK(cmake_source.find("set(GGML_METAL OFF CACHE BOOL \"CPU-only benchmark "
                          "reference lanes\" FORCE)") != std::string::npos);
  CHECK(emel_source.find("#include <ggml") == std::string::npos);
  CHECK(emel_source.find(
            "predictor_initialize.sampling_seed = config.sampling_seed") !=
        std::string::npos);
  CHECK(emel_source.find("flush_steps == 0") != std::string::npos);
  CHECK(emel_source.find("sampling_seed = 1234") == std::string::npos);
  CHECK(emel_source.find("std::optional<emel::kernel::matmul::lane_pool>") !=
        std::string::npos);
  CHECK(emel_source.find("stage_worker_count") != std::string::npos);
  CHECK(emel_source.find("matmul_lane_count") != std::string::npos);
  CHECK(emel_source.find(
            "prediction_matmul_lanes.emplace(matmul_lane_count - 1u)") !=
        std::string::npos);
  CHECK(emel_source.find("active_worker_count()") != std::string::npos);
  CHECK(compare_source.find("emel_stage_workers") != std::string::npos);
  CHECK(compare_source.find("emel_matmul_lanes") != std::string::npos);
  CHECK(compare_source.find("emel_matmul_workers") != std::string::npos);
  CHECK(emel_source.find("accepted ?") == std::string::npos);
  CHECK(compare_source.find("reference_requested") != std::string::npos);
  CHECK(compare_source.find(
            "\"acceptance_scope\": \"bounded_quality_performance\"") !=
        std::string::npos);
  CHECK(compare_source.find("\"semantic_parity\": {") != std::string::npos);
  CHECK(compare_source.find("pass_bounded_quality_performance") !=
        std::string::npos);
  CHECK(compare_source.find("\"implementation\": implementation") !=
        std::string::npos);
  CHECK(compare_source.find("\"emel_runner\": runner") != std::string::npos);
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
  CHECK(setup_source.find("--quantize q8_0") != std::string::npos);
  CHECK(setup_source.find("manifest.get(\"quantization\") != \"q8_0\"") !=
        std::string::npos);
  CHECK(setup_source.find(
            "manifest.get(\"quantized_tensors\") != expected_tensors") !=
        std::string::npos);
  CHECK(setup_source.find(
            "manifest.get(\"enriched_sha256\") != digest.hexdigest()") !=
        std::string::npos);
  CHECK(setup_source.find("MIMI_PERSONAPLEX_Q8_TENSORS=68") !=
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

TEST_CASE("personaplex Q8 freshness reconverts a swapped F32 artifact") {
  const std::string setup = quote_arg_posix(
      (repo_root() / "scripts" / "setup_moshi_cpp_reference.sh").string());
  const std::string program =
      "import hashlib,json,pathlib,subprocess,tempfile; "
      "root=pathlib.Path(tempfile.mkdtemp()); "
      "output=root/'mimi.gguf'; manifest=root/'mimi.manifest.json'; "
      "source=root/'raw.gguf'; converter=root/'converter.py'; "
      "marker=root/'converter-runs'; source.write_bytes(b'raw'); "
      "candidate=b'candidate-q8-gguf'; output.write_bytes(b'legacy-f32-gguf'); "
      "manifest.write_text(json.dumps({'quantization':'q8_0',"
      "'quantized_tensors':68,'enriched_sha256':"
      "hashlib.sha256(candidate).hexdigest()})); "
      "converter.write_text(\"import hashlib,json,pathlib,sys\\n"
      "a=sys.argv\\n"
      "o=pathlib.Path(a[a.index('--output')+1])\\n"
      "m=pathlib.Path(a[a.index('--manifest')+1])\\n"
      "r=pathlib.Path(a[a.index('--marker')+1])\\n"
      "b=b'candidate-q8-gguf'\\n"
      "o.write_bytes(b)\\n"
      "m.write_text(json.dumps({'quantization':'q8_0',"
      "'quantized_tensors':68,'enriched_sha256':"
      "hashlib.sha256(b).hexdigest()}))\\n"
      "r.write_text((r.read_text() if r.exists() else '')+'x')\\n\"); "
      "cmd=['bash'," +
      setup +
      ",'--ensure-q8-artifact',str(output),str(manifest),'68',"
      "str(converter),str(source),'--marker',str(marker)]; "
      "assert subprocess.run(cmd).returncode == 0; "
      "assert output.read_bytes() == candidate; assert marker.read_text() == "
      "'x'; "
      "assert subprocess.run(cmd).returncode == 0; "
      "assert marker.read_text() == 'x'; "
      "data=json.loads(manifest.read_text()); data['quantized_tensors']=67; "
      "manifest.write_text(json.dumps(data)); "
      "assert subprocess.run(cmd).returncode == 0; "
      "assert marker.read_text() == 'xx'; "
      "assert json.loads(manifest.read_text())['quantized_tensors'] == 68";
  CHECK(command_exit_code("python3 -B -c " + quote_arg_posix(program)) == 0);
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

TEST_CASE("personaplex converter checks exact sparse depformer weights") {
  const std::string converter_source =
      read_file(repo_root() / "tools" / "bench" / "moshi_gguf_convert.py");
  CHECK(converter_source.find("required_depformer_indices = set(schedule)") !=
        std::string::npos);
  CHECK(converter_source.find(
            "present_depformer_names != required_depformer_names") !=
        std::string::npos);
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
