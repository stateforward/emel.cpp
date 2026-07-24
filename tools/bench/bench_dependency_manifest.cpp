#include "bench_dependency_manifest.hpp"

#include <array>
#include <fstream>
#include <sstream>

namespace emel::bench::dependency_manifest {

namespace {

using kind = dependency_kind;

constexpr std::array k_records{
    dependency_record{"all", kind::source, "tools/bench/CMakeLists.txt", "build_registration"},
    dependency_record{"all", kind::source, "tools/bench/bench_runner.cpp", "runner_dispatch"},
    dependency_record{"all", kind::source, "tools/bench/bench_main.cpp", "process_entrypoint"},
    dependency_record{"all",
                      kind::source,
                      "tools/bench/bench_runner_registry.cpp",
                      "runner_registration"},
    dependency_record{"all",
                      kind::source,
                      "tools/bench/bench_runner_contract.hpp",
                      "runner_contract"},
    dependency_record{"all", kind::source, "tools/bench/bench_cases.hpp", "case_contract"},
    dependency_record{"all", kind::source, "tools/bench/bench_common.hpp", "result_contract"},
    dependency_record{"all",
                      kind::source,
                      "tools/bench/bench_disabled_cases.cpp",
                      "filtered_suite_stubs"},
    dependency_record{"all",
                      kind::source,
                      "tools/bench/bench_dependency_manifest.cpp",
                      "manifest_registry"},
    dependency_record{"all",
                      kind::source,
                      "tools/bench/bench_dependency_manifest.hpp",
                      "manifest_registry"},
    dependency_record{"all", kind::config, "tools/bench/reference_ref.txt", "reference_revision"},
    dependency_record{"all", kind::script, "scripts/quality_gates.sh", "benchmark_gate_entrypoint"},

    dependency_record{"batch_planner",
                      kind::source,
                      "tools/bench/batch/planner_bench.cpp",
                      "suite_source"},
    dependency_record{"memory_kv",
                      kind::source,
                      "tools/bench/memory/kv_bench.cpp",
                      "suite_source"},
    dependency_record{"memory_kv", kind::source, "tools/bench/memory/bench_common.hpp", "common"},
    dependency_record{"memory_recurrent",
                      kind::source,
                      "tools/bench/memory/recurrent_bench.cpp",
                      "suite_source"},
    dependency_record{"memory_recurrent",
                      kind::source,
                      "tools/bench/memory/bench_common.hpp",
                      "common"},
    dependency_record{"memory_hybrid",
                      kind::source,
                      "tools/bench/memory/hybrid_bench.cpp",
                      "suite_source"},
    dependency_record{"memory_hybrid",
                      kind::source,
                      "tools/bench/memory/bench_common.hpp",
                      "common"},

    dependency_record{"jinja_parser",
                      kind::source,
                      "tools/bench/text/jinja/parser_bench.cpp",
                      "suite_source"},
    dependency_record{"jinja_parser", kind::fixture, "tests/text/jinja", "jinja_cases"},
    dependency_record{"jinja_formatter",
                      kind::source,
                      "tools/bench/text/jinja/formatter_bench.cpp",
                      "suite_source"},
    dependency_record{"jinja_formatter", kind::fixture, "tests/text/jinja", "jinja_cases"},

    dependency_record{"gbnf_rule_parser",
                      kind::source,
                      "tools/bench/gbnf/rule_parser_bench.cpp",
                      "suite_source"},
    dependency_record{"gbnf_rule_parser", kind::fixture, "tests/gbnf", "grammar_cases"},

    dependency_record{"generation",
                      kind::source,
                      "tools/bench/generation_bench.cpp",
                      "suite_source"},
    dependency_record{"generation",
                      kind::source,
                      "tools/bench/generation_compare_contract.hpp",
                      "compare_contract"},
    dependency_record{"generation",
                      kind::source,
                      "tools/bench/generation_workload_manifest.hpp",
                      "workload_manifest_parser"},
    dependency_record{"generation",
                      kind::source,
                      "tools/generation_fixture_registry.hpp",
                      "fixture_registry"},
    dependency_record{"generation",
                      kind::source,
                      "tools/generation_formatter_contract.hpp",
                      "formatter_contract"},
    dependency_record{"generation",
                      kind::source,
                      "tools/generation_route_policy.hpp",
                      "runtime_policy"},
    dependency_record{"generation",
                      kind::config,
                      "tools/bench/generation_variants",
                      "workload_manifests"},
    dependency_record{"generation",
                      kind::config,
                      "tools/bench/reference_backends/llama_cpp_generation.json",
                      "reference_backend"},
    dependency_record{"generation",
                      kind::fixture,
                      "tools/bench/generation_prompts",
                      "prompt_fixtures"},
    dependency_record{"generation", kind::model, "tests/models", "generation_model_fixtures"},
    dependency_record{"generation",
                      kind::snapshot,
                      "snapshots/bench/generation_pre_flash_baseline.txt",
                      "flash_baseline"},
    dependency_record{"generation",
                      kind::script,
                      "tools/bench/generation_compare.py",
                      "operator_compare"},
    dependency_record{"generation",
                      kind::script,
                      "tools/bench/compare_flash_baseline.py",
                      "flash_baseline_check"},

    dependency_record{"diarization_sortformer",
                      kind::source,
                      "tools/bench/diarization/sortformer_bench.cpp",
                      "suite_source"},
    dependency_record{"diarization_sortformer",
                      kind::source,
                      "tools/bench/diarization/sortformer_fixture.hpp",
                      "fixture_contract"},
    dependency_record{"diarization_sortformer",
                      kind::source,
                      "tools/bench/diarization_compare_contract.hpp",
                      "compare_contract"},
    dependency_record{"diarization_sortformer",
                      kind::fixture,
                      "tests/models/diar_streaming_sortformer_4spk-v2.1.gguf",
                      "maintained_model_fixture"},
    dependency_record{"diarization_sortformer",
                      kind::model,
                      "tests/models/diar_streaming_sortformer_4spk-v2.1.gguf",
                      "maintained_model"},
    dependency_record{"diarization_sortformer",
                      kind::script,
                      "tools/bench/diarization_compare.py",
                      "operator_compare"},
    dependency_record{"diarization_sortformer",
                      kind::script,
                      "tools/bench/diarization_sortformer_pytorch_reference.py",
                      "reference_lane"},

    dependency_record{"flash_attention",
                      kind::source,
                      "tools/bench/kernel/flash_attention_bench.cpp",
                      "suite_source"},
    dependency_record{"flash_attention",
                      kind::source,
                      "tools/bench/kernel/bench_common.hpp",
                      "common"},
    dependency_record{"flash_attention", kind::source, "src/emel/kernel", "kernel_inputs"},

    dependency_record{"logits_validator",
                      kind::source,
                      "tools/bench/logits/validator_bench.cpp",
                      "suite_source"},
    dependency_record{"logits_validator",
                      kind::source,
                      "tools/bench/logits/bench_common.hpp",
                      "common"},
    dependency_record{"logits_sampler",
                      kind::source,
                      "tools/bench/logits/sampler_bench.cpp",
                      "suite_source"},
    dependency_record{"logits_sampler",
                      kind::source,
                      "tools/bench/logits/bench_common.hpp",
                      "common"},

    dependency_record{"kernel_x86_64",
                      kind::source,
                      "tools/bench/kernel/x86_64_bench.cpp",
                      "suite_source"},
    dependency_record{"kernel_x86_64",
                      kind::source,
                      "tools/bench/kernel/bench_common.hpp",
                      "common"},
    dependency_record{"kernel_x86_64", kind::source, "src/emel/kernel", "kernel_inputs"},
    dependency_record{"kernel_aarch64",
                      kind::source,
                      "tools/bench/kernel/aarch64_bench.cpp",
                      "suite_source"},
    dependency_record{"kernel_aarch64",
                      kind::source,
                      "tools/bench/kernel/bench_common.hpp",
                      "common"},
    dependency_record{"kernel_aarch64", kind::source, "src/emel/kernel", "kernel_inputs"},

    dependency_record{"sm_any", kind::source, "tools/bench/sm_any_bench.cpp", "suite_source"},
    dependency_record{"sm_scheduler",
                      kind::source,
                      "tools/bench/sm_scheduler_bench.cpp",
                      "suite_source"},
    dependency_record{"sm_scheduler", kind::source, "src/emel/sm.hpp", "scheduler_policy"},
    dependency_record{"graph_processor",
                      kind::source,
                      "tools/bench/graph/processor_bench.cpp",
                      "suite_source"},
    dependency_record{"decode_wavefront",
                      kind::source,
                      "tools/bench/text/generator/decode_wavefront_bench.cpp",
                      "suite_source"},
    dependency_record{"decode_wavefront",
                      kind::source,
                      "src/emel/text/generator/decode_wavefront",
                      "wavefront_actor"},
    dependency_record{"parallel_matmul",
                      kind::source,
                      "tools/bench/text/generator/parallel_matmul_bench.cpp",
                      "suite_source"},
    dependency_record{"parallel_matmul",
                      kind::source,
                      "src/emel/kernel/matmul",
                      "parallel_matmul_actor"},
    dependency_record{"weight_streaming",
                      kind::source,
                      "tools/bench/model/tensor/window_streaming_bench.cpp",
                      "suite_source"},
    dependency_record{"weight_streaming",
                      kind::source,
                      "src/emel/model/tensor/window",
                      "window_actor"},
    // The bench constructs emel::io::mmap::sm directly for the emel_mmap
    // baseline and the streaming lane, so mmap regressions must gate this
    // suite.
    dependency_record{"weight_streaming",
                      kind::source,
                      "src/emel/io/mmap",
                      "mmap_actor"},
    dependency_record{"weight_streaming",
                      kind::source,
                      "scripts/bench.sh",
                      "memory_max_wrapper"},

    dependency_record{"tokenizer_preprocessor_bpe",
                      kind::source,
                      "tools/bench/text/tokenizer/preprocessor/bpe_bench.cpp",
                      "suite_source"},
    dependency_record{"tokenizer_preprocessor_spm",
                      kind::source,
                      "tools/bench/text/tokenizer/preprocessor/spm_bench.cpp",
                      "suite_source"},
    dependency_record{"tokenizer_preprocessor_ugm",
                      kind::source,
                      "tools/bench/text/tokenizer/preprocessor/ugm_bench.cpp",
                      "suite_source"},
    dependency_record{"tokenizer_preprocessor_wpm",
                      kind::source,
                      "tools/bench/text/tokenizer/preprocessor/wpm_bench.cpp",
                      "suite_source"},
    dependency_record{"tokenizer_preprocessor_rwkv",
                      kind::source,
                      "tools/bench/text/tokenizer/preprocessor/rwkv_bench.cpp",
                      "suite_source"},
    dependency_record{"tokenizer_preprocessor_plamo2",
                      kind::source,
                      "tools/bench/text/tokenizer/preprocessor/plamo2_bench.cpp",
                      "suite_source"},

    dependency_record{"encoder_bpe",
                      kind::source,
                      "tools/bench/text/encoders/bpe_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_bpe",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},
    dependency_record{"encoder_spm",
                      kind::source,
                      "tools/bench/text/encoders/spm_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_spm",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},
    dependency_record{"encoder_wpm",
                      kind::source,
                      "tools/bench/text/encoders/wpm_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_wpm",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},
    dependency_record{"encoder_ugm",
                      kind::source,
                      "tools/bench/text/encoders/ugm_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_ugm",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},
    dependency_record{"encoder_rwkv",
                      kind::source,
                      "tools/bench/text/encoders/rwkv_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_rwkv",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},
    dependency_record{"encoder_plamo2",
                      kind::source,
                      "tools/bench/text/encoders/plamo2_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_plamo2",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},
    dependency_record{"encoder_fallback",
                      kind::source,
                      "tools/bench/text/encoders/fallback_bench.cpp",
                      "suite_source"},
    dependency_record{"encoder_fallback",
                      kind::source,
                      "tools/bench/text/encoders/bench_common.hpp",
                      "common"},

    dependency_record{"tokenizer",
                      kind::source,
                      "tools/bench/text/tokenizer/tokenizer_bench.cpp",
                      "suite_source"},
    dependency_record{"tokenizer", kind::fixture, "tests/models/tokenizer-tiny.json", "fixture"},
    dependency_record{"tokenizer", kind::model, "tests/models", "tokenizer_model_inputs"},

    dependency_record{"speech_codec_mimi",
                      kind::source,
                      "tools/bench/speech/codec_mimi_bench.cpp",
                      "suite_source"},
    // The maintained Mimi codec runtime is what the parity lane exercises:
    // codec-only changes must select this suite, not skip it.
    dependency_record{"speech_codec_mimi",
                      kind::source,
                      "src/emel/speech/codec/mimi",
                      "maintained_runtime"},
    // The parity runner loads the enriched artifact through the Moshi model
    // binding (hparams + execution contract) before codec initialization, so
    // contract changes must gate the compare lanes too.
    dependency_record{"speech_codec_mimi",
                      kind::source,
                      "src/emel/model/moshi",
                      "model_binding"},
    dependency_record{"speech_codec_mimi",
                      kind::source,
                      "tools/bench/speech/mimi_emel_parity_runner.cpp",
                      "emel_lane_runner"},
    dependency_record{"speech_codec_mimi",
                      kind::source,
                      "tools/bench/speech/moshi_reference_driver.cpp",
                      "reference_lane_driver"},
    dependency_record{"speech_codec_mimi", kind::fixture, "tests/models/mimi-tiny.gguf", "fixture"},
    dependency_record{"speech_codec_mimi",
                      kind::script,
                      "tools/bench/mimi_compare.py",
                      "operator_compare"},
    dependency_record{"speech_codec_mimi",
                      kind::script,
                      "scripts/bench_mimi_compare.sh",
                      "compare_wrapper"},
    dependency_record{"speech_codec_mimi",
                      kind::script,
                      "tools/bench/moshi_gguf_convert.py",
                      "emel_lane_converter"},
    dependency_record{"speech_codec_mimi",
                      kind::config,
                      "tools/bench/personaplex-inference.json",
                      "inference_contract"},
    dependency_record{"speech_codec_mimi",
                      kind::config,
                      "tools/bench/moshi_reference_ref.txt",
                      "reference_pin"},
    dependency_record{"speech_codec_mimi",
                      kind::config,
                      "tools/bench/reference_backends/moshi_cpp_mimi.json",
                      "reference_backend_descriptor"},
    dependency_record{"speech_codec_mimi",
                      kind::script,
                      "scripts/setup_moshi_cpp_reference.sh",
                      "reference_setup"},
    dependency_record{"speech_lm_moshi",
                      kind::source,
                      "tools/bench/speech/lm_moshi_bench.cpp",
                      "suite_source"},
    dependency_record{"speech_lm_moshi",
                      kind::source,
                      "tools/bench/speech/personaplex_emel_runner.cpp",
                      "e2e_emel_lane_runner"},
    dependency_record{"speech_lm_moshi",
                      kind::source,
                      "src/emel/model/moshi",
                      "model_binding"},
    dependency_record{"speech_lm_moshi",
                      kind::script,
                      "scripts/bench_moshi_lm_compare.sh",
                      "compare_wrapper"},
    dependency_record{"speech_lm_moshi",
                      kind::script,
                      "scripts/setup_moshi_cpp_reference.sh",
                      "reference_setup"},
    dependency_record{"speech_lm_moshi",
                      kind::script,
                      "tools/bench/moshi_gguf_convert.py",
                      "emel_lane_converter"},
    dependency_record{"speech_lm_moshi",
                      kind::config,
                      "tools/bench/personaplex-inference.json",
                      "inference_contract"},
    // MLX-only inputs map to the MLX suite: quality_gates.sh runs the
    // personaplex-mlx compare lane only for speech_codec_mimi_mlx, so routing
    // these through the generic mimi suite would gate MLX regressions with
    // the moshi.cpp lane instead. The shared compare wrapper and comparator
    // carry MLX-specific branches, so they map to both suites.
    dependency_record{"speech_codec_mimi_mlx",
                      kind::script,
                      "tools/bench/mimi_compare.py",
                      "operator_compare"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::script,
                      "scripts/bench_mimi_compare.sh",
                      "compare_wrapper"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::script,
                      "scripts/setup_personaplex_mlx_reference.sh",
                      "reference_setup"},
    // The MLX setup script also runs this converter to build the enriched
    // EMEL artifact for the comparison, so converter-only patches must gate
    // the MLX lane too, not just the moshi.cpp lane.
    dependency_record{"speech_codec_mimi_mlx",
                      kind::script,
                      "tools/bench/moshi_gguf_convert.py",
                      "emel_lane_converter"},
    // The MLX comparison exercises the same maintained codec runtime through
    // the EMEL parity runner, so codec-only changes must gate this suite on
    // Apple Silicon too.
    dependency_record{"speech_codec_mimi_mlx",
                      kind::source,
                      "src/emel/speech/codec/mimi",
                      "maintained_runtime"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::source,
                      "src/emel/model/moshi",
                      "model_binding"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::source,
                      "tools/bench/speech/mimi_emel_parity_runner.cpp",
                      "emel_lane_runner"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::script,
                      "tools/bench/speech/personaplex_mlx_mimi_driver.py",
                      "reference_lane_driver"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::config,
                      "tools/bench/personaplex_mlx_ref.txt",
                      "reference_pin"},
    dependency_record{"speech_codec_mimi_mlx",
                      kind::config,
                      "tools/bench/reference_backends/personaplex_mlx_mimi.json",
                      "reference_backend_descriptor"},
};

}  // namespace

std::string_view kind_name(const dependency_kind kind_value) {
  switch (kind_value) {
    case dependency_kind::source:
      return "source";
    case dependency_kind::config:
      return "config";
    case dependency_kind::fixture:
      return "fixture";
    case dependency_kind::model:
      return "model";
    case dependency_kind::script:
      return "script";
    case dependency_kind::snapshot:
      return "snapshot";
    default:
      return "unknown";
  }
}

std::span<const dependency_record> records() {
  return k_records;
}

std::span<const dependency_record> records_for(const std::string_view runner) {
  std::size_t begin = k_records.size();
  std::size_t count = 0u;
  for (std::size_t i = 0u; i < k_records.size(); ++i) {
    if (k_records[i].runner != runner) {
      if (count != 0u) {
        break;
      }
      continue;
    }
    if (begin == k_records.size()) {
      begin = i;
    }
    ++count;
  }
  if (count == 0u) {
    return {};
  }
  return std::span<const dependency_record>(k_records.data() + begin, count);
}

bool requires_full_gate(const freshness_state state) {
  return state.missing || state.stale || state.uncertain;
}

freshness_state inspect(const std::filesystem::path & path, const bool uncertain) {
  freshness_state state{};
  state.uncertain = uncertain;

  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    state.missing = true;
    return state;
  }

  std::ostringstream current;
  current << input.rdbuf();
  if (!input.eof() && input.fail()) {
    state.uncertain = true;
    return state;
  }
  state.stale = current.str() != render();
  return state;
}

std::string render() {
  std::ostringstream out;
  out << k_schema << '\n';
  out << "full_gate_on=missing,stale,uncertain\n";
  for (const dependency_record & record : k_records) {
    out << "record"
        << " runner=" << record.runner
        << " kind=" << kind_name(record.kind)
        << " path=" << record.path
        << " reason=" << record.reason
        << '\n';
  }
  return out.str();
}

bool write(const std::filesystem::path & path) {
  std::ofstream output(path, std::ios::binary);
  if (!output.good()) {
    return false;
  }
  output << render();
  return output.good();
}

}  // namespace emel::bench::dependency_manifest
