#include "parity_dependency_manifest.hpp"

#include <array>
#include <fstream>
#include <sstream>

namespace emel::paritychecker::dependency_manifest {

namespace {

using kind = dependency_kind;
using mode = parity_mode;

constexpr std::array k_records{
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/CMakeLists.txt",
                      "build_registration"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/parity_runner.cpp",
                      "runner_dispatch"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/parity_engine.cpp",
                      "engine_registration"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/parity_engines.cpp",
                      "engine_lane_execution"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_parity_common.cpp",
                      "tokenizer_lane_common"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_spm_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_bpe_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_wpm_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_ugm_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_rwkv_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_plamo2_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::source,
                      "tools/paritychecker/tokenizer_fallback_parity.cpp",
                      "tokenizer_variant"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::config,
                      "tools/paritychecker/reference_ref.txt",
                      "reference_revision"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::fixture,
                      "tests/text/tokenizer/parity_texts",
                      "tokenizer_text_cases"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::model,
                      "tests/models",
                      "tokenizer_model_sweep"},
    dependency_record{mode::tokenizer,
                      "tokenizer",
                      kind::script,
                      "scripts/quality_gates.sh",
                      "parity_gate_entrypoint"},

    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::source,
                      "tools/paritychecker/CMakeLists.txt",
                      "build_registration"},
    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::source,
                      "tools/paritychecker/parity_runner.cpp",
                      "runner_dispatch"},
    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::source,
                      "tools/paritychecker/parity_engine.cpp",
                      "engine_registration"},
    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::source,
                      "tools/paritychecker/parity_engines.cpp",
                      "gbnf_lane_execution"},
    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::config,
                      "tools/paritychecker/reference_ref.txt",
                      "reference_revision"},
    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::fixture,
                      "tests/gbnf/parity_texts",
                      "gbnf_grammar_cases"},
    dependency_record{mode::gbnf_parser,
                      "gbnf_parser",
                      kind::script,
                      "scripts/quality_gates.sh",
                      "parity_gate_entrypoint"},

    dependency_record{mode::kernel,
                      "kernel",
                      kind::source,
                      "tools/paritychecker/CMakeLists.txt",
                      "build_registration"},
    dependency_record{mode::kernel,
                      "kernel",
                      kind::source,
                      "tools/paritychecker/parity_runner.cpp",
                      "runner_dispatch"},
    dependency_record{mode::kernel,
                      "kernel",
                      kind::source,
                      "tools/paritychecker/parity_engine.cpp",
                      "engine_registration"},
    dependency_record{mode::kernel,
                      "kernel",
                      kind::source,
                      "tools/paritychecker/parity_engines.cpp",
                      "kernel_lane_execution"},
    dependency_record{mode::kernel,
                      "kernel",
                      kind::source,
                      "src/emel/kernel",
                      "emel_kernel_inputs"},
    dependency_record{mode::kernel,
                      "kernel",
                      kind::config,
                      "tools/paritychecker/reference_ref.txt",
                      "reference_revision"},
    dependency_record{mode::kernel,
                      "kernel",
                      kind::script,
                      "scripts/quality_gates.sh",
                      "parity_gate_entrypoint"},

    dependency_record{mode::jinja,
                      "jinja",
                      kind::source,
                      "tools/paritychecker/CMakeLists.txt",
                      "build_registration"},
    dependency_record{mode::jinja,
                      "jinja",
                      kind::source,
                      "tools/paritychecker/parity_runner.cpp",
                      "runner_dispatch"},
    dependency_record{mode::jinja,
                      "jinja",
                      kind::source,
                      "tools/paritychecker/parity_engine.cpp",
                      "engine_registration"},
    dependency_record{mode::jinja,
                      "jinja",
                      kind::source,
                      "tools/paritychecker/parity_engines.cpp",
                      "jinja_lane_execution"},
    dependency_record{mode::jinja,
                      "jinja",
                      kind::config,
                      "tools/paritychecker/reference_ref.txt",
                      "reference_revision"},
    dependency_record{mode::jinja,
                      "jinja",
                      kind::fixture,
                      "tests/text/jinja/parity_texts",
                      "jinja_template_cases"},
    dependency_record{mode::jinja,
                      "jinja",
                      kind::script,
                      "scripts/quality_gates.sh",
                      "parity_gate_entrypoint"},

    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/paritychecker/CMakeLists.txt",
                      "build_registration"},
    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/paritychecker/parity_runner.cpp",
                      "runner_dispatch"},
    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/paritychecker/parity_engine.cpp",
                      "engine_registration"},
    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/paritychecker/parity_engines.cpp",
                      "generation_lane_execution"},
    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/paritychecker/parity_assets.cpp",
                      "maintained_fixture_resolution"},
    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/generation_fixture_registry.hpp",
                      "maintained_fixture_registration"},
    dependency_record{mode::generation,
                      "generation",
                      kind::source,
                      "tools/generation_formatter_contract.hpp",
                      "formatter_contract"},
    dependency_record{mode::generation,
                      "generation",
                      kind::config,
                      "tools/paritychecker/reference_ref.txt",
                      "reference_revision"},
    dependency_record{mode::generation,
                      "generation",
                      kind::fixture,
                      "tests/models",
                      "maintained_generation_fixture_paths"},
    dependency_record{mode::generation,
                      "generation",
                      kind::model,
                      "tests/models",
                      "maintained_generation_fixtures"},
    dependency_record{mode::generation,
                      "generation",
                      kind::snapshot,
                      "snapshots/parity",
                      "append_only_generation_baselines"},
    dependency_record{mode::generation,
                      "generation",
                      kind::script,
                      "scripts/quality_gates.sh",
                      "parity_gate_entrypoint"},
};

std::string_view mode_name(const parity_mode mode_value) {
  switch (mode_value) {
    case parity_mode::tokenizer:
      return "tokenizer";
    case parity_mode::gbnf_parser:
      return "gbnf_parser";
    case parity_mode::kernel:
      return "kernel";
    case parity_mode::jinja:
      return "jinja";
    case parity_mode::generation:
      return "generation";
    default:
      return "unknown";
  }
}

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

std::span<const dependency_record> records_for(const parity_mode mode_value) {
  size_t begin = k_records.size();
  size_t count = 0;
  for (size_t i = 0; i < k_records.size(); ++i) {
    if (k_records[i].mode != mode_value) {
      if (count != 0) {
        break;
      }
      continue;
    }
    if (begin == k_records.size()) {
      begin = i;
    }
    ++count;
  }
  if (count == 0) {
    return {};
  }
  return std::span<const dependency_record>(k_records.data() + begin, count);
}

bool requires_full_gate(const freshness_state state) {
  return state.missing || state.stale || state.uncertain;
}

std::string render() {
  std::ostringstream out;
  out << k_schema << '\n';
  out << "full_gate_on=missing,stale,uncertain\n";
  for (const dependency_record & record : k_records) {
    out << "record"
        << " runner=" << record.runner
        << " mode=" << mode_name(record.mode)
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

}  // namespace emel::paritychecker::dependency_manifest
