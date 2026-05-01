#include "parity_runner.hpp"
#include "parity_assets.hpp"
#include "parity_dependency_manifest.hpp"
#include "parity_engine.hpp"
#include "../generation_fixture_registry.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace emel::paritychecker {

namespace {

enum class manifest_operation {
  none,
  write,
  check,
};

struct cli_args {
  parity_options parity;
  manifest_operation manifest = manifest_operation::none;
  std::string manifest_path;
  bool manifest_uncertain = false;
};

void print_usage(const char * exe) {
  std::fprintf(stderr,
               "usage: %s [--gbnf | --kernel | --jinja | --generation] [--model <path>] "
               "(--text <text> | --text-file <path>) "
               "[--max-tokens <count>] [--add-special] [--parse-special] [--dump] "
               "[--attribution] [--write-generation-baseline <path>]\n"
               "       %s --write-dependency-manifest <path>\n"
               "       %s --check-dependency-manifest <path> "
               "[--dependency-manifest-uncertain]\n"
               "  default mode compares tokenizer parity and requires --model\n"
               "  --gbnf mode compares GBNF parser parity and ignores --model\n"
               "  --kernel mode compares kernel parity and ignores --model\n"
               "  --jinja mode compares jinja parser/formatter parity and ignores --model\n"
               "  --generation mode requires --model one maintained fixture under tests/models/ "
               "and prompt text; maintained baselines are append-only under snapshots/parity/\n"
               "  dependency manifest operations emit or check parity_dependency_manifest/v1\n",
               exe,
               exe,
               exe);
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    std::fprintf(stderr,
                 "    - %.*s\n",
                 static_cast<int>(fixture.fixture_rel.size()),
                 fixture.fixture_rel.data());
  }
}

bool parse_positive_i32(std::string_view text, int32_t & out) {
  if (text.empty()) {
    return false;
  }
  char buffer[32];
  if (text.size() >= sizeof(buffer)) {
    return false;
  }
  for (size_t i = 0; i < text.size(); ++i) {
    buffer[i] = text[i];
  }
  buffer[text.size()] = '\0';
  char * end = nullptr;
  const long parsed = std::strtol(buffer, &end, 10);
  if (end == buffer || *end != '\0') {
    return false;
  }
  if (parsed <= 0 || parsed > 0x7fffffffL) {
    return false;
  }
  out = static_cast<int32_t>(parsed);
  return true;
}

bool load_text_file(const char * path, std::string & out) {
  std::vector<uint8_t> bytes;
  if (!assets::read_file_bytes(path, bytes)) {
    return false;
  }
  out.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  return true;
}

bool parse_args(int argc, char ** argv, cli_args & out) {
  bool have_text = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--gbnf") {
      out.parity.mode = parity_mode::gbnf_parser;
      continue;
    }
    if (arg == "--kernel") {
      out.parity.mode = parity_mode::kernel;
      continue;
    }
    if (arg == "--jinja") {
      out.parity.mode = parity_mode::jinja;
      continue;
    }
    if (arg == "--generation") {
      out.parity.mode = parity_mode::generation;
      continue;
    }
    if (arg == "--model") {
      if (i + 1 >= argc) {
        return false;
      }
      out.parity.model_path = argv[++i];
      continue;
    }
    if (arg == "--text") {
      if (i + 1 >= argc) {
        return false;
      }
      out.parity.text = argv[++i];
      have_text = true;
      continue;
    }
    if (arg == "--text-file") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!load_text_file(argv[++i], out.parity.text)) {
        return false;
      }
      have_text = true;
      continue;
    }
    if (arg == "--add-special") {
      out.parity.add_special = true;
      continue;
    }
    if (arg == "--parse-special") {
      out.parity.parse_special = true;
      continue;
    }
    if (arg == "--dump") {
      out.parity.dump = true;
      continue;
    }
    if (arg == "--attribution") {
      out.parity.attribution = true;
      continue;
    }
    if (arg == "--write-generation-baseline") {
      if (i + 1 >= argc) {
        return false;
      }
      out.parity.write_generation_baseline_path = argv[++i];
      continue;
    }
    if (arg == "--write-dependency-manifest") {
      if (i + 1 >= argc || out.manifest != manifest_operation::none) {
        return false;
      }
      out.manifest = manifest_operation::write;
      out.manifest_path = argv[++i];
      continue;
    }
    if (arg == "--check-dependency-manifest") {
      if (i + 1 >= argc || out.manifest != manifest_operation::none) {
        return false;
      }
      out.manifest = manifest_operation::check;
      out.manifest_path = argv[++i];
      continue;
    }
    if (arg == "--dependency-manifest-uncertain") {
      out.manifest_uncertain = true;
      continue;
    }
    if (arg == "--max-tokens") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!parse_positive_i32(argv[++i], out.parity.max_tokens)) {
        return false;
      }
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      return false;
    }
    return false;
  }

  if (out.manifest != manifest_operation::none) {
    const bool has_parity_arg = have_text || !out.parity.model_path.empty() ||
                                out.parity.mode != parity_mode::tokenizer ||
                                out.parity.max_tokens != 32 || out.parity.add_special ||
                                out.parity.parse_special || out.parity.dump ||
                                out.parity.attribution ||
                                !out.parity.write_generation_baseline_path.empty();
    if (has_parity_arg || out.manifest_path.empty()) {
      return false;
    }
    if (out.manifest == manifest_operation::write && out.manifest_uncertain) {
      return false;
    }
    return true;
  }

  if (out.manifest_uncertain) {
    return false;
  }
  if (!have_text && out.parity.mode != parity_mode::kernel) {
    return false;
  }
  if (out.parity.mode == parity_mode::tokenizer && out.parity.model_path.empty()) {
    return false;
  }
  if (out.parity.mode == parity_mode::generation &&
      (out.parity.model_path.empty() || !have_text)) {
    return false;
  }
  if (out.parity.mode == parity_mode::gbnf_parser &&
      (out.parity.add_special || out.parity.parse_special || out.parity.attribution)) {
    return false;
  }
  if (out.parity.mode == parity_mode::jinja &&
      (out.parity.add_special || out.parity.parse_special || out.parity.attribution)) {
    return false;
  }
  if (out.parity.mode == parity_mode::generation &&
      (out.parity.add_special || out.parity.parse_special)) {
    return false;
  }
  if (out.parity.mode != parity_mode::generation && out.parity.attribution) {
    return false;
  }
  if (out.parity.mode != parity_mode::generation &&
      !out.parity.write_generation_baseline_path.empty()) {
    return false;
  }
  return true;
}

std::string freshness_reason(const dependency_manifest::freshness_state state) {
  if (!dependency_manifest::requires_full_gate(state)) {
    return "fresh";
  }

  std::string reason;
  if (state.missing) {
    reason += "missing";
  }
  if (state.stale) {
    if (!reason.empty()) {
      reason += ",";
    }
    reason += "stale";
  }
  if (state.uncertain) {
    if (!reason.empty()) {
      reason += ",";
    }
    reason += "uncertain";
  }
  return reason;
}

}  // namespace

int run_parity_cli(int argc, char ** argv) {
  cli_args args;
  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return 2;
  }

  if (args.manifest == manifest_operation::write) {
    if (!dependency_manifest::write(args.manifest_path)) {
      std::fprintf(stderr,
                   "error: failed to write dependency manifest: %s\n",
                   args.manifest_path.c_str());
      return 1;
    }
    std::printf("dependency_manifest: action=write schema=%.*s records=%zu path=%s\n",
                static_cast<int>(dependency_manifest::k_schema.size()),
                dependency_manifest::k_schema.data(),
                dependency_manifest::records().size(),
                args.manifest_path.c_str());
    return 0;
  }

  if (args.manifest == manifest_operation::check) {
    const dependency_manifest::freshness_state state =
        dependency_manifest::inspect(args.manifest_path, args.manifest_uncertain);
    const bool full_gate = dependency_manifest::requires_full_gate(state);
    const std::string reason = freshness_reason(state);
    std::printf("dependency_manifest: action=check schema=%.*s full_gate=%u reason=%s path=%s\n",
                static_cast<int>(dependency_manifest::k_schema.size()),
                dependency_manifest::k_schema.data(),
                full_gate ? 1u : 0u,
                reason.c_str(),
                args.manifest_path.c_str());
    return full_gate ? 3 : 0;
  }

  return run_parity(args.parity);
}

int run_parity(const parity_options & opts) {
  const engine_adapter * engine = find_engine(opts.mode);
  if (engine == nullptr || engine->run == nullptr) {
    std::fprintf(stderr, "unsupported parity mode: %u\n", static_cast<unsigned>(opts.mode));
    return 1;
  }
  return engine->run(opts);
}

}  // namespace emel::paritychecker
