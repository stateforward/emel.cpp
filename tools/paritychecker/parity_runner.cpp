#include "parity_runner.hpp"
#include "parity_assets.hpp"
#include "parity_engine.hpp"
#include "../generation_fixture_registry.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace emel::paritychecker {

namespace {

void print_usage(const char * exe) {
  std::fprintf(stderr,
               "usage: %s [--gbnf | --kernel | --jinja | --generation] [--model <path>] "
               "(--text <text> | --text-file <path>) "
               "[--max-tokens <count>] [--add-special] [--parse-special] [--dump] "
               "[--attribution] [--write-generation-baseline <path>]\n"
               "  default mode compares tokenizer parity and requires --model\n"
               "  --gbnf mode compares GBNF parser parity and ignores --model\n"
               "  --kernel mode compares kernel parity and ignores --model\n"
               "  --jinja mode compares jinja parser/formatter parity and ignores --model\n"
               "  --generation mode requires --model one maintained fixture under tests/models/ "
               "and prompt text; maintained baselines are append-only under snapshots/parity/\n",
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

bool parse_args(int argc, char ** argv, parity_options & out) {
  bool have_text = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--gbnf") {
      out.mode = parity_mode::gbnf_parser;
      continue;
    }
    if (arg == "--kernel") {
      out.mode = parity_mode::kernel;
      continue;
    }
    if (arg == "--jinja") {
      out.mode = parity_mode::jinja;
      continue;
    }
    if (arg == "--generation") {
      out.mode = parity_mode::generation;
      continue;
    }
    if (arg == "--model") {
      if (i + 1 >= argc) {
        return false;
      }
      out.model_path = argv[++i];
      continue;
    }
    if (arg == "--text") {
      if (i + 1 >= argc) {
        return false;
      }
      out.text = argv[++i];
      have_text = true;
      continue;
    }
    if (arg == "--text-file") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!load_text_file(argv[++i], out.text)) {
        return false;
      }
      have_text = true;
      continue;
    }
    if (arg == "--add-special") {
      out.add_special = true;
      continue;
    }
    if (arg == "--parse-special") {
      out.parse_special = true;
      continue;
    }
    if (arg == "--dump") {
      out.dump = true;
      continue;
    }
    if (arg == "--attribution") {
      out.attribution = true;
      continue;
    }
    if (arg == "--write-generation-baseline") {
      if (i + 1 >= argc) {
        return false;
      }
      out.write_generation_baseline_path = argv[++i];
      continue;
    }
    if (arg == "--max-tokens") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!parse_positive_i32(argv[++i], out.max_tokens)) {
        return false;
      }
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      return false;
    }
    return false;
  }
  if (!have_text && out.mode != parity_mode::kernel) {
    return false;
  }
  if (out.mode == parity_mode::tokenizer && out.model_path.empty()) {
    return false;
  }
  if (out.mode == parity_mode::generation && (out.model_path.empty() || !have_text)) {
    return false;
  }
  if (out.mode == parity_mode::gbnf_parser &&
      (out.add_special || out.parse_special || out.attribution)) {
    return false;
  }
  if (out.mode == parity_mode::jinja &&
      (out.add_special || out.parse_special || out.attribution)) {
    return false;
  }
  if (out.mode == parity_mode::generation && (out.add_special || out.parse_special)) {
    return false;
  }
  if (out.mode != parity_mode::generation && out.attribution) {
    return false;
  }
  if (out.mode != parity_mode::generation && !out.write_generation_baseline_path.empty()) {
    return false;
  }
  return true;
}

}  // namespace

int run_parity_cli(int argc, char ** argv) {
  parity_options opts;
  if (!parse_args(argc, argv, opts)) {
    print_usage(argv[0]);
    return 2;
  }

  return run_parity(opts);
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
