#include <cstdio>
#include <string>
#include <string_view>

#include "parity_runner.hpp"

namespace {

using emel::paritychecker::parity_options;

void print_usage(const char * exe) {
  std::fprintf(stderr,
               "usage: %s --model <path> (--text <text> | --text-file <path>) "
               "[--add-special] [--parse-special] [--dump]\n",
               exe);
}

bool load_text_file(const char * path, std::string & out) {
  std::FILE * file = std::fopen(path, "rb");
  if (file == nullptr) {
    return false;
  }
  std::string data;
  char buffer[4096];
  while (true) {
    const size_t read = std::fread(buffer, 1, sizeof(buffer), file);
    if (read == 0) {
      break;
    }
    data.append(buffer, read);
  }
  std::fclose(file);
  out = std::move(data);
  return true;
}

bool parse_args(int argc, char ** argv, parity_options & out) {
  bool have_text = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
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
      out.dump_tokens = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      return false;
    }
    return false;
  }
  if (out.model_path.empty() || !have_text) {
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char ** argv) {
  parity_options opts;
  if (!parse_args(argc, argv, opts)) {
    print_usage(argv[0]);
    return 2;
  }

  return emel::paritychecker::run_parity(opts);
}
